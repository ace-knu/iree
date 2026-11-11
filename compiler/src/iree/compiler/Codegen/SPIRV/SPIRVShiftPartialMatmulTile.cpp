// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- SPIRVShiftPartialMatmulTile.cpp ------------------------------------===//
//
// Makes partial workgroup tiles of a contraction static by shifting them back:
// in the last workgroup along a dimension whose extent E is not a multiple of
// the workgroup tile T, the tile starts at min(offset, E - T) instead of
// offset, so every workgroup computes a full, statically shaped T tile. The
// rows/columns the shifted tile shares with the previous workgroup are
// computed by both (same inputs and accumulation order, so the same values)
// into a fresh accumulator, and only the workgroup's own part of the result is
// inserted (computing in place in the output would race on the overlap). The
// extra work equals that of zero padding the tile, without materializing
// padded inputs.
//
// Why: the SPIR-V MatmulPromoteVectorize pipeline has no workgroup
// specialization, so a partial tile makes the contraction dynamically shaped in
// every workgroup. GPUTensorAlloc then does not promote it to workgroup memory,
// GenericVectorization (masking off) does not vectorize it and multi-buffering
// cannot allocate it. Zero padding (as in the LLVMGPU pipeline) does not fit
// here: the reduction tiling does not fuse tensor.pad producers and the thread
// tiling splits the pad into dynamic column copies.
//
// Runs right after workgroup tiling. Handles a contraction inside an
// scf.forall whose inputs and output init (through a linalg.fill) are
// tensor.extract_slice ops of statically shaped tensors; leaves anything else
// untouched.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/SPIRV/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#define DEBUG_TYPE "iree-spirv-shift-partial-matmul-tile"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_SPIRVSHIFTPARTIALMATMULTILEPASS
#include "iree/compiler/Codegen/SPIRV/Passes.h.inc"

namespace {

/// A loop dim of the contraction whose workgroup tile is partial.
struct ShiftedDim {
  OpFoldResult offset; // original start (workgroup offset)
  OpFoldResult size;   // original (dynamic) size
  Value shifted;       // min(offset, extent - tile)
  int64_t tile;
};

/// Rebuilds `slice` so that its dims mapped (by `map`) to shifted loop dims
/// start at the shifted offset with the static tile size. Fails if a mapped
/// dim does not have the original offset/size or another dim is dynamic.
static FailureOr<Value>
shiftSlice(RewriterBase &rewriter, tensor::ExtractSliceOp slice, AffineMap map,
           ArrayRef<std::optional<ShiftedDim>> shifted) {
  if (!slice.hasUnitStride() ||
      map.getNumResults() != slice.getType().getRank()) {
    return failure();
  }
  SmallVector<OpFoldResult> offsets = slice.getMixedOffsets();
  SmallVector<OpFoldResult> sizes = slice.getMixedSizes();
  // Rank-reducing slices are not handled.
  if (offsets.size() != map.getNumResults()) {
    return failure();
  }
  for (auto [operandDim, expr] : llvm::enumerate(map.getResults())) {
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    if (!dimExpr) {
      return failure();
    }
    const std::optional<ShiftedDim> &s = shifted[dimExpr.getPosition()];
    if (!s) {
      if (!getConstantIntValue(sizes[operandDim])) {
        return failure();
      }
      continue;
    }
    if (!isEqualConstantIntOrValue(offsets[operandDim], s->offset) ||
        !isEqualConstantIntOrValue(sizes[operandDim], s->size)) {
      return failure();
    }
    offsets[operandDim] = s->shifted;
    sizes[operandDim] = rewriter.getIndexAttr(s->tile);
  }
  rewriter.setInsertionPoint(slice);
  return tensor::ExtractSliceOp::create(rewriter, slice.getLoc(),
                                        slice.getSource(), offsets, sizes,
                                        slice.getMixedStrides())
      .getResult();
}

static LogicalResult shiftPartialTile(RewriterBase &rewriter,
                                      linalg::LinalgOp op) {
  if (op.getNumDpsInits() != 1 || op->getNumResults() != 1) {
    return failure();
  }
  SmallVector<int64_t> tiles = getLoweringConfig(op).getWorkgroupTileSizes();
  if (tiles.size() != op.getNumLoops()) {
    return failure();
  }
  // The output init: a linalg.fill of a slice of the workgroup's output.
  auto fill = op.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
  auto outSlice =
      fill ? fill.getOutputs()[0].getDefiningOp<tensor::ExtractSliceOp>()
           : tensor::ExtractSliceOp();
  if (!fill || !outSlice || !fill->hasOneUse()) {
    return failure();
  }
  RankedTensorType fullType = outSlice.getSourceType();
  if (!fullType.hasStaticShape()) {
    return failure();
  }
  AffineMap outMap = op.getMatchingIndexingMap(op.getDpsInitOperand(0));
  SmallVector<OpFoldResult> outOffsets = outSlice.getMixedOffsets();
  SmallVector<OpFoldResult> outSizes = outSlice.getMixedSizes();
  if (outOffsets.size() != outMap.getNumResults()) {
    return failure();
  }

  // The input slices; the shifted starts are created before the first of
  // them (and the output slice) so that they dominate all rebuilt slices.
  SmallVector<tensor::ExtractSliceOp> inputSlices;
  Operation *first = outSlice;
  for (OpOperand *operand : op.getDpsInputOperands()) {
    auto slice = operand->get().getDefiningOp<tensor::ExtractSliceOp>();
    if (!slice || !slice.getSourceType().hasStaticShape() ||
        slice->getBlock() != outSlice->getBlock()) {
      return failure();
    }
    inputSlices.push_back(slice);
    if (slice->isBeforeInBlock(first)) {
      first = slice;
    }
  }

  // A shifted start for every dynamically sized output dim.
  SmallVector<std::optional<ShiftedDim>> shifted(op.getNumLoops());
  bool any = false;
  for (auto [resultDim, expr] : llvm::enumerate(outMap.getResults())) {
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    if (!dimExpr) {
      return failure();
    }
    if (getConstantIntValue(outSizes[resultDim])) {
      continue;
    }
    int64_t loop = dimExpr.getPosition();
    int64_t tile = tiles[loop];
    int64_t extent = fullType.getDimSize(resultDim);
    if (tile <= 0 || tile > extent) {
      return failure();
    }
    rewriter.setInsertionPoint(first);
    Location loc = outSlice.getLoc();
    AffineExpr d0 = rewriter.getAffineDimExpr(0);
    AffineMap minMap = AffineMap::get(
        1, 0, {d0, rewriter.getAffineConstantExpr(extent - tile)},
        rewriter.getContext());
    Value offsetValue =
        getValueOrCreateConstantIndexOp(rewriter, loc, outOffsets[resultDim]);
    Value shiftedOffset = affine::AffineMinOp::create(rewriter, loc, minMap,
                                                      ValueRange{offsetValue});
    shifted[loop] = ShiftedDim{outOffsets[resultDim], outSizes[resultDim],
                               shiftedOffset, tile};
    any = true;
  }
  if (!any) {
    return failure();
  }

  // Shifted, static input slices and output init.
  SmallVector<Value> operands;
  for (auto [slice, operand] :
       llvm::zip_equal(inputSlices, op.getDpsInputOperands())) {
    FailureOr<Value> newSlice = shiftSlice(
        rewriter, slice, op.getMatchingIndexingMap(operand), shifted);
    if (failed(newSlice)) {
      return failure();
    }
    operands.push_back(*newSlice);
  }
  // A fresh, static accumulator rather than a slice of the shared output: the
  // shifted tile overlaps the previous workgroup's tile, and computing in
  // place in the output would race on the overlap.
  SmallVector<int64_t> tileShape;
  for (auto [resultDim, expr] : llvm::enumerate(outMap.getResults())) {
    const std::optional<ShiftedDim> &s =
        shifted[cast<AffineDimExpr>(expr).getPosition()];
    if (s) {
      tileShape.push_back(s->tile);
      continue;
    }
    std::optional<int64_t> size = getConstantIntValue(outSizes[resultDim]);
    if (!size) {
      return failure();
    }
    tileShape.push_back(*size);
  }
  rewriter.setInsertionPoint(fill);
  Value empty = tensor::EmptyOp::create(rewriter, fill.getLoc(), tileShape,
                                        fullType.getElementType());
  Value newInit = linalg::FillOp::create(rewriter, fill.getLoc(),
                                         fill.getInputs()[0], empty)
                      .getResult(0);
  operands.push_back(newInit);

  // Static contraction, then the workgroup's own part of the result.
  rewriter.setInsertionPoint(op);
  Operation *newOp =
      mlir::clone(rewriter, op.getOperation(), TypeRange{newInit.getType()}, operands);
  Location loc = op.getLoc();
  SmallVector<OpFoldResult> resOffsets(outMap.getNumResults(),
                                       rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> resSizes;
  SmallVector<OpFoldResult> resStrides(outMap.getNumResults(),
                                       rewriter.getIndexAttr(1));
  AffineExpr d0 = rewriter.getAffineDimExpr(0);
  AffineExpr d1 = rewriter.getAffineDimExpr(1);
  for (auto [resultDim, expr] : llvm::enumerate(outMap.getResults())) {
    const std::optional<ShiftedDim> &s =
        shifted[cast<AffineDimExpr>(expr).getPosition()];
    if (!s) {
      resSizes.push_back(outSizes[resultDim]);
      continue;
    }
    // Own part: from (offset - shifted), with the original size.
    Value offsetValue =
        getValueOrCreateConstantIndexOp(rewriter, loc, s->offset);
    resOffsets[resultDim] =
        affine::AffineApplyOp::create(
            rewriter, loc,
            AffineMap::get(2, 0, {d0 - d1}, rewriter.getContext()),
            ValueRange{offsetValue, s->shifted})
            .getResult();
    resSizes.push_back(s->size);
  }
  Value own = tensor::ExtractSliceOp::create(
                  rewriter, loc,
                  cast<RankedTensorType>(op->getResult(0).getType()),
                  newOp->getResult(0), resOffsets, resSizes, resStrides)
                  .getResult();
  rewriter.replaceOp(op, own);
  rewriter.eraseOp(fill);
  return success();
}

struct SPIRVShiftPartialMatmulTilePass final
    : impl::SPIRVShiftPartialMatmulTilePassBase<
          SPIRVShiftPartialMatmulTilePass> {
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    SmallVector<linalg::LinalgOp> candidates;
    funcOp.walk([&](linalg::LinalgOp op) {
      // Only partial (dynamically shaped) contraction tiles inside a workgroup
      // forall; divisible shapes are already static and left untouched.
      if (!linalg::isaContractionOpInterface(op) ||
          !op.hasPureTensorSemantics() || !op.hasDynamicShape() ||
          !op->getParentOfType<scf::ForallOp>() || !getLoweringConfig(op)) {
        return;
      }
      candidates.push_back(op);
    });
    IRRewriter rewriter(&getContext());
    for (linalg::LinalgOp op : candidates) {
      if (failed(shiftPartialTile(rewriter, op))) {
        // Not an error: the op keeps its dynamic shape and the pipeline
        // handles it as before (unpromoted, scalar).
        LLVM_DEBUG(llvm::dbgs() << "left as is: " << op << "\n");
      }
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
