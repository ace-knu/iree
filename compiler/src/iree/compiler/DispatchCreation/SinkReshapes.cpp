// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===--- SinkReshapes.cpp --- Pass to sink reshapes -----------------------===//
//
// This pass sinks reshapes (tensor.expand_shape/tensor.collapse_shape) that
// block producer-consumer fusion. These reshapes are generally produced by
// the `BubbleExpandShapes.cpp` pass that propagates reshapes towards the
// arguments, but get blocked on named op.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/DispatchCreation/FusionUtils.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-dispatch-creation-sink-reshapes"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_SINKRESHAPESPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

namespace {

struct SinkReshapesPass final : impl::SinkReshapesPassBase<SinkReshapesPass> {
  using Base::Base;
  void runOnOperation() override;
};

/// Returns true if two operations are fusable through tile and fuse. Ideally
/// this should use the same method as dispatch region formation where this
/// fusion analysis actually happens, but that requires a direct producer ->
/// consumer relationship and indexing maps for the right analysis. Here
/// we just approximate it (and try to be optimistic)
static bool isFusableUsingTileAndFuse(Operation *producer,
                                      Operation *consumer) {
  return isa_and_nonnull<IREE::LinalgExt::LinalgFusionOpInterface,
                         linalg::LinalgOp, linalg::UnPackOp,
                         IREE::Encoding::UnsetEncodingOp>(producer);
}

/// Control function to check if a `tensor.collapse_shape` (which is the
/// consumer of `opOperand`) should be bubbled through the `genericOp`.
static bool shouldBubbleCollapseShapeOp(tensor::CollapseShapeOp collapseOp,
                                        OpOperand *opOperand) {
  auto *producer = opOperand->get().getDefiningOp();
  if (!producer) {
    return false;
  }
  return IREE::Flow::isCloneableIntoDispatchOp(
      opOperand->get().getDefiningOp());
}

/// Control function to check if a `tensor.expand_shape` (which is producer of
/// `opOperand`) should be pushed past the `genericOp` (which is the consumer of
/// `opOperand`).
static bool shouldSinkExpandShapeOp(tensor::ExpandShapeOp expandOp,
                                    OpOperand *opOperand) {
  Operation *consumer = opOperand->getOwner();
  if (!IREE::Flow::isNonNullAndOutsideDispatch({expandOp, consumer})) {
    return false;
  }
  auto consumerGenericOp = dyn_cast<linalg::GenericOp>(consumer);
  if (!consumerGenericOp) {
    return false;
  }
  // Only sink across parallel generic ops for now.
  if (consumerGenericOp.getNumParallelLoops() !=
      consumerGenericOp.getNumLoops()) {
    return false;
  }

  // Do not sink reshapes across dequantize operations since they are
  // cloned into their consumers.
  if (IREE::LinalgExt::isBitExtendOp(consumer)) {
    return false;
  }

  // First check that the expand_shape producer and consumer can be fused.
  Operation *reshapeProducer = expandOp.getSrc().getDefiningOp();
  if (!reshapeProducer) {
    return false;
  }
  if (!isFusableUsingTileAndFuse(expandOp.getSrc().getDefiningOp(), consumer)) {
    return false;
  }

  // If the op is already fusable with producer using tile and fuse,
  // do nothing.
  for (OpOperand &opOperand : consumer->getOpOperands()) {
    Operation *currProducer = opOperand.get().getDefiningOp();
    if (!currProducer) {
      continue;
    }

    // The check for the producer having a single use is not fully
    // worked out. Ideally we can fuse with a producer irrespective
    // of number of uses, but is a good thumb rule in practice.
    if (!llvm::hasSingleElement(currProducer->getUses())) {
      continue;
    }

    // Check if a producer can already be tiled and fused with the consumer.
    if (!isFusableUsingTileAndFuse(currProducer, consumer)) {
      continue;
    }

    // There is already a tile-and-fusable producer to fuse with. Still prefer
    // fusing with the producer whose parallel iteration space rank matches
    // the consumer parallel iteration space rank to avoid loss of parallelism.
    if (auto currLinalgProducer = dyn_cast<linalg::LinalgOp>(currProducer)) {
      auto reshapeLinalgProducer = dyn_cast<linalg::LinalgOp>(reshapeProducer);
      if (!reshapeLinalgProducer) {
        // For now we will prefer to fold with Linalg op. So if the reshape
        // producer is not a Linalg op, bail.
        return false;
      }

      // Somehow this logic does not seem to work well when the reshape producer
      // is an elementwise operation. For one, should never have a reshape
      // "after" an elementwise operation, since bubble expand shape should
      // already account for it, and fuse the elementwise producer of reshape
      // and the consumer (which is also elementwise). Needs more investigation
      // but removes regressions and lit test failures.
      if (reshapeLinalgProducer.getNumLoops() ==
              reshapeLinalgProducer.getNumParallelLoops() &&
          currLinalgProducer.getNumLoops() !=
              currLinalgProducer.getNumParallelLoops()) {
        return false;
      }

      unsigned currConsumerNumParallelLoops =
          consumerGenericOp.getNumParallelLoops();
      unsigned currProducerNumParallelLoops =
          currLinalgProducer.getNumParallelLoops();
      if (currProducerNumParallelLoops == currConsumerNumParallelLoops) {
        // If the producer has same number of parallel loops as consumer,
        // then this is the operand to fuse along. So do nothing.
        return false;
      }
      // If the producer has less number of parallel loops as the consumer,
      // ignore this operand.
      if (currProducerNumParallelLoops < currConsumerNumParallelLoops) {
        continue;
      }
      unsigned reshapeProducerNumParallelLoops =
          reshapeLinalgProducer.getNumParallelLoops();
      if (currProducerNumParallelLoops < reshapeProducerNumParallelLoops) {
        return false;
      }
    }
  }
  return true;
}

/// Sinks a `collapse_shape` that only drops unit dimensions below the
/// `tensor.insert_slice` that writes its result into a filled tensor:
///
///   insert_slice(collapse_shape(x) into fill(c, empty))
///     -> collapse_shape(insert_slice(x into fill(c, expanded empty)))
///
/// This is how a K-axis bias fold widens an activation whose producer yields a
/// unit leading dimension. Dispatch formation fuses an insert-into-fill with its
/// producer only when the inserted value is that producer's result itself, so
/// with the collapse in between the widening becomes a copy dispatch of its own.
/// With the collapse sunk, the producer writes the widened buffer in place.
struct SinkUnitCollapseBelowInsertIntoFill final
    : OpRewritePattern<tensor::InsertSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp insertOp,
                                PatternRewriter &rewriter) const override {
    auto collapseOp =
        insertOp.getSource().getDefiningOp<tensor::CollapseShapeOp>();
    // Dispatch formation fuses the insert with its producer only when the
    // inserted value has no other use, so sinking is only worth it then;
    // otherwise it would merely reshape a widening that already fuses.
    if (!collapseOp || !collapseOp->hasOneUse() ||
        !collapseOp.getSrc().hasOneUse()) {
      return failure();
    }
    // The fill may be shared -- CSE merges the identical constant fills of
    // every layer's widening into one. A fresh fill is built below either way.
    auto fillOp = insertOp.getDest().getDefiningOp<linalg::FillOp>();
    if (!fillOp || !fillOp.getOutputs()[0].getDefiningOp<tensor::EmptyOp>()) {
      return failure();
    }
    RankedTensorType srcType = collapseOp.getSrcType();
    RankedTensorType destType = insertOp.getDestType();
    if (!srcType.hasStaticShape() || !destType.hasStaticShape() ||
        insertOp.getSourceType() != collapseOp.getResultType()) {
      return failure();
    }
    ArrayRef<int64_t> offsets = insertOp.getStaticOffsets();
    ArrayRef<int64_t> sizes = insertOp.getStaticSizes();
    ArrayRef<int64_t> strides = insertOp.getStaticStrides();
    if (ShapedType::isDynamicShape(offsets) ||
        ArrayRef<int64_t>(sizes) != collapseOp.getResultType().getShape() ||
        llvm::any_of(strides, [](int64_t s) { return s != 1; })) {
      return failure();
    }

    // Each collapsed dimension merges a group of `x`'s dimensions of which at
    // most one is not a unit; that one takes the destination's extent and the
    // insert offset, the unit ones stay 1 at offset 0.
    SmallVector<int64_t> expandedShape, newOffsets;
    for (auto [destDim, group] :
         llvm::enumerate(collapseOp.getReassociationIndices())) {
      int64_t carrier = group.back();
      int64_t nonUnit = 0;
      for (int64_t dim : group) {
        if (srcType.getDimSize(dim) != 1) {
          carrier = dim;
          ++nonUnit;
        }
      }
      if (nonUnit > 1) {
        return failure();
      }
      for (int64_t dim : group) {
        bool isCarrier = dim == carrier;
        expandedShape.push_back(isCarrier ? destType.getDimSize(destDim) : 1);
        newOffsets.push_back(isCarrier ? offsets[destDim] : 0);
      }
    }

    Location loc = insertOp.getLoc();
    Value empty = tensor::EmptyOp::create(rewriter, loc, expandedShape,
                                          destType.getElementType());
    Value fill = linalg::FillOp::create(rewriter, loc, fillOp.getInputs(),
                                        ValueRange{empty})
                     .getResult(0);
    SmallVector<OpFoldResult> offsetValues =
        getAsIndexOpFoldResult(rewriter.getContext(), newOffsets);
    SmallVector<OpFoldResult> sizeValues =
        getAsIndexOpFoldResult(rewriter.getContext(), srcType.getShape());
    SmallVector<OpFoldResult> strideValues(srcType.getRank(),
                                           rewriter.getIndexAttr(1));
    Value inserted = tensor::InsertSliceOp::create(
        rewriter, loc, collapseOp.getSrc(), fill, offsetValues, sizeValues,
        strideValues);
    rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
        insertOp, destType, inserted, collapseOp.getReassociationIndices());
    return success();
  }
};

void SinkReshapesPass::runOnOperation() {
  MLIRContext *context = &getContext();

  RewritePatternSet sinkReshapePatterns(context);

  auto collapsingControlFn = [](OpOperand *opOperand) {
    auto collapseOp =
        dyn_cast_if_present<tensor::CollapseShapeOp>(opOperand->getOwner());
    if (collapseOp) {
      return shouldBubbleCollapseShapeOp(collapseOp, opOperand);
    }

    auto expandOp =
        dyn_cast<tensor::ExpandShapeOp>(opOperand->get().getDefiningOp());
    if (expandOp) {
      return shouldSinkExpandShapeOp(expandOp, opOperand);
    }
    llvm_unreachable("reshape is neither a collapse or expand op");
  };
  linalg::populateFoldReshapeOpsByCollapsingPatterns(sinkReshapePatterns,
                                                     collapsingControlFn);
  // Add patterns to fold `tensor.empty` and reshape ops.
  tensor::populateFoldTensorEmptyPatterns(sinkReshapePatterns);
  memref::populateResolveRankedShapedTypeResultDimsPatterns(
      sinkReshapePatterns);
  tensor::ExpandShapeOp::getCanonicalizationPatterns(sinkReshapePatterns,
                                                     context);
  tensor::CollapseShapeOp::getCanonicalizationPatterns(sinkReshapePatterns,
                                                       context);
  if (failed(applyPatternsGreedily(getOperation(),
                                   std::move(sinkReshapePatterns)))) {
    getOperation()->emitOpError("failed to sink reshape ops");
    return signalPassFailure();
  }

  // Separately and last: the patterns above move collapses the other way (up
  // through fills), and mixed into the same driver the two settle on different
  // shapes for widenings that already fuse. Run alone, with no folding, this
  // only touches the insert-into-fill widenings that still have a collapse
  // between them and their producer.
  RewritePatternSet widenPatterns(context);
  widenPatterns.add<SinkUnitCollapseBelowInsertIntoFill>(context);
  GreedyRewriteConfig config;
  config.enableFolding(false).enableConstantCSE(false);
  config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
  if (failed(applyPatternsGreedily(getOperation(), std::move(widenPatterns),
                                   config))) {
    getOperation()->emitOpError("failed to sink unit collapses below inserts");
    return signalPassFailure();
  }
}

} // namespace

} // namespace mlir::iree_compiler::DispatchCreation
