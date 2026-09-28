// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE                                                             \
  "iree-dispatch-creation-clone-producers-into-dispatch-regions"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_CLONEPRODUCERSINTODISPATCHREGIONSPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

namespace {

/// Folds `collapse_shape(extract_slice(expand_shape(x)))` into a slice of `x`
/// when the expand only inserts unit dimensions and the slice leaves them
/// alone.
///
/// Unit-dim folding leaves this chain around a slice that reads part of a wider
/// producer -- the unpadded columns of a K-axis bias fold's widened activation,
/// for example. The reshapes are not cloneable into a dispatch, so the slice
/// is stranded between them and becomes a copy dispatch of its own. As a plain
/// slice of `x` it is cloned into its consumer, which then reads the sub-range
/// in place.
///
/// It runs here, once the dispatch regions exist, and not before: while `x` is
/// still the `tensor.insert_slice` that widens the producer's result, a slice
/// of it folds straight back to that result, the producer is left with two
/// consumers again, and the widening can no longer be written in place.
struct FoldCollapseOfSliceOfUnitExpand final
    : OpRewritePattern<tensor::CollapseShapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::CollapseShapeOp collapseOp,
                                PatternRewriter &rewriter) const override {
    auto sliceOp = collapseOp.getSrc().getDefiningOp<tensor::ExtractSliceOp>();
    if (!sliceOp || !sliceOp->hasOneUse()) {
      return failure();
    }
    auto expandOp = sliceOp.getSource().getDefiningOp<tensor::ExpandShapeOp>();
    if (!expandOp) {
      return failure();
    }
    RankedTensorType srcType = expandOp.getSrcType();
    RankedTensorType expandedType = expandOp.getResultType();
    RankedTensorType resultType = collapseOp.getResultType();
    if (!srcType.hasStaticShape() || !expandedType.hasStaticShape() ||
        !resultType.hasStaticShape() ||
        resultType.getRank() != srcType.getRank()) {
      return failure();
    }
    ArrayRef<int64_t> offsets = sliceOp.getStaticOffsets();
    ArrayRef<int64_t> sizes = sliceOp.getStaticSizes();
    ArrayRef<int64_t> strides = sliceOp.getStaticStrides();
    if (ShapedType::isDynamicShape(offsets) ||
        ShapedType::isDynamicShape(sizes) ||
        llvm::any_of(strides, [](int64_t s) { return s != 1; })) {
      return failure();
    }

    // Each source dimension expands into a group in which at most one
    // dimension is not a unit; that one carries the source dimension. The
    // slice must take the whole of every other dimension in the group.
    SmallVector<int64_t> newOffsets, newSizes;
    for (ReassociationIndicesRef group : expandOp.getReassociationIndices()) {
      int64_t carrier = group.back();
      int64_t nonUnit = 0;
      for (int64_t dim : group) {
        if (expandedType.getDimSize(dim) != 1) {
          carrier = dim;
          ++nonUnit;
        }
      }
      if (nonUnit > 1) {
        return failure();
      }
      for (int64_t dim : group) {
        if (dim != carrier && (offsets[dim] != 0 || sizes[dim] != 1)) {
          return failure();
        }
      }
      newOffsets.push_back(offsets[carrier]);
      newSizes.push_back(sizes[carrier]);
    }
    // Only unit dimensions were added and removed, so the elements are in the
    // same order; the shapes agreeing is what makes the two the same tensor.
    if (ArrayRef<int64_t>(newSizes) != resultType.getShape()) {
      return failure();
    }

    SmallVector<OpFoldResult> offsetValues =
        getAsIndexOpFoldResult(rewriter.getContext(), newOffsets);
    SmallVector<OpFoldResult> sizeValues =
        getAsIndexOpFoldResult(rewriter.getContext(), newSizes);
    SmallVector<OpFoldResult> strideValues(newSizes.size(),
                                           rewriter.getIndexAttr(1));
    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        collapseOp, resultType, expandOp.getSrc(), offsetValues, sizeValues,
        strideValues);
    return success();
  }
};


struct CloneProducersIntoDispatchRegionsPass final
    : impl::CloneProducersIntoDispatchRegionsPassBase<
          CloneProducersIntoDispatchRegionsPass> {
  using Base::Base;
  void runOnOperation() override {
    mlir::FunctionOpInterface funcOp = getOperation();
    IRRewriter rewriter(funcOp->getContext());

    {
      RewritePatternSet patterns(funcOp->getContext());
      patterns.add<FoldCollapseOfSliceOfUnitExpand>(funcOp->getContext());
      // Only this rewrite: folding or constant CSE here would reshape the
      // dispatch regions that were just formed.
      GreedyRewriteConfig config;
      config.enableFolding(false).enableConstantCSE(false);
      config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns), config))) {
        return signalPassFailure();
      }
    }

    IREE::Flow::CloneableIntoDispatchOptions options;
    options.aggressive = aggressive;
    funcOp->walk([&](IREE::Flow::DispatchRegionOp regionOp) {
      if (failed(cloneProducersToRegion(rewriter, regionOp, options))) {
        return signalPassFailure();
      }
    });

    funcOp->walk<WalkOrder::PostOrder, ReverseIterator>([&](Operation *op) {
      if (isOpTriviallyDead(op)) {
        return rewriter.eraseOp(op);
      }
    });

    funcOp->walk([&](Operation *op) {
      if (!IREE::Flow::isNonNullAndOutsideDispatch(op) ||
          !isa<linalg::GenericOp, IREE::LinalgExt::GatherOp>(op)) {
        return;
      }
      if (failed(IREE::Flow::wrapOpInDispatchRegion(rewriter, op))) {
        return signalPassFailure();
      }
    });

    // Rerun the cloning again to move still cloneable operations into
    // dispatches.
    funcOp->walk([&](IREE::Flow::DispatchRegionOp regionOp) {
      if (failed(cloneProducersToRegion(rewriter, regionOp, options))) {
        return signalPassFailure();
      }
    });
  }
};

} // namespace

} // namespace mlir::iree_compiler::DispatchCreation
