// RUN: iree-opt --split-input-file --iree-dispatch-creation-pipeline --iree-flow-transformation-pipeline %s | FileCheck %s
// RUN: iree-opt --split-input-file --iree-flow-enable-executable-deduplication=false --iree-dispatch-creation-pipeline --iree-flow-transformation-pipeline %s | FileCheck %s --check-prefix=NODEDUP

// `--iree-flow-enable-executable-deduplication` controls whether the
// deduplication pass is placed in the flow pipeline at all, so it cannot be
// observed by running that pass directly (see deduplicate_executables.mlir);
// this exercises the pipeline instead.
//
// Two identical elementwise ops become two dispatches of the same shape. By
// default they collapse onto one executable. With deduplication disabled each
// keeps its own, which is what a backend needs when a per-dispatch constant
// offset has to stay static in the executable.

util.func public @two_identical_dispatches(%arg0: tensor<4xf32>,
    %arg1: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %empty = tensor.empty() : tensor<4xf32>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%arg0 : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %m = arith.mulf %in, %in : f32
      linalg.yield %m : f32
  } -> tensor<4xf32>
  %1 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%arg1 : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %m = arith.mulf %in, %in : f32
      linalg.yield %m : f32
  } -> tensor<4xf32>
  util.return %0, %1 : tensor<4xf32>, tensor<4xf32>
}

// Enabled (the default): one executable, dispatched twice.
//      CHECK: flow.executable private @two_identical_dispatches_dispatch_0
//  CHECK-NOT: flow.executable private @two_identical_dispatches_dispatch_1
//      CHECK: util.func public @two_identical_dispatches
//      CHECK:   flow.dispatch @two_identical_dispatches_dispatch_0
//      CHECK:   flow.dispatch @two_identical_dispatches_dispatch_0

// Disabled: both executables survive and each dispatch keeps its own.
//      NODEDUP: flow.executable private @two_identical_dispatches_dispatch_0
//      NODEDUP: flow.executable private @two_identical_dispatches_dispatch_1
//      NODEDUP: util.func public @two_identical_dispatches
//      NODEDUP:   flow.dispatch @two_identical_dispatches_dispatch_0
//      NODEDUP:   flow.dispatch @two_identical_dispatches_dispatch_1
