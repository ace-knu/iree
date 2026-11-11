// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(func.func(iree-spirv-shift-partial-matmul-tile))' %s | FileCheck %s

// A 289x192x1024 matmul tiled to 64x64 workgroups (research/inception-opt
// partial M tiles): the last workgroup row has 33 rows. Its tile is shifted
// back to start at min(offset, 289 - 64) = 225, so the contraction and its
// fill are static 64x64 (into a fresh accumulator: computing in place in the
// output would race on the rows shared with the previous workgroup), and only
// the workgroup's own rows (from offset - shifted) are inserted.
#config = #iree_codegen.lowering_config<tile_sizes = [[64, 64, 16]]>
#map_a = affine_map<(d0, d1, d2) -> (d0, d2)>
#map_b = affine_map<(d0, d1, d2) -> (d2, d1)>
#map_c = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @partial_m_tile(%lhs: tensor<289x1024xf32>, %rhs: tensor<1024x192xf32>) -> tensor<289x192xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<289x192xf32>
  %r = scf.forall (%i, %j) = (0, 0) to (289, 192) step (64, 64) shared_outs(%out = %e) -> (tensor<289x192xf32>) {
    %m = affine.min affine_map<(d0) -> (-d0 + 289, 64)>(%i)
    %a = tensor.extract_slice %lhs[%i, 0] [%m, 1024] [1, 1] : tensor<289x1024xf32> to tensor<?x1024xf32>
    %b = tensor.extract_slice %rhs[0, %j] [1024, 64] [1, 1] : tensor<1024x192xf32> to tensor<1024x64xf32>
    %o = tensor.extract_slice %out[%i, %j] [%m, 64] [1, 1] : tensor<289x192xf32> to tensor<?x64xf32>
    %f = linalg.fill ins(%cst : f32) outs(%o : tensor<?x64xf32>) -> tensor<?x64xf32>
    %mm = linalg.generic {indexing_maps = [#map_a, #map_b, #map_c], iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%a, %b : tensor<?x1024xf32>, tensor<1024x64xf32>) outs(%f : tensor<?x64xf32>)
        attrs = {lowering_config = #config} {
    ^bb0(%x: f32, %y: f32, %acc: f32):
      %0 = arith.mulf %x, %y : f32
      %1 = arith.addf %acc, %0 : f32
      linalg.yield %1 : f32
    } -> tensor<?x64xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %mm into %out[%i, %j] [%m, 64] [1, 1] : tensor<?x64xf32> into tensor<289x192xf32>
    }
  }
  return %r : tensor<289x192xf32>
}
// CHECK-DAG: #[[$SHIFT:.+]] = affine_map<(d0) -> (d0, 225)>
// CHECK-DAG: #[[$OWN:.+]] = affine_map<(d0, d1) -> (d0 - d1)>
// CHECK-LABEL: func.func @partial_m_tile
// CHECK:         scf.forall (%[[I:.+]], %[[J:.+]]) =
// CHECK:           %[[M:.+]] = affine.min
// CHECK:           %[[SH:.+]] = affine.min #[[$SHIFT]](%[[I]])
// CHECK:           %[[A:.+]] = tensor.extract_slice %{{.+}}[%[[SH]], 0] [64, 1024] [1, 1] : tensor<289x1024xf32> to tensor<64x1024xf32>
// CHECK:           %[[E:.+]] = tensor.empty() : tensor<64x64xf32>
// CHECK:           %[[F:.+]] = linalg.fill ins(%{{.+}} : f32) outs(%[[E]] : tensor<64x64xf32>)
// CHECK:           %[[MM:.+]] = linalg.generic
// CHECK-SAME:        ins(%[[A]], %{{.+}} : tensor<64x1024xf32>, tensor<1024x64xf32>) outs(%[[F]] : tensor<64x64xf32>)
// CHECK-SAME:        lowering_config = #config
// CHECK:           %[[START:.+]] = affine.apply #[[$OWN]](%[[I]], %[[SH]])
// CHECK:           %[[OWN_PART:.+]] = tensor.extract_slice %[[MM]][%[[START]], 0] [%[[M]], 64] [1, 1] : tensor<64x64xf32> to tensor<?x64xf32>
// CHECK:           tensor.parallel_insert_slice %[[OWN_PART]] into %{{.+}}[%[[I]], %[[J]]] [%[[M]], 64]

// -----

// A divisible (static) tile is left untouched.
#config = #iree_codegen.lowering_config<tile_sizes = [[64, 64, 16]]>
#map_a = affine_map<(d0, d1, d2) -> (d0, d2)>
#map_b = affine_map<(d0, d1, d2) -> (d2, d1)>
#map_c = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @static_tile(%lhs: tensor<256x1024xf32>, %rhs: tensor<1024x192xf32>) -> tensor<256x192xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<256x192xf32>
  %r = scf.forall (%i, %j) = (0, 0) to (256, 192) step (64, 64) shared_outs(%out = %e) -> (tensor<256x192xf32>) {
    %a = tensor.extract_slice %lhs[%i, 0] [64, 1024] [1, 1] : tensor<256x1024xf32> to tensor<64x1024xf32>
    %b = tensor.extract_slice %rhs[0, %j] [1024, 64] [1, 1] : tensor<1024x192xf32> to tensor<1024x64xf32>
    %o = tensor.extract_slice %out[%i, %j] [64, 64] [1, 1] : tensor<256x192xf32> to tensor<64x64xf32>
    %f = linalg.fill ins(%cst : f32) outs(%o : tensor<64x64xf32>) -> tensor<64x64xf32>
    %mm = linalg.generic {indexing_maps = [#map_a, #map_b, #map_c], iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%a, %b : tensor<64x1024xf32>, tensor<1024x64xf32>) outs(%f : tensor<64x64xf32>)
        attrs = {lowering_config = #config} {
    ^bb0(%x: f32, %y: f32, %acc: f32):
      %0 = arith.mulf %x, %y : f32
      %1 = arith.addf %acc, %0 : f32
      linalg.yield %1 : f32
    } -> tensor<64x64xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %mm into %out[%i, %j] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<256x192xf32>
    }
  }
  return %r : tensor<256x192xf32>
}
// CHECK-LABEL: func.func @static_tile
// CHECK-NOT:     affine.min
// CHECK:         linalg.generic {{.*}} outs(%{{.+}} : tensor<64x64xf32>)
// CHECK-NOT:     affine.apply
