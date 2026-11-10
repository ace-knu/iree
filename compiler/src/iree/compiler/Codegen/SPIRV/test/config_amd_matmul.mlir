// RUN: iree-opt --split-input-file --iree-gpu-test-target=rdna2@vulkan --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' %s | FileCheck %s
// RUN: iree-opt --split-input-file --iree-gpu-test-target=rdna2@vulkan --iree-spirv-matmul-partial-m-max-waste=0 --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' %s | FileCheck %s --check-prefix=OFF

func.func @batch_matmul_f32_16x4096x40x4096(%3: tensor<16x4096x4096xf32>, %4: tensor<16x4096x40xf32>, %5: tensor<16x4096x40xf32>) -> tensor<16x4096x40xf32> {
  %6 = linalg.batch_matmul ins(%3, %4 : tensor<16x4096x4096xf32>, tensor<16x4096x40xf32>) outs(%5 : tensor<16x4096x40xf32>) -> tensor<16x4096x40xf32>
  return %6 : tensor<16x4096x40xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 256, 8, 32]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_gpu.spirv_pipeline<MatmulPromoteVectorize> workgroup_size = [2, 32, 1], {pipeline_depth = 1 : i64, store_stage = 1 : i64}>
//      CHECK: func.func @batch_matmul_f32_16x4096x40x4096(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.batch_matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]


// -----

func.func @matmul_f16_64x640x320(%3: tensor<64x320xf16>, %4: tensor<320x640xf16>) -> tensor<64x640xf16> {
  %cst = arith.constant 0.000000e+00 : f16
  %5 = tensor.empty() : tensor<64x640xf16>
  %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<64x640xf16>) -> tensor<64x640xf16>
  %7 = linalg.matmul ins(%3, %4 : tensor<64x320xf16>, tensor<320x640xf16>) outs(%6 : tensor<64x640xf16>) -> tensor<64x640xf16>
  return %7 : tensor<64x640xf16>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 128, 32]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_gpu.spirv_pipeline<MatmulPromoteVectorize> workgroup_size = [16, 8, 1], {pipeline_depth = 2 : i64, store_stage = 0 : i64}>
//      CHECK: func.func @matmul_f16_64x640x320(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

func.func @batch_matmul_f32_16x4096x48x4096(%3: tensor<16x4096x4096xf32>, %4: tensor<16x4096x48xf32>) -> tensor<16x4096x48xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %5 = tensor.empty() : tensor<16x4096x48xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<16x4096x48xf32>) -> tensor<16x4096x48xf32>
  %7 = linalg.batch_matmul ins(%3, %4 : tensor<16x4096x4096xf32>, tensor<16x4096x48xf32>) outs(%6 : tensor<16x4096x48xf32>) -> tensor<16x4096x48xf32>
  return %7 : tensor<16x4096x48xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 128, 16, 32]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_gpu.spirv_pipeline<MatmulPromoteVectorize> workgroup_size = [4, 16, 1], {pipeline_depth = 2 : i64, store_stage = 0 : i64}>
//      CHECK: func.func @batch_matmul_f32_16x4096x48x4096(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.batch_matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @batch_matmul_f16_1x4096x4096x512(%3: tensor<1x4096x512xf16>, %4: tensor<1x512x4096xf16>) -> tensor<1x4096x4096xf32> {
  %cst = arith.constant 0.000000e+00 : f16
  %5 = tensor.empty() : tensor<1x4096x4096xf32>
  %6 = tensor.empty() : tensor<1x4096x4096xf16>
  %7 = linalg.fill ins(%cst : f16) outs(%6 : tensor<1x4096x4096xf16>) -> tensor<1x4096x4096xf16>
  %8 = linalg.batch_matmul ins(%3, %4 : tensor<1x4096x512xf16>, tensor<1x512x4096xf16>) outs(%7 : tensor<1x4096x4096xf16>) -> tensor<1x4096x4096xf16>
  %9 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8 : tensor<1x4096x4096xf16>) outs(%5 : tensor<1x4096x4096xf32>) {
  ^bb0(%in: f16, %out: f32):
    %10 = arith.extf %in : f16 to f32
    linalg.yield %10 : f32
  } -> tensor<1x4096x4096xf32>
  return %9 : tensor<1x4096x4096xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 64, 128, 32]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_gpu.spirv_pipeline<MatmulPromoteVectorize> workgroup_size = [16, 8, 1], {pipeline_depth = 1 : i64, store_stage = 1 : i64}>
//      CHECK: func.func @batch_matmul_f16_1x4096x4096x512(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.batch_matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
func.func @matmul_multi_reduce_i4xf32xf32(%15: tensor<11008x32x128xi4>, %16: tensor<11008x32xf32>, %17: tensor<11008x32xf32>, %18: tensor<512x32x128xf32>) -> tensor<512x11008xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %19 = tensor.empty() : tensor<512x11008xf32>
  %20 = tensor.empty() : tensor<11008x32x128xf32>
  %21 = linalg.fill ins(%cst : f32) outs(%19 : tensor<512x11008xf32>) -> tensor<512x11008xf32>
  %22 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %16, %17 : tensor<11008x32x128xi4>, tensor<11008x32xf32>, tensor<11008x32xf32>) outs(%20 : tensor<11008x32x128xf32>) {
  ^bb0(%in: i4, %in_0: f32, %in_1: f32, %out: f32):
    %24 = arith.extui %in : i4 to i32
    %25 = arith.uitofp %24 : i32 to f32
    %26 = arith.subf %25, %in_1 : f32
    %27 = arith.mulf %26, %in_0 : f32
    linalg.yield %27 : f32
  } -> tensor<11008x32x128xf32>
  %23 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%18, %22 : tensor<512x32x128xf32>, tensor<11008x32x128xf32>) outs(%21 : tensor<512x11008xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %24 = arith.mulf %in, %in_0 : f32
    %25 = arith.addf %24, %out : f32
    linalg.yield %25 : f32
  } -> tensor<512x11008xf32>
  return %23 : tensor<512x11008xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64, 1, 16]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_gpu.spirv_pipeline<MatmulPromoteVectorize> workgroup_size = [16, 8, 1], {pipeline_depth = 0 : i64, store_stage = 1 : i64}>
//      CHECK: func.func @matmul_multi_reduce_i4xf32xf32(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
//      CHECK:   linalg.generic
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

// Partial last M tile (research/inception-opt, --iree-spirv-matmul-partial-m-max-waste):
// M = 289 (17x17 conv im2col) has no divisible tile with more than one
// workgroup row. A 64-row tile wastes 31/289 <= 12.5% and enables the
// promotion pipeline, with a single buffer (no multi-buffering).

#map_a = affine_map<(d0, d1, d2) -> (d0, d2)>
#map_b = affine_map<(d0, d1, d2) -> (d2, d1)>
#map_c = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @generic_matmul_f32_289x384x1024(%a: tensor<289x1024xf32>, %b: tensor<1024x384xf32>) -> tensor<289x384xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<289x384xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<289x384xf32>) -> tensor<289x384xf32>
  %r = linalg.generic {indexing_maps = [#map_a, #map_b, #map_c], iterator_types = ["parallel", "parallel", "reduction"]} ins(%a, %b : tensor<289x1024xf32>, tensor<1024x384xf32>) outs(%f : tensor<289x384xf32>) {
  ^bb0(%x: f32, %y: f32, %acc: f32):
    %m = arith.mulf %x, %y : f32
    %s = arith.addf %acc, %m : f32
    linalg.yield %s : f32
  } -> tensor<289x384xf32>
  return %r : tensor<289x384xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64, 16]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_gpu.spirv_pipeline<MatmulPromoteVectorize> workgroup_size = [16, 8, 1], {pipeline_depth = 1 : i64, store_stage = 1 : i64}>
//      CHECK: func.func @generic_matmul_f32_289x384x1024(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:     lowering_config = #[[CONFIG]]

//  OFF-DAG: #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 64], [1, 4], [0, 0, 16]{{\]}}>
//  OFF-DAG: #iree_codegen.translation_info<pipeline = #iree_gpu.spirv_pipeline<BaseVectorize> workgroup_size = [16, 1, 1]>
//      OFF: func.func @generic_matmul_f32_289x384x1024(

// -----

// M = 1225 (35x35): the divisible choice is a 7-row tile in one workgroup row;
// a 128-row tile (waste 55/1225) with 16 workgroup rows is used instead.

#map_a = affine_map<(d0, d1, d2) -> (d0, d2)>
#map_b = affine_map<(d0, d1, d2) -> (d2, d1)>
#map_c = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @generic_matmul_f32_1225x96x576(%a: tensor<1225x576xf32>, %b: tensor<576x96xf32>) -> tensor<1225x96xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<1225x96xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<1225x96xf32>) -> tensor<1225x96xf32>
  %r = linalg.generic {indexing_maps = [#map_a, #map_b, #map_c], iterator_types = ["parallel", "parallel", "reduction"]} ins(%a, %b : tensor<1225x576xf32>, tensor<576x96xf32>) outs(%f : tensor<1225x96xf32>) {
  ^bb0(%x: f32, %y: f32, %acc: f32):
    %m = arith.mulf %x, %y : f32
    %s = arith.addf %acc, %m : f32
    linalg.yield %s : f32
  } -> tensor<1225x96xf32>
  return %r : tensor<1225x96xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[128, 32, 16]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_gpu.spirv_pipeline<MatmulPromoteVectorize> workgroup_size = [8, 16, 1], {pipeline_depth = 1 : i64, store_stage = 1 : i64}>
//      CHECK: func.func @generic_matmul_f32_1225x96x576(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

// M = 49: no partial tile within the waste bound covers more than the
// divisible 7-row tile, so the configuration is unchanged.

#map_a = affine_map<(d0, d1, d2) -> (d0, d2)>
#map_b = affine_map<(d0, d1, d2) -> (d2, d1)>
#map_c = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @generic_matmul_f32_49x64x576(%a: tensor<49x576xf32>, %b: tensor<576x64xf32>) -> tensor<49x64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<49x64xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<49x64xf32>) -> tensor<49x64xf32>
  %r = linalg.generic {indexing_maps = [#map_a, #map_b, #map_c], iterator_types = ["parallel", "parallel", "reduction"]} ins(%a, %b : tensor<49x576xf32>, tensor<576x64xf32>) outs(%f : tensor<49x64xf32>) {
  ^bb0(%x: f32, %y: f32, %acc: f32):
    %m = arith.mulf %x, %y : f32
    %s = arith.addf %acc, %m : f32
    linalg.yield %s : f32
  } -> tensor<49x64xf32>
  return %r : tensor<49x64xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[7, 64], [7, 4], [0, 0, 16]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_gpu.spirv_pipeline<BaseVectorize> workgroup_size = [16, 1, 1]>
//      CHECK: func.func @generic_matmul_f32_49x64x576(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:     lowering_config = #[[CONFIG]]
