#include <stdio.h>
// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/direct_command_buffer.h"

#include <cstddef>
#include <cstdint>

#include "iree/base/api.h"
#include "iree/base/internal/inline_array.h"
#include "iree/base/internal/math.h"
#include "iree/hal/drivers/vulkan/base_buffer.h"
#include "iree/hal/drivers/vulkan/descriptor_set_arena.h"
#include "iree/hal/drivers/vulkan/dynamic_symbols.h"
#include "iree/hal/drivers/vulkan/native_buffer.h"
#include "iree/hal/drivers/vulkan/native_event.h"
#include "iree/hal/drivers/vulkan/native_executable.h"
#include "iree/hal/drivers/vulkan/pipeline_layout.h"
#include "iree/hal/drivers/vulkan/sparse_buffer.h"
#include "iree/hal/drivers/vulkan/status_util.h"
#include "iree/hal/drivers/vulkan/util/ref_ptr.h"
#include "iree/hal/utils/resource_set.h"

using namespace iree::hal::vulkan;

// Command buffer implementation that directly maps to VkCommandBuffer.
// This records the commands on the calling thread without additional threading
// indirection.
typedef struct iree_hal_vulkan_direct_command_buffer_t {
  iree_hal_command_buffer_t base;
  // Retained device allocator used for command-buffer-private staging buffers.
  iree_hal_allocator_t* device_allocator;
  VkDeviceHandle* logical_device;
  iree_hal_vulkan_tracing_context_t* tracing_context;
  iree_arena_block_pool_t* block_pool;

  VkCommandPoolHandle* command_pool;
  VkCommandBuffer handle;

  DynamicSymbols* syms;

  // Maintains a reference to all resources used within the command buffer.
  // Reset on each begin.
  iree_hal_resource_set_t* resource_set;

  // TODO(benvanik): may grow large - should try to reclaim or reuse.
  DescriptorSetArena descriptor_set_arena;

  // The current descriptor set group in use by the command buffer, if any.
  // This must remain valid until all in-flight submissions of the command
  // buffer complete.
  DescriptorSetGroup descriptor_set_group;

  BuiltinExecutables* builtin_executables;
} iree_hal_vulkan_direct_command_buffer_t;

namespace {
extern const iree_hal_command_buffer_vtable_t
    iree_hal_vulkan_direct_command_buffer_vtable;
}  // namespace

static iree_hal_vulkan_direct_command_buffer_t*
iree_hal_vulkan_direct_command_buffer_cast(
    iree_hal_command_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value,
                       &iree_hal_vulkan_direct_command_buffer_vtable);
  return (iree_hal_vulkan_direct_command_buffer_t*)base_value;
}

// 3DEV: Returns true if |buffer| is backed by a Vulkan buffer and can be passed
// to iree_hal_vulkan_buffer_handle. Buffers owned by other devices (amdxdna
// BOs, local heap buffers) are not and must never be cast.
static bool iree_hal_vulkan_direct_command_buffer_is_vulkan_buffer(
    iree_hal_buffer_t* buffer) {
  iree_hal_buffer_t* allocated = iree_hal_buffer_allocated_buffer(buffer);
  return iree_hal_vulkan_native_buffer_isa(allocated) ||
         iree_hal_vulkan_sparse_buffer_isa(allocated);
}

// 3DEV: Builds the error returned when a command receives a buffer it cannot
// use. |role| names the operand ("source", "binding", ...) and |index| is
// appended when not -1.
static iree_status_t iree_hal_vulkan_direct_command_buffer_foreign_buffer_error(
    const char* op, const char* role, int index, iree_hal_buffer_t* buffer,
    const char* reason) {
  iree_hal_buffer_t* allocated = iree_hal_buffer_allocated_buffer(buffer);
  iree_hal_device_t* device =
      iree_hal_buffer_allocation_placement(allocated).device;
  char index_str[16] = {0};
  if (index >= 0) snprintf(index_str, sizeof(index_str), "[%d]", index);
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "vulkan %s: %s%s buffer %p (allocated %p) is not a Vulkan buffer and "
      "cannot be used here (%s); memory_type=0x%x allowed_usage=0x%x "
      "placement.device=%s%p",
      op, role, index_str, (void*)buffer, (void*)allocated, reason,
      (unsigned)iree_hal_buffer_memory_type(allocated),
      (unsigned)iree_hal_buffer_allowed_usage(allocated),
      device ? "" : "NULL ", (void*)device);
}

// 3DEV: Consumes |cause| (a failed BRIDGE export/import) and returns the
// foreign buffer error for dispatch binding |index| with the cause appended.
static iree_status_t iree_hal_vulkan_direct_command_buffer_bridge_error(
    int index, iree_hal_buffer_t* buffer, const char* stage,
    iree_status_t cause) {
  char cause_str[512] = {0};
  iree_host_size_t cause_length = 0;
  if (!iree_status_format(cause, sizeof(cause_str), cause_str,
                          &cause_length)) {
    snprintf(cause_str, sizeof(cause_str), "%s",
             iree_status_code_string(iree_status_code(cause)));
  }
  iree_status_ignore(cause);
  char reason[640] = {0};
  snprintf(reason, sizeof(reason), "BRIDGE %s failed: %s", stage, cause_str);
  return iree_hal_vulkan_direct_command_buffer_foreign_buffer_error(
      "dispatch", "binding", index, buffer, reason);
}

// 3DEV: Returns an error if |buffer| is not a Vulkan buffer. NULL is allowed.
static iree_status_t iree_hal_vulkan_direct_command_buffer_require_vulkan_buffer(
    const char* op, const char* role, int index, iree_hal_buffer_t* buffer) {
  if (!buffer ||
      iree_hal_vulkan_direct_command_buffer_is_vulkan_buffer(buffer)) {
    return iree_ok_status();
  }
  return iree_hal_vulkan_direct_command_buffer_foreign_buffer_error(
      op, role, index, buffer,
      "foreign buffers are not supported by this command");
}

iree_status_t iree_hal_vulkan_direct_command_buffer_allocate(
    iree_hal_allocator_t* device_allocator,
    iree::hal::vulkan::VkDeviceHandle* logical_device,
    iree::hal::vulkan::VkCommandPoolHandle* command_pool,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_vulkan_tracing_context_t* tracing_context,
    iree::hal::vulkan::DescriptorPoolCache* descriptor_pool_cache,
    iree::hal::vulkan::BuiltinExecutables* builtin_executables,
    iree_arena_block_pool_t* block_pool,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(device_allocator);
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(command_pool);
  IREE_ASSERT_ARGUMENT(descriptor_pool_cache);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = NULL;

  if (binding_capacity > 0) {
    // TODO(#10144): support indirect command buffers with binding tables.
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "indirect command buffers not yet implemented");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  VkCommandBufferAllocateInfo allocate_info;
  allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocate_info.pNext = NULL;
  allocate_info.commandPool = *command_pool;
  allocate_info.commandBufferCount = 1;
  allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

  VkCommandBuffer handle = VK_NULL_HANDLE;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, command_pool->Allocate(&allocate_info, &handle));

  iree_hal_vulkan_direct_command_buffer_t* command_buffer = NULL;
  iree_status_t status = iree_allocator_malloc(
      logical_device->host_allocator(),
      sizeof(*command_buffer) +
          iree_hal_command_buffer_validation_state_size(mode, binding_capacity),
      (void**)&command_buffer);
  if (iree_status_is_ok(status)) {
    iree_hal_command_buffer_initialize(
        device_allocator, mode, command_categories, queue_affinity,
        binding_capacity, (uint8_t*)command_buffer + sizeof(*command_buffer),
        &iree_hal_vulkan_direct_command_buffer_vtable, &command_buffer->base);
    command_buffer->device_allocator = NULL;
    command_buffer->logical_device = logical_device;
    command_buffer->tracing_context = tracing_context;
    command_buffer->block_pool = block_pool;
    command_buffer->command_pool = command_pool;
    command_buffer->handle = handle;
    command_buffer->syms = logical_device->syms().get();

    new (&command_buffer->descriptor_set_arena)
        DescriptorSetArena(descriptor_pool_cache);
    new (&command_buffer->descriptor_set_group) DescriptorSetGroup();

    command_buffer->builtin_executables = builtin_executables;
    command_buffer->resource_set = NULL;
    if (!iree_all_bits_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED)) {
      status = iree_hal_resource_set_allocate(block_pool,
                                              &command_buffer->resource_set);
    }
  }

  if (iree_status_is_ok(status)) {
    iree_hal_allocator_retain(device_allocator);
    command_buffer->device_allocator = device_allocator;
    *out_command_buffer = &command_buffer->base;
  } else {
    if (command_buffer) {
      command_buffer->descriptor_set_group.~DescriptorSetGroup();
      command_buffer->descriptor_set_arena.~DescriptorSetArena();
      iree_hal_resource_set_free(command_buffer->resource_set);
      iree_allocator_free(logical_device->host_allocator(), command_buffer);
    }
    command_pool->Free(handle);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

bool iree_hal_vulkan_direct_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer) {
  return iree_hal_resource_is(&command_buffer->resource,
                              &iree_hal_vulkan_direct_command_buffer_vtable);
}

static void iree_hal_vulkan_direct_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_vulkan_direct_command_buffer_t* command_buffer =
      iree_hal_vulkan_direct_command_buffer_cast(base_command_buffer);
  iree_allocator_t host_allocator =
      command_buffer->logical_device->host_allocator();
  IREE_TRACE_ZONE_BEGIN(z0);

  command_buffer->command_pool->Free(command_buffer->handle);

  IREE_IGNORE_ERROR(command_buffer->descriptor_set_group.Reset());
  command_buffer->descriptor_set_group.~DescriptorSetGroup();
  command_buffer->descriptor_set_arena.~DescriptorSetArena();

  iree_hal_resource_set_free(command_buffer->resource_set);
  if (command_buffer->device_allocator) {
    iree_hal_allocator_release(command_buffer->device_allocator);
  }
  iree_allocator_free(host_allocator, command_buffer);

  IREE_TRACE_ZONE_END(z0);
}

VkCommandBuffer iree_hal_vulkan_direct_command_buffer_handle(
    iree_hal_command_buffer_t* base_command_buffer) {
  if (!iree_hal_vulkan_direct_command_buffer_isa(base_command_buffer)) {
    return VK_NULL_HANDLE;
  }
  iree_hal_vulkan_direct_command_buffer_t* command_buffer =
      iree_hal_vulkan_direct_command_buffer_cast(base_command_buffer);
  return command_buffer->handle;
}

static iree_status_t iree_hal_vulkan_direct_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_vulkan_direct_command_buffer_t* command_buffer =
      iree_hal_vulkan_direct_command_buffer_cast(base_command_buffer);

  VkCommandBufferBeginInfo begin_info;
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.pNext = NULL;
  begin_info.flags = iree_all_bits_set(command_buffer->base.mode,
                                       IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT)
                         ? VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
                         : 0;
  begin_info.pInheritanceInfo = NULL;
  VK_RETURN_IF_ERROR(command_buffer->syms->vkBeginCommandBuffer(
                         command_buffer->handle, &begin_info),
                     "vkBeginCommandBuffer");

  IREE_VULKAN_TRACE_ZONE_BEGIN_EXTERNAL(
      command_buffer->tracing_context, command_buffer->handle,
      /*file_name=*/NULL, 0,
      /*line=*/0, "iree_hal_vulkan_direct_command_buffer",
      strlen("iree_hal_vulkan_direct_command_buffer"), /*name=*/NULL, 0);

  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_direct_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_vulkan_direct_command_buffer_t* command_buffer =
      iree_hal_vulkan_direct_command_buffer_cast(base_command_buffer);

  IREE_VULKAN_TRACE_ZONE_END(command_buffer->tracing_context,
                             command_buffer->handle);

  VK_RETURN_IF_ERROR(
      command_buffer->syms->vkEndCommandBuffer(command_buffer->handle),
      "vkEndCommandBuffer");

  // Flush all pending descriptor set writes (if any).
  command_buffer->descriptor_set_group =
      command_buffer->descriptor_set_arena.Flush();

  iree_hal_resource_set_freeze(command_buffer->resource_set);

  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_direct_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t* base_command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color,
    const iree_hal_label_location_t* location) {
  iree_hal_vulkan_direct_command_buffer_t* command_buffer =
      iree_hal_vulkan_direct_command_buffer_cast(base_command_buffer);
  IREE_VULKAN_TRACE_ZONE_BEGIN_EXTERNAL(
      command_buffer->tracing_context, command_buffer->handle,
      location ? location->file.data : NULL, location ? location->file.size : 0,
      location ? location->line : 0, /*func_name=*/NULL, 0, label.data,
      label.size);
  if (command_buffer->syms->vkCmdBeginDebugUtilsLabelEXT) {
    char label_buffer[128];
    iree_snprintf(label_buffer, sizeof(label_buffer), "%.*s", (int)label.size,
                  label.data);
    VkDebugUtilsLabelEXT label_info = {
        /*.sType=*/VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
        /*.pNext=*/NULL,
        /*.pLabelName=*/label_buffer,
        /*.color=*/
        {
            /*r=*/label_color.r / 255.0f,
            /*g=*/label_color.g / 255.0f,
            /*b=*/label_color.b / 255.0f,
            /*a=*/label_color.a / 255.0f,
        },
    };
    command_buffer->syms->vkCmdBeginDebugUtilsLabelEXT(command_buffer->handle,
                                                       &label_info);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_direct_command_buffer_end_debug_group(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_vulkan_direct_command_buffer_t* command_buffer =
      iree_hal_vulkan_direct_command_buffer_cast(base_command_buffer);
  if (command_buffer->syms->vkCmdEndDebugUtilsLabelEXT) {
    command_buffer->syms->vkCmdEndDebugUtilsLabelEXT(command_buffer->handle);
  }
  IREE_VULKAN_TRACE_ZONE_END(command_buffer->tracing_context,
                             command_buffer->handle);
  return iree_ok_status();
}

static VkPipelineStageFlags iree_hal_vulkan_convert_pipeline_stage_flags(
    iree_hal_execution_stage_t stage_mask) {
  VkPipelineStageFlags flags = 0;
  flags |= iree_any_bit_set(stage_mask, IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE)
               ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
               : 0;
  flags |=
      iree_any_bit_set(stage_mask, IREE_HAL_EXECUTION_STAGE_COMMAND_PROCESS)
          ? VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT
          : 0;
  flags |= iree_any_bit_set(stage_mask, IREE_HAL_EXECUTION_STAGE_DISPATCH)
               ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
               : 0;
  flags |= iree_any_bit_set(stage_mask, IREE_HAL_EXECUTION_STAGE_TRANSFER)
               ? VK_PIPELINE_STAGE_TRANSFER_BIT
               : 0;
  flags |= iree_any_bit_set(stage_mask, IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE)
               ? VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT
               : 0;
  flags |= iree_any_bit_set(stage_mask, IREE_HAL_EXECUTION_STAGE_HOST)
               ? VK_PIPELINE_STAGE_HOST_BIT
               : 0;
  return flags;
}

static VkAccessFlags iree_hal_vulkan_convert_access_mask(
    iree_hal_access_scope_t access_mask) {
  VkAccessFlags flags = 0;
  flags |=
      iree_any_bit_set(access_mask, IREE_HAL_ACCESS_SCOPE_INDIRECT_COMMAND_READ)
          ? VK_ACCESS_INDIRECT_COMMAND_READ_BIT
          : 0;
  flags |= iree_any_bit_set(access_mask, IREE_HAL_ACCESS_SCOPE_CONSTANT_READ)
               ? VK_ACCESS_UNIFORM_READ_BIT
               : 0;
  flags |= iree_any_bit_set(access_mask, IREE_HAL_ACCESS_SCOPE_DISPATCH_READ)
               ? VK_ACCESS_SHADER_READ_BIT
               : 0;
  flags |= iree_any_bit_set(access_mask, IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE)
               ? VK_ACCESS_SHADER_WRITE_BIT
               : 0;
  flags |= iree_any_bit_set(access_mask, IREE_HAL_ACCESS_SCOPE_TRANSFER_READ)
               ? VK_ACCESS_TRANSFER_READ_BIT
               : 0;
  flags |= iree_any_bit_set(access_mask, IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE)
               ? VK_ACCESS_TRANSFER_WRITE_BIT
               : 0;
  flags |= iree_any_bit_set(access_mask, IREE_HAL_ACCESS_SCOPE_HOST_READ)
               ? VK_ACCESS_HOST_READ_BIT
               : 0;
  flags |= iree_any_bit_set(access_mask, IREE_HAL_ACCESS_SCOPE_HOST_WRITE)
               ? VK_ACCESS_HOST_WRITE_BIT
               : 0;
  flags |= iree_any_bit_set(access_mask, IREE_HAL_ACCESS_SCOPE_MEMORY_READ)
               ? VK_ACCESS_MEMORY_READ_BIT
               : 0;
  flags |= iree_any_bit_set(access_mask, IREE_HAL_ACCESS_SCOPE_MEMORY_WRITE)
               ? VK_ACCESS_MEMORY_WRITE_BIT
               : 0;
  return flags;
}

static iree_status_t iree_hal_vulkan_direct_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  iree_hal_vulkan_direct_command_buffer_t* command_buffer =
      iree_hal_vulkan_direct_command_buffer_cast(base_command_buffer);
  iree_allocator_t host_allocator =
      command_buffer->logical_device->host_allocator();

  for (iree_host_size_t i = 0; i < buffer_barrier_count; ++i) {
    IREE_RETURN_IF_ERROR(
        iree_hal_vulkan_direct_command_buffer_require_vulkan_buffer(
            "execution_barrier", "buffer_barrier", (int)i,
            buffer_barriers[i].buffer_ref.buffer));
  }

  iree_inline_array(VkMemoryBarrier, memory_barrier_infos, memory_barrier_count,
                    host_allocator);
  for (int i = 0; i < memory_barrier_count; ++i) {
    const iree_hal_memory_barrier_t& memory_barrier = memory_barriers[i];
    VkMemoryBarrier* info = iree_inline_array_at(memory_barrier_infos, i);
    info->sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    info->pNext = NULL;
    info->srcAccessMask =
        iree_hal_vulkan_convert_access_mask(memory_barrier.source_scope);
    info->dstAccessMask =
        iree_hal_vulkan_convert_access_mask(memory_barrier.target_scope);
  }

  iree_inline_array(VkBufferMemoryBarrier, buffer_barrier_infos,
                    buffer_barrier_count, host_allocator);
  for (int i = 0; i < buffer_barrier_count; ++i) {
    const iree_hal_buffer_barrier_t& buffer_barrier = buffer_barriers[i];
    VkBufferMemoryBarrier* info = iree_inline_array_at(buffer_barrier_infos, i);
    info->sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    info->pNext = NULL;
    info->srcAccessMask =
        iree_hal_vulkan_convert_access_mask(buffer_barrier.source_scope);
    info->dstAccessMask =
        iree_hal_vulkan_convert_access_mask(buffer_barrier.target_scope);
    info->srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    info->dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    info->buffer =
        iree_hal_vulkan_buffer_handle(buffer_barrier.buffer_ref.buffer);
    info->offset = buffer_barrier.buffer_ref.offset;
    info->size = buffer_barrier.buffer_ref.length;
  }

  command_buffer->syms->vkCmdPipelineBarrier(
      command_buffer->handle,
      iree_hal_vulkan_convert_pipeline_stage_flags(source_stage_mask),
      iree_hal_vulkan_convert_pipeline_stage_flags(target_stage_mask),
      /*dependencyFlags=*/0, (uint32_t)memory_barrier_count,
      iree_inline_array_data(memory_barrier_infos),
      (uint32_t)buffer_barrier_count,
      iree_inline_array_data(buffer_barrier_infos), 0, NULL);

  iree_inline_array_deinitialize(memory_barrier_infos);
  iree_inline_array_deinitialize(buffer_barrier_infos);

  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_direct_command_buffer_signal_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  iree_hal_vulkan_direct_command_buffer_t* command_buffer =
      iree_hal_vulkan_direct_command_buffer_cast(base_command_buffer);

  IREE_RETURN_IF_ERROR(
      iree_hal_resource_set_insert(command_buffer->resource_set, 1, &event));

  command_buffer->syms->vkCmdSetEvent(
      command_buffer->handle, iree_hal_vulkan_native_event_handle(event),
      iree_hal_vulkan_convert_pipeline_stage_flags(source_stage_mask));

  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_direct_command_buffer_reset_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  iree_hal_vulkan_direct_command_buffer_t* command_buffer =
      iree_hal_vulkan_direct_command_buffer_cast(base_command_buffer);

  IREE_RETURN_IF_ERROR(
      iree_hal_resource_set_insert(command_buffer->resource_set, 1, &event));

  command_buffer->syms->vkCmdResetEvent(
      command_buffer->handle, iree_hal_vulkan_native_event_handle(event),
      iree_hal_vulkan_convert_pipeline_stage_flags(source_stage_mask));

  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_direct_command_buffer_wait_events(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_host_size_t event_count, const iree_hal_event_t** events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  iree_hal_vulkan_direct_command_buffer_t* command_buffer =
      iree_hal_vulkan_direct_command_buffer_cast(base_command_buffer);
  iree_allocator_t host_allocator =
      command_buffer->logical_device->host_allocator();

  for (iree_host_size_t i = 0; i < buffer_barrier_count; ++i) {
    IREE_RETURN_IF_ERROR(
        iree_hal_vulkan_direct_command_buffer_require_vulkan_buffer(
            "wait_events", "buffer_barrier", (int)i,
            buffer_barriers[i].buffer_ref.buffer));
  }

  IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert(
      command_buffer->resource_set, event_count, events));

  iree_inline_array(VkEvent, event_handles, event_count, host_allocator);
  for (int i = 0; i < event_count; ++i) {
    *iree_inline_array_at(event_handles, i) =
        iree_hal_vulkan_native_event_handle(events[i]);
  }

  iree_inline_array(VkMemoryBarrier, memory_barrier_infos, memory_barrier_count,
                    host_allocator);
  for (int i = 0; i < memory_barrier_count; ++i) {
    const iree_hal_memory_barrier_t& memory_barrier = memory_barriers[i];
    VkMemoryBarrier* info = iree_inline_array_at(memory_barrier_infos, i);
    info->sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    info->pNext = NULL;
    info->srcAccessMask =
        iree_hal_vulkan_convert_access_mask(memory_barrier.source_scope);
    info->dstAccessMask =
        iree_hal_vulkan_convert_access_mask(memory_barrier.target_scope);
  }

  iree_inline_array(VkBufferMemoryBarrier, buffer_barrier_infos,
                    buffer_barrier_count, host_allocator);
  for (int i = 0; i < buffer_barrier_count; ++i) {
    const iree_hal_buffer_barrier_t& buffer_barrier = buffer_barriers[i];
    VkBufferMemoryBarrier* info = iree_inline_array_at(buffer_barrier_infos, i);
    info->sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    info->pNext = NULL;
    info->srcAccessMask =
        iree_hal_vulkan_convert_access_mask(buffer_barrier.source_scope);
    info->dstAccessMask =
        iree_hal_vulkan_convert_access_mask(buffer_barrier.target_scope);
    info->srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    info->dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    info->buffer =
        iree_hal_vulkan_buffer_handle(buffer_barrier.buffer_ref.buffer);
    info->offset = buffer_barrier.buffer_ref.offset;
    info->size = buffer_barrier.buffer_ref.length;
  }

  command_buffer->syms->vkCmdWaitEvents(
      command_buffer->handle, (uint32_t)event_count,
      iree_inline_array_data(event_handles),
      iree_hal_vulkan_convert_pipeline_stage_flags(source_stage_mask),
      iree_hal_vulkan_convert_pipeline_stage_flags(target_stage_mask),
      (uint32_t)memory_barrier_count,
      iree_inline_array_data(memory_barrier_infos),
      (uint32_t)buffer_barrier_count,
      iree_inline_array_data(buffer_barrier_infos), 0, NULL);

  iree_inline_array_deinitialize(event_handles);
  iree_inline_array_deinitialize(memory_barrier_infos);
  iree_inline_array_deinitialize(buffer_barrier_infos);

  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_direct_command_buffer_advise_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t buffer_ref, iree_hal_memory_advise_flags_t flags,
    uint64_t arg0, uint64_t arg1) {
  // NOTE: we could use this to prevent queue family transitions.
  return iree_ok_status();
}

// Splats a pattern value of 1, 2, or 4 bytes out to a 4 byte value.
static uint32_t iree_hal_vulkan_splat_pattern(const void* pattern,
                                              size_t pattern_length) {
  switch (pattern_length) {
    case 1: {
      uint32_t pattern_value = *static_cast<const uint8_t*>(pattern);
      return (pattern_value << 24) | (pattern_value << 16) |
             (pattern_value << 8) | pattern_value;
    }
    case 2: {
      uint32_t pattern_value = *static_cast<const uint16_t*>(pattern);
      return (pattern_value << 16) | pattern_value;
    }
    case 4: {
      uint32_t pattern_value = *static_cast<const uint32_t*>(pattern);
      return pattern_value;
    }
    default:
      return 0;  // Already verified that this should not be possible.
  }
}

static iree_status_t iree_hal_vulkan_direct_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t target_ref, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  iree_hal_vulkan_direct_command_buffer_t* command_buffer =
      iree_hal_vulkan_direct_command_buffer_cast(base_command_buffer);
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_direct_command_buffer_require_vulkan_buffer(
          "fill_buffer", "target", -1, target_ref.buffer));
  VkBuffer target_device_buffer =
      iree_hal_vulkan_buffer_handle(target_ref.buffer);

  IREE_VULKAN_TRACE_ZONE_BEGIN(command_buffer->tracing_context,
                               command_buffer->handle);

  IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert(
      command_buffer->resource_set, 1, &target_ref.buffer));

  // vkCmdFillBuffer requires a 4 byte alignment for the offset, pattern, and
  // length. We use a polyfill here that fills the unaligned start and end of
  // fill operations, if needed.

  iree_device_size_t target_offset = target_ref.offset;
  iree_device_size_t length = target_ref.length;
  if (target_offset % 4 != 0 || length % 4 != 0) {
    // TODO(scotttodd): only restore push constants that have been modified?
    //                  (this can pass uninitialized memory right now, which
    //                   *should* be safe but is wasteful)
    IREE_RETURN_IF_ERROR(
        command_buffer->builtin_executables->FillBufferUnaligned(
            command_buffer->handle, &(command_buffer->descriptor_set_arena),
            target_ref.buffer, target_offset, length, pattern, pattern_length));

    // Continue using vkCmdFillBuffer below, but only for the inner aligned
    // portion of the fill operation.
    // For example:
    //   original offset 2, length 8
    //   aligned  offset 4, length 4
    // [0x00,0x00,0xAB,0xAB | 0xAB,0xAB,0xAB,0xAB | 0xAB,0xAB,0x00,0x00]
    //            <-------> <---------------------> <------->
    //            unaligned     vkCmdFillBuffer     unaligned
    iree_device_size_t aligned_target_offset =
        iree_device_align(target_offset, 4);
    iree_device_size_t target_end = target_offset + length;
    iree_device_size_t rounded_down_target_end = (target_end / 4) * 4;
    length -= (aligned_target_offset - target_ref.offset) +
              (target_end - rounded_down_target_end);
    target_offset = aligned_target_offset;
  }

  if (length > 0) {
    // Note that vkCmdFillBuffer only accepts 4-byte aligned values so we need
    // to splat out our variable-length pattern.
    target_offset += iree_hal_buffer_byte_offset(target_ref.buffer);
    uint32_t dword_pattern =
        iree_hal_vulkan_splat_pattern(pattern, pattern_length);
    command_buffer->syms->vkCmdFillBuffer(command_buffer->handle,
                                          target_device_buffer, target_offset,
                                          length, dword_pattern);
  }

  IREE_VULKAN_TRACE_ZONE_END(command_buffer->tracing_context,
                             command_buffer->handle);

  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_direct_command_buffer_insert_resources(
    iree_hal_vulkan_direct_command_buffer_t* command_buffer,
    iree_host_size_t count, const void* resources) {
  if (iree_all_bits_set(command_buffer->base.mode,
                        IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED)) {
    return iree_ok_status();
  }
  return iree_hal_resource_set_insert(command_buffer->resource_set, count,
                                      resources);
}

static iree_status_t
iree_hal_vulkan_direct_command_buffer_insert_internal_resources(
    iree_hal_vulkan_direct_command_buffer_t* command_buffer,
    iree_host_size_t count, const void* resources) {
  if (!command_buffer->resource_set) {
    IREE_RETURN_IF_ERROR(iree_hal_resource_set_allocate(
        command_buffer->block_pool, &command_buffer->resource_set));
  }
  return iree_hal_resource_set_insert(command_buffer->resource_set, count,
                                      resources);
}

static void iree_hal_vulkan_direct_command_buffer_record_copy_buffer(
    iree_hal_vulkan_direct_command_buffer_t* command_buffer,
    iree_hal_buffer_ref_t source_ref, iree_hal_buffer_ref_t target_ref) {
  VkBuffer source_device_buffer =
      iree_hal_vulkan_buffer_handle(source_ref.buffer);
  VkBuffer target_device_buffer =
      iree_hal_vulkan_buffer_handle(target_ref.buffer);

  VkBufferCopy region;
  region.srcOffset =
      iree_hal_buffer_byte_offset(source_ref.buffer) + source_ref.offset;
  region.dstOffset =
      iree_hal_buffer_byte_offset(target_ref.buffer) + target_ref.offset;
  region.size = target_ref.length;
  command_buffer->syms->vkCmdCopyBuffer(command_buffer->handle,
                                        source_device_buffer,
                                        target_device_buffer, 1, &region);
}

static iree_status_t
iree_hal_vulkan_direct_command_buffer_update_buffer_via_copy(
    iree_hal_vulkan_direct_command_buffer_t* command_buffer,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_ref_t target_ref) {
  iree_hal_buffer_params_t staging_params = {0};
  staging_params.type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  staging_params.access = IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE;
  staging_params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE | IREE_HAL_BUFFER_USAGE_MAPPING;

  iree_hal_buffer_t* staging_buffer = NULL;
  iree_status_t status = iree_hal_allocator_allocate_buffer(
      command_buffer->device_allocator, staging_params, target_ref.length,
      &staging_buffer);
  const uint8_t* source_buffer_ptr =
      static_cast<const uint8_t*>(source_buffer) + source_offset;
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_map_write(staging_buffer, /*target_offset=*/0,
                                       source_buffer_ptr, target_ref.length);
  }
  if (iree_status_is_ok(status)) {
    const iree_hal_buffer_t* buffers[1] = {staging_buffer};
    status = iree_hal_vulkan_direct_command_buffer_insert_internal_resources(
        command_buffer, 1, buffers);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_ref_t source_ref = iree_hal_make_buffer_ref(
        staging_buffer, /*offset=*/0, target_ref.length);
    iree_hal_vulkan_direct_command_buffer_record_copy_buffer(
        command_buffer, source_ref, target_ref);
  }
  iree_hal_buffer_release(staging_buffer);
  return status;
}

static iree_status_t iree_hal_vulkan_direct_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_ref_t target_ref,
    iree_hal_update_flags_t flags) {
  iree_hal_vulkan_direct_command_buffer_t* command_buffer =
      iree_hal_vulkan_direct_command_buffer_cast(base_command_buffer);
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_direct_command_buffer_require_vulkan_buffer(
          "update_buffer", "target", -1, target_ref.buffer));
  VkBuffer target_device_buffer =
      iree_hal_vulkan_buffer_handle(target_ref.buffer);

  IREE_VULKAN_TRACE_ZONE_BEGIN(command_buffer->tracing_context,
                               command_buffer->handle);

  const iree_hal_buffer_t* target_buffers[1] = {target_ref.buffer};
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_direct_command_buffer_insert_resources(
      command_buffer, 1, target_buffers));

  // Vulkan only allows updates of <= 65536 because you really, really, really
  // shouldn't do large updates like this (as it wastes command buffer space and
  // may be slower than just using write-through mapped memory). The
  // recommendation in the spec for larger updates is to split the single update
  // into multiple updates over the entire desired range.
  const auto* source_buffer_ptr =
      static_cast<const uint8_t*>(source_buffer) + source_offset;
  iree_device_size_t target_offset =
      iree_hal_buffer_byte_offset(target_ref.buffer) + target_ref.offset;
  if (!iree_host_size_has_alignment(
          (iree_host_size_t)(uintptr_t)source_buffer_ptr,
          /*alignment=*/4) ||
      !iree_device_size_has_alignment(target_offset, /*alignment=*/4) ||
      !iree_device_size_has_alignment(target_ref.length, /*alignment=*/4)) {
    iree_status_t status =
        iree_hal_vulkan_direct_command_buffer_update_buffer_via_copy(
            command_buffer, source_buffer, source_offset, target_ref);
    IREE_VULKAN_TRACE_ZONE_END(command_buffer->tracing_context,
                               command_buffer->handle);
    return status;
  }

  iree_device_size_t length = target_ref.length;
  while (length > 0) {
    iree_device_size_t chunk_length =
        iree_min((iree_device_size_t)65536u, length);
    command_buffer->syms->vkCmdUpdateBuffer(command_buffer->handle,
                                            target_device_buffer, target_offset,
                                            chunk_length, source_buffer_ptr);
    source_buffer_ptr += chunk_length;
    target_offset += chunk_length;
    length -= chunk_length;
  }

  IREE_VULKAN_TRACE_ZONE_END(command_buffer->tracing_context,
                             command_buffer->handle);

  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_direct_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t source_ref, iree_hal_buffer_ref_t target_ref,
    iree_hal_copy_flags_t flags) {
  iree_hal_vulkan_direct_command_buffer_t* command_buffer =
      iree_hal_vulkan_direct_command_buffer_cast(base_command_buffer);
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_direct_command_buffer_require_vulkan_buffer(
          "copy_buffer", "source", -1, source_ref.buffer));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_direct_command_buffer_require_vulkan_buffer(
          "copy_buffer", "target", -1, target_ref.buffer));

  IREE_VULKAN_TRACE_ZONE_BEGIN(command_buffer->tracing_context,
                               command_buffer->handle);

  const iree_hal_buffer_t* buffers[2] = {source_ref.buffer, target_ref.buffer};
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_direct_command_buffer_insert_resources(
      command_buffer, 2, buffers));

  iree_hal_vulkan_direct_command_buffer_record_copy_buffer(
      command_buffer, source_ref, target_ref);

  IREE_VULKAN_TRACE_ZONE_END(command_buffer->tracing_context,
                             command_buffer->handle);

  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_direct_command_buffer_collective(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_channel_t* channel,
    iree_hal_collective_op_t op, uint32_t param, iree_hal_buffer_ref_t send_ref,
    iree_hal_buffer_ref_t recv_ref, iree_device_size_t element_count) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "collectives not yet implemented on Vulkan");
}

static iree_status_t iree_hal_vulkan_direct_command_buffer_dispatch_bind(
    iree_hal_vulkan_direct_command_buffer_t* command_buffer,
    const iree_hal_vulkan_pipeline_t* pipeline,
    iree_const_byte_span_t constants, iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags) {
  return iree_ok_status();
}

iree_hal_buffer_ref_t g_last_bindings[8];
iree_host_size_t g_last_binding_count = 0;
static iree_status_t iree_hal_vulkan_direct_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  g_last_binding_count = bindings.count < 8 ? bindings.count : 8;
  for (iree_host_size_t gi = 0; gi < g_last_binding_count; ++gi) g_last_bindings[gi] = bindings.values[gi];
  fprintf(stderr, "[BR] vk dispatch entry=%u wg=[%u %u %u] bindings=%zu\n", (unsigned)export_ordinal,
          (unsigned)config.workgroup_count[0], (unsigned)config.workgroup_count[1],
          (unsigned)config.workgroup_count[2], (size_t)bindings.count);
  for (iree_host_size_t bi = 0; bi < bindings.count; ++bi) {
    iree_hal_buffer_ref_t ref = bindings.values[bi];
    iree_hal_buffer_mapping_t m;
    iree_status_t ms = iree_hal_buffer_map_range(ref.buffer, IREE_HAL_MAPPING_MODE_SCOPED,
        IREE_HAL_MEMORY_ACCESS_READ, ref.offset, 16, &m);
    if (iree_status_is_ok(ms)) {
      const float* f = (const float*)m.contents.data;
      fprintf(stderr, "[BR] vk dispatch binding[%zu] buf=%p dev=%p type=0x%x first=[%g %g %g %g]\n",
              (size_t)bi, (void*)ref.buffer, (void*)iree_hal_buffer_allocation_placement(ref.buffer).device,
              (unsigned)iree_hal_buffer_memory_type(ref.buffer), f[0], f[1], f[2], f[3]);
      iree_hal_buffer_unmap_range(&m);
    } else {
      fprintf(stderr, "[BR] vk dispatch binding[%zu] buf=%p type=0x%x unmappable\n",
              (size_t)bi, (void*)ref.buffer, (unsigned)iree_hal_buffer_memory_type(ref.buffer));
      iree_status_ignore(ms);
    }
    fflush(stderr);
  }
  iree_hal_vulkan_direct_command_buffer_t* command_buffer =
      iree_hal_vulkan_direct_command_buffer_cast(base_command_buffer);

  // BRIDGE: bindings owned by another device (e.g. amdxdna BOs) are not
  // VkBuffers. Export them as dma-buf fds and import as VkDeviceMemory so the
  // dispatch can bind them directly. Without this the buffer_handle cast in
  // descriptor_set_arena reads garbage and the shader silently sees zeros.
  iree_hal_buffer_ref_t patched_bindings[8];
  bool any_imported = false;
  if (bindings.count <= 8) {
    for (iree_host_size_t bi = 0; bi < bindings.count; ++bi) {
      patched_bindings[bi] = bindings.values[bi];
      iree_hal_buffer_t* src = bindings.values[bi].buffer;
      if (!src) continue;
      iree_hal_buffer_t* allocated = iree_hal_buffer_allocated_buffer(src);
      iree_hal_device_t* src_device =
          iree_hal_buffer_allocation_placement(allocated).device;


      iree_hal_external_buffer_t ext;
      memset(&ext, 0, sizeof(ext));
      iree_status_t s = iree_ok_status();
      if (!src_device) {
        // Host (heap) buffers carry no owning device in their placement, so
        // there is no allocator to ask for an export. Map the buffer to get
        // its host pointer and import it as HOST_ALLOCATION. Vulkan-allocated
        // buffers always record their device, so they never reach here.
        iree_hal_buffer_mapping_t mapping;


        // READ|WRITE only: ACCESS_ALL includes DISCARD, and heap buffers in
        // debug builds scribble 0xCD over the range when mapped for discard
        // (buffer_heap.c), which wiped the input data before the GPU read it.
        s = iree_hal_buffer_map_range(
            allocated, IREE_HAL_MAPPING_MODE_PERSISTENT,
            IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE, 0,
            IREE_HAL_WHOLE_BUFFER, &mapping);

            
        if (iree_status_is_ok(s)) {
          ext.type = IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION;
          ext.flags = IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE;
          ext.size = iree_hal_buffer_allocation_size(allocated);
          ext.handle.host_allocation.ptr = mapping.contents.data;
          // Heap unmap is a no-op; the pointer stays valid while the buffer
          // lives, and the binding keeps the buffer alive for the dispatch.
          iree_hal_buffer_unmap_range(&mapping);
        }
      } else {
        iree_hal_allocator_t* src_allocator =
            iree_hal_device_allocator(src_device);
        if (src_allocator == command_buffer->device_allocator) continue;
        // Ask the owner for a handle this allocator can import. Try the
        // cheapest-to-share types in order:
        //  1. OPAQUE_FD (dma-buf): device allocations such as amdxdna BOs.
        //  2. HOST_ALLOCATION: host allocations such as local-task buffers;
        //     imported via VK_EXT_external_memory_host.
        s = iree_hal_allocator_export_buffer(
            src_allocator, allocated, IREE_HAL_EXTERNAL_BUFFER_TYPE_OPAQUE_FD,
            IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE, &ext);
        if (!iree_status_is_ok(s)) {
          iree_status_ignore(s);
          memset(&ext, 0, sizeof(ext));
          s = iree_hal_allocator_export_buffer(
              src_allocator, allocated,
              IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION,
              IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE, &ext);
        }
      }
      if (!iree_status_is_ok(s)) {
        fprintf(stderr, "[BR] import binding[%zu] export failed\n", (size_t)bi);
        fflush(stderr);
        // 3DEV: fail instead of binding the foreign buffer as a VkBuffer.
        return iree_hal_vulkan_direct_command_buffer_bridge_error(
            (int)bi, src,
            src_device ? "export from owning allocator"
                       : "host pointer export (no placement.device)",
            s);
      }

      
      iree_hal_buffer_params_t params;
      memset(&params, 0, sizeof(params));
      params.type = iree_hal_buffer_memory_type(allocated);
      params.access = IREE_HAL_MEMORY_ACCESS_ALL;


      // Imported memory is host-visible (host pointer or HOST_VISIBLE dma-buf),
      // so allow persistent mapping as well; host-side copies (e.g. tooling
      // output transfer via block_command_buffer) map with PERSISTENT mode.
      params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                     IREE_HAL_BUFFER_USAGE_TRANSFER |
                     IREE_HAL_BUFFER_USAGE_MAPPING |
                     IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT;

                     
      params.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
      iree_hal_buffer_t* imported = NULL;
      s = iree_hal_allocator_import_buffer(
          command_buffer->device_allocator, params, &ext,
          iree_hal_buffer_release_callback_null(), &imported);



      if (!iree_status_is_ok(s)) {
        fprintf(stderr, "[BR] import binding[%zu] import failed type=%d",
                (size_t)bi, (int)ext.type);
        if (ext.type == IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION) {
          fprintf(stderr, " ptr=%p ptr%%4096=%zu size=%zu size%%4096=%zu",
                  ext.handle.host_allocation.ptr,
                  (size_t)((uintptr_t)ext.handle.host_allocation.ptr % 4096),
                  (size_t)ext.size, (size_t)(ext.size % 4096));
        }
        fprintf(stderr, "\n[BR]   reason: ");
        iree_status_fprint(stderr, s);
        fprintf(stderr, "\n");
        fflush(stderr);
        // 3DEV: fail instead of binding the foreign buffer as a VkBuffer.
        return iree_hal_vulkan_direct_command_buffer_bridge_error(
            (int)bi, src, "import into vulkan allocator", s);
      }


      if (ext.type == IREE_HAL_EXTERNAL_BUFFER_TYPE_OPAQUE_FD) {
        fprintf(stderr, "[BR] import binding[%zu] ok fd=%d\n", (size_t)bi,
                ext.handle.opaque_fd.fd);
      } else {
        fprintf(stderr, "[BR] import binding[%zu] ok host_ptr=%p\n",
                (size_t)bi, ext.handle.host_allocation.ptr);
      }



      fflush(stderr);
      IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert(
          command_buffer->resource_set, 1, &imported));
      iree_hal_buffer_release(imported);
      patched_bindings[bi].buffer = imported;
      any_imported = true;
    }
    if (any_imported) {
      bindings.values = patched_bindings;
    }
  }
  // 3DEV: every binding must now be a Vulkan buffer (this also covers the
  // >8 bindings case where the BRIDGE above does not run).
  for (iree_host_size_t bi = 0; bi < bindings.count; ++bi) {
    IREE_RETURN_IF_ERROR(
        iree_hal_vulkan_direct_command_buffer_require_vulkan_buffer(
            "dispatch", "binding", (int)bi, bindings.values[bi].buffer));
  }
  if (iree_hal_dispatch_uses_indirect_parameters(flags)) {
    IREE_RETURN_IF_ERROR(
        iree_hal_vulkan_direct_command_buffer_require_vulkan_buffer(
            "dispatch", "workgroup_count", -1,
            config.workgroup_count_ref.buffer));
  }
  // TODO: we can support CUSTOM_DIRECT_ARGUMENTS pretty easily. Indirect
  // arguments may require a uniform buffer and shader changes.
  if (iree_hal_dispatch_uses_custom_arguments(flags)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "direct/indirect arguments are not supported in Vulkan yet");
  }

  const iree_hal_vulkan_pipeline_t* pipeline = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_native_executable_lookup_pipeline(
      executable, export_ordinal, &pipeline));

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE
  iree_hal_vulkan_source_location_t source_location = pipeline->source_location;
  IREE_VULKAN_TRACE_ZONE_BEGIN_EXTERNAL(
      command_buffer->tracing_context, command_buffer->handle,
      source_location.file_name.data, source_location.file_name.size,
      source_location.line, source_location.func_name.data,
      source_location.func_name.size, /*name=*/NULL, 0);
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE

  // Retain executable.
  IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert(
      command_buffer->resource_set, 1, &executable));

  // Retain executable and workgroup count buffer (if indirect).
  iree_host_size_t resource_count = 1;
  const void* resources[2] = {executable, NULL};
  if (iree_hal_dispatch_uses_indirect_parameters(flags)) {
    resources[resource_count++] = config.workgroup_count_ref.buffer;
  }
  IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert(
      command_buffer->resource_set, resource_count, resources));

  // Update push constants.
  if (!iree_const_byte_span_is_empty(constants)) {
    VkPipelineLayout pipeline_layout_handle =
        iree_hal_vulkan_pipeline_layout_handle(pipeline->layout);
    command_buffer->syms->vkCmdPushConstants(
        command_buffer->handle, pipeline_layout_handle,
        VK_SHADER_STAGE_COMPUTE_BIT, (uint32_t)0,
        (uint32_t)constants.data_length, constants.data);
  }

  // Retain bound buffers until the command buffer is reset.
  IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert_strided(
      command_buffer->resource_set, bindings.count, bindings.values,
      offsetof(iree_hal_buffer_ref_t, buffer), sizeof(iree_hal_buffer_ref_t)));

  // Either allocate, update, and bind a descriptor set or use push descriptor
  // sets to use the command buffer pool when supported.
  IREE_RETURN_IF_ERROR(command_buffer->descriptor_set_arena.BindDescriptorSet(
      command_buffer->handle, pipeline->layout, 0, bindings.count,
      bindings.values));

  // Bind and dispatch the pipeline.
  command_buffer->syms->vkCmdBindPipeline(
      command_buffer->handle, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->handle);
  if (iree_hal_dispatch_uses_indirect_parameters(flags)) {
    VkBuffer workgroup_count_buffer =
        iree_hal_vulkan_buffer_handle(config.workgroup_count_ref.buffer);
    iree_device_size_t workgroup_count_offset =
        iree_hal_buffer_byte_offset(config.workgroup_count_ref.buffer) +
        config.workgroup_count_ref.offset;
    command_buffer->syms->vkCmdDispatchIndirect(
        command_buffer->handle, workgroup_count_buffer, workgroup_count_offset);
  } else {
    command_buffer->syms->vkCmdDispatch(
        command_buffer->handle, config.workgroup_count[0],
        config.workgroup_count[1], config.workgroup_count[2]);
  }

  IREE_VULKAN_TRACE_ZONE_END(command_buffer->tracing_context,
                             command_buffer->handle);

  return iree_ok_status();
}

namespace {
const iree_hal_command_buffer_vtable_t
    iree_hal_vulkan_direct_command_buffer_vtable = {
        /*.destroy=*/iree_hal_vulkan_direct_command_buffer_destroy,
        /*.begin=*/iree_hal_vulkan_direct_command_buffer_begin,
        /*.end=*/iree_hal_vulkan_direct_command_buffer_end,
        /*.begin_debug_group=*/
        iree_hal_vulkan_direct_command_buffer_begin_debug_group,
        /*.end_debug_group=*/
        iree_hal_vulkan_direct_command_buffer_end_debug_group,
        /*.execution_barrier=*/
        iree_hal_vulkan_direct_command_buffer_execution_barrier,
        /*.signal_event=*/
        iree_hal_vulkan_direct_command_buffer_signal_event,
        /*.reset_event=*/iree_hal_vulkan_direct_command_buffer_reset_event,
        /*.wait_events=*/iree_hal_vulkan_direct_command_buffer_wait_events,
        /*.advise_buffer=*/
        iree_hal_vulkan_direct_command_buffer_advise_buffer,
        /*.fill_buffer=*/iree_hal_vulkan_direct_command_buffer_fill_buffer,
        /*.update_buffer=*/
        iree_hal_vulkan_direct_command_buffer_update_buffer,
        /*.copy_buffer=*/iree_hal_vulkan_direct_command_buffer_copy_buffer,
        /*.collective=*/
        iree_hal_vulkan_direct_command_buffer_collective,
        /*.dispatch=*/iree_hal_vulkan_direct_command_buffer_dispatch,
};
}  // namespace
