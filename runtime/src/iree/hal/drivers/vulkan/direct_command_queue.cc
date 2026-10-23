// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/direct_command_queue.h"
extern iree_hal_buffer_ref_t g_last_bindings[8];
extern iree_host_size_t g_last_binding_count;

#include <cstdio>

#include <cstdint>

#include "iree/hal/drivers/vulkan/direct_command_buffer.h"
#include "iree/hal/drivers/vulkan/dynamic_symbols.h"
#include "iree/hal/drivers/vulkan/native_semaphore.h"
#include "iree/hal/drivers/vulkan/status_util.h"
#include "iree/hal/drivers/vulkan/tracing.h"
#include "iree/hal/drivers/vulkan/util/ref_ptr.h"

namespace iree {
namespace hal {
namespace vulkan {

DirectCommandQueue::DirectCommandQueue(
    VkDeviceHandle* logical_device,
    iree_hal_command_category_t supported_categories, VkQueue queue)
    : CommandQueue(logical_device, supported_categories, queue) {}

DirectCommandQueue::~DirectCommandQueue() = default;

iree_status_t DirectCommandQueue::TranslateBatchInfo(
    const iree_hal_vulkan_submission_batch_t* batch, VkSubmitInfo* submit_info,
    VkTimelineSemaphoreSubmitInfo* timeline_submit_info, Arena* arena,
    bool* out_has_foreign_signal) {
  // TODO(benvanik): see if we can go to finer-grained stages.
  // For example, if this was just queue ownership transfers then we can use
  // the pseudo-stage of VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT.
  VkPipelineStageFlags dst_stage_mask =
      VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

  auto wait_semaphore_handles =
      arena->AllocateSpan<VkSemaphore>(batch->wait_semaphores.count);
  auto wait_semaphore_values =
      arena->AllocateSpan<uint64_t>(batch->wait_semaphores.count);
  auto wait_dst_stage_masks =
      arena->AllocateSpan<VkPipelineStageFlags>(batch->wait_semaphores.count);
  iree_host_size_t wait_count = 0;
  for (iree_host_size_t i = 0; i < batch->wait_semaphores.count; ++i) {
    iree_hal_semaphore_t* semaphore = batch->wait_semaphores.semaphores[i];
    uint64_t value = batch->wait_semaphores.payload_values[i];
    if (!iree_hal_vulkan_native_semaphore_isa(semaphore)) {
      // BRIDGE(L1): semaphore owned by another device; timeline import is not
      // implemented so block on the host until it reaches the value.
      fprintf(stderr, "[BR] vulkan wait-foreign sem=%p value=%llu\n", (void*)semaphore, (unsigned long long)value); fflush(stderr);
      IREE_RETURN_IF_ERROR(
          iree_hal_semaphore_wait(semaphore, value, iree_infinite_timeout(),
                                  IREE_ASYNC_WAIT_FLAG_NONE));
      continue;
    }
    wait_semaphore_handles[wait_count] =
        iree_hal_vulkan_native_semaphore_handle(semaphore);
    wait_semaphore_values[wait_count] = value;
    wait_dst_stage_masks[wait_count] = dst_stage_mask;
    ++wait_count;
  }

  auto signal_semaphore_handles =
      arena->AllocateSpan<VkSemaphore>(batch->signal_semaphores.count);
  auto signal_semaphore_values =
      arena->AllocateSpan<uint64_t>(batch->signal_semaphores.count);
  iree_host_size_t signal_count = 0;
  bool has_foreign_signal = false;
  for (iree_host_size_t i = 0; i < batch->signal_semaphores.count; ++i) {
    iree_hal_semaphore_t* semaphore = batch->signal_semaphores.semaphores[i];
    if (!iree_hal_vulkan_native_semaphore_isa(semaphore)) {
      // BRIDGE(L1): signaled from the host after the submission completes.
      has_foreign_signal = true;
      fprintf(stderr, "[BR] vulkan signal-foreign sem=%p\n", (void*)semaphore); fflush(stderr);
      continue;
    }
    signal_semaphore_handles[signal_count] =
        iree_hal_vulkan_native_semaphore_handle(semaphore);
    signal_semaphore_values[signal_count] =
        batch->signal_semaphores.payload_values[i];
    ++signal_count;
  }
  *out_has_foreign_signal = has_foreign_signal;

  auto command_buffer_handles =
      arena->AllocateSpan<VkCommandBuffer>(batch->command_buffer_count);
  for (iree_host_size_t i = 0; i < batch->command_buffer_count; ++i) {
    command_buffer_handles[i] =
        iree_hal_vulkan_direct_command_buffer_handle(batch->command_buffers[i]);
  }

  submit_info->sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info->pNext = timeline_submit_info;
  submit_info->waitSemaphoreCount =
      static_cast<uint32_t>(wait_count);
  submit_info->pWaitSemaphores = wait_semaphore_handles.data();
  submit_info->pWaitDstStageMask = wait_dst_stage_masks.data();
  submit_info->commandBufferCount =
      static_cast<uint32_t>(command_buffer_handles.size());
  submit_info->pCommandBuffers = command_buffer_handles.data();
  submit_info->signalSemaphoreCount =
      static_cast<uint32_t>(signal_count);
  submit_info->pSignalSemaphores = signal_semaphore_handles.data();

  timeline_submit_info->sType =
      VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
  timeline_submit_info->pNext = nullptr;
  timeline_submit_info->waitSemaphoreValueCount =
      static_cast<uint32_t>(wait_count);
  timeline_submit_info->pWaitSemaphoreValues = wait_semaphore_values.data();
  timeline_submit_info->signalSemaphoreValueCount =
      static_cast<uint32_t>(signal_count);
  timeline_submit_info->pSignalSemaphoreValues = signal_semaphore_values.data();

  return iree_ok_status();
}

iree_status_t DirectCommandQueue::Submit(
    iree_host_size_t batch_count,
    const iree_hal_vulkan_submission_batch_t* batches) {
  IREE_TRACE_SCOPE_NAMED("DirectCommandQueue::Submit");

  // Map the submission batches to VkSubmitInfos.
  // Note that we must keep all arrays referenced alive until submission
  // completes and since there are a bunch of them we use an arena.
  Arena arena(4 * 1024);
  auto submit_infos = arena.AllocateSpan<VkSubmitInfo>(batch_count);
  auto timeline_submit_infos =
      arena.AllocateSpan<VkTimelineSemaphoreSubmitInfo>(batch_count);
  bool any_foreign_signal = false;
  for (int i = 0; i < batch_count; ++i) {
    bool has_foreign_signal = false;
    IREE_RETURN_IF_ERROR(TranslateBatchInfo(&batches[i], &submit_infos[i],
                                            &timeline_submit_infos[i], &arena,
                                            &has_foreign_signal));
    any_foreign_signal |= has_foreign_signal;
  }

  // BRIDGE(L1): if any batch signals a semaphore owned by another device we
  // attach a fence, wait for it on the host, then signal those semaphores.
  VkFence fence = VK_NULL_HANDLE;
  if (any_foreign_signal) {
    VkFenceCreateInfo fence_info;
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.pNext = nullptr;
    fence_info.flags = 0;
    IREE_RETURN_IF_ERROR(VK_RESULT_TO_STATUS(
        syms()->vkCreateFence(*logical_device_, &fence_info,
                              logical_device_->allocator(), &fence),
        "vkCreateFence"));
  }

  iree_slim_mutex_lock(&queue_mutex_);
  iree_status_t status = VK_RESULT_TO_STATUS(
      syms()->vkQueueSubmit(queue_, static_cast<uint32_t>(submit_infos.size()),
                            submit_infos.data(), fence),
      "vkQueueSubmit");
  iree_slim_mutex_unlock(&queue_mutex_);
  for (iree_host_size_t b = 0; b < batch_count; ++b) {
    fprintf(stderr, "[BR] vk submit cmdbufs=%zu waits=%zu signals=%zu ok=%d t=%.3fms\n",
            (size_t)batches[b].command_buffer_count, (size_t)batches[b].wait_semaphores.count,
            (size_t)batches[b].signal_semaphores.count, (int)iree_status_is_ok(status),
            iree_time_now() / 1e6);
  }
  fflush(stderr);
  if (fence != VK_NULL_HANDLE) {
    if (iree_status_is_ok(status)) {
      status = VK_RESULT_TO_STATUS(
          syms()->vkWaitForFences(*logical_device_, 1, &fence, VK_TRUE,
                                  UINT64_MAX),
          "vkWaitForFences");
    }
    syms()->vkDestroyFence(*logical_device_, fence,
                           logical_device_->allocator());
    if (iree_status_is_ok(status)) {
      for (iree_host_size_t b = 0; b < batch_count; ++b) {
        const iree_hal_semaphore_list_t& signals = batches[b].signal_semaphores;
        for (iree_host_size_t i = 0; i < signals.count; ++i) {
          if (iree_hal_vulkan_native_semaphore_isa(signals.semaphores[i])) {
            continue;
          }
          status = iree_hal_semaphore_signal(signals.semaphores[i],
                                             signals.payload_values[i], NULL);
          if (!iree_status_is_ok(status)) break;
        }
        if (!iree_status_is_ok(status)) break;
      }
    }
  }
  IREE_RETURN_IF_ERROR(status);

  return iree_ok_status();
}

iree_status_t DirectCommandQueue::WaitIdle(iree_timeout_t timeout) {
  iree_time_t deadline_ns = iree_timeout_as_deadline_ns(timeout);
  if (deadline_ns == IREE_TIME_INFINITE_FUTURE) {
    // Fast path for using vkQueueWaitIdle, which is usually cheaper (as it
    // requires fewer calls into the driver).
    IREE_TRACE_SCOPE_NAMED("DirectCommandQueue::WaitIdle#vkQueueWaitIdle");
    iree_slim_mutex_lock(&queue_mutex_);
    iree_status_t status =
        VK_RESULT_TO_STATUS(syms()->vkQueueWaitIdle(queue_), "vkQueueWaitIdle");
    iree_slim_mutex_unlock(&queue_mutex_);
    iree_hal_vulkan_tracing_context_collect(tracing_context(), VK_NULL_HANDLE);
    return status;
  }

  IREE_TRACE_SCOPE_NAMED("DirectCommandQueue::WaitIdle#Fence");

  // Create a new fence just for this wait. This keeps us thread-safe as the
  // behavior of wait+reset is racey.
  VkFenceCreateInfo create_info;
  create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  create_info.pNext = nullptr;
  create_info.flags = 0;
  VkFence fence = VK_NULL_HANDLE;
  VK_RETURN_IF_ERROR(
      syms()->vkCreateFence(*logical_device_, &create_info,
                            logical_device_->allocator(), &fence),
      "vkCreateFence");

  uint64_t timeout_ns;
  if (deadline_ns == IREE_TIME_INFINITE_PAST) {
    // Do not wait.
    timeout_ns = 0;
  } else if (deadline_ns == IREE_TIME_INFINITE_FUTURE) {
    // Wait forever.
    timeout_ns = UINT64_MAX;
  } else {
    // Convert to relative time in nanoseconds.
    // The implementation may not wait with this granularity (like by 10000x).
    iree_time_t now_ns = iree_time_now();
    if (deadline_ns < now_ns) {
      return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
    }
    timeout_ns = (uint64_t)(deadline_ns - now_ns);
  }

  iree_slim_mutex_lock(&queue_mutex_);
  iree_status_t status = VK_RESULT_TO_STATUS(
      syms()->vkQueueSubmit(queue_, 0, nullptr, fence), "vkQueueSubmit");
  iree_slim_mutex_unlock(&queue_mutex_);

  if (iree_status_is_ok(status)) {
    VkResult result = syms()->vkWaitForFences(*logical_device_, 1, &fence,
                                              VK_TRUE, timeout_ns);
    switch (result) {
      case VK_SUCCESS:
        status = iree_ok_status();
        break;
      case VK_TIMEOUT:
        status = iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
        break;
      default:
        status = VK_RESULT_TO_STATUS(result, "vkWaitForFences");
        break;
    }
  }

  syms()->vkDestroyFence(*logical_device_, fence, logical_device_->allocator());

  iree_hal_vulkan_tracing_context_collect(tracing_context(), VK_NULL_HANDLE);

  return status;
}

}  // namespace vulkan
}  // namespace hal
}  // namespace iree
