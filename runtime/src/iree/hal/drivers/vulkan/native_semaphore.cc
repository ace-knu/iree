// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/native_semaphore.h"

#include <cstddef>

#include "iree/async/semaphore.h"
#include "iree/base/api.h"
#include "iree/hal/drivers/vulkan/completion_watcher.h"
#include "iree/hal/drivers/vulkan/dynamic_symbol_tables.h"
#include "iree/hal/drivers/vulkan/dynamic_symbols.h"
#include "iree/hal/drivers/vulkan/status_util.h"
#include "iree/hal/drivers/vulkan/util/ref_ptr.h"

using namespace iree::hal::vulkan;

typedef struct iree_hal_vulkan_native_semaphore_t {
  iree_async_semaphore_t async;
  VkDeviceHandle* logical_device;
  VkSemaphore handle;
  // The device's completion watcher (retained, may be NULL). See
  // iree_hal_vulkan_native_semaphore_release_before_visible.
  iree_hal_vulkan_completion_watcher_t* completion_watcher;
} iree_hal_vulkan_native_semaphore_t;

namespace {
extern const iree_hal_semaphore_vtable_t
    iree_hal_vulkan_native_semaphore_vtable;
}  // namespace

static iree_hal_vulkan_native_semaphore_t*
iree_hal_vulkan_native_semaphore_cast(iree_hal_semaphore_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_vulkan_native_semaphore_vtable);
  return (iree_hal_vulkan_native_semaphore_t*)base_value;
}

iree_status_t iree_hal_vulkan_native_semaphore_create(
    iree::hal::vulkan::VkDeviceHandle* logical_device,
    iree_async_proactor_t* proactor,
    iree_hal_vulkan_completion_watcher_t* completion_watcher,
    uint64_t initial_value, iree_hal_semaphore_t** out_semaphore) {
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(out_semaphore);
  *out_semaphore = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  VkSemaphoreTypeCreateInfo timeline_create_info;
  timeline_create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
  timeline_create_info.pNext = NULL;
  timeline_create_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  timeline_create_info.initialValue = initial_value;

  VkSemaphoreCreateInfo create_info;
  create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  create_info.pNext = &timeline_create_info;
  create_info.flags = 0;
  VkSemaphore handle = VK_NULL_HANDLE;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, VK_RESULT_TO_STATUS(logical_device->syms()->vkCreateSemaphore(
                                  *logical_device, &create_info,
                                  logical_device->allocator(), &handle),
                              "vkCreateSemaphore"));

  iree_hal_vulkan_native_semaphore_t* semaphore = NULL;
  iree_host_size_t frontier_offset = 0, total_size = 0;
  iree_status_t status = iree_async_semaphore_layout(
      sizeof(*semaphore), 0, &frontier_offset, &total_size);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(logical_device->host_allocator(), total_size,
                                   (void**)&semaphore);
  }
  if (iree_status_is_ok(status)) {
    iree_async_semaphore_initialize(
        (const iree_async_semaphore_vtable_t*)&iree_hal_vulkan_native_semaphore_vtable,
        proactor, initial_value, frontier_offset, 0, &semaphore->async);
    semaphore->logical_device = logical_device;
    semaphore->handle = handle;
    semaphore->completion_watcher = completion_watcher;
    iree_hal_vulkan_completion_watcher_retain(completion_watcher);
    *out_semaphore = iree_hal_semaphore_cast(&semaphore->async);
  } else {
    logical_device->syms()->vkDestroySemaphore(*logical_device, handle,
                                               logical_device->allocator());
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_vulkan_native_semaphore_destroy(
    iree_async_semaphore_t* base_semaphore) {
  iree_hal_vulkan_native_semaphore_t* semaphore =
      iree_hal_vulkan_native_semaphore_cast(
          iree_hal_semaphore_cast(base_semaphore));
  iree_allocator_t host_allocator = semaphore->logical_device->host_allocator();
  IREE_TRACE_ZONE_BEGIN(z0);

  semaphore->logical_device->syms()->vkDestroySemaphore(
      *semaphore->logical_device, semaphore->handle,
      semaphore->logical_device->allocator());
  iree_hal_vulkan_completion_watcher_release(semaphore->completion_watcher);

  iree_async_semaphore_deinitialize(&semaphore->async);
  iree_allocator_free(host_allocator, semaphore);

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_vulkan_native_semaphore_isa(iree_hal_semaphore_t* semaphore) {
  return iree_hal_resource_is((const iree_hal_resource_t*)semaphore,
                              &iree_hal_vulkan_native_semaphore_vtable);
}

VkSemaphore iree_hal_vulkan_native_semaphore_handle(
    iree_hal_semaphore_t* base_semaphore) {
  iree_hal_vulkan_native_semaphore_t* semaphore =
      iree_hal_vulkan_native_semaphore_cast(base_semaphore);
  return semaphore->handle;
}

// A host query or wait read |value| straight from Vulkan, without the
// completion watcher. If the watcher has not signaled that value yet, the
// completion is about to become visible outside of it (returned to the caller
// or dispatched to timepoints), so first free the resources of every completed
// submission, as the watcher does before it signals: a caller that learns of
// the completion this way (for example a CPU queue checking its waits) may
// finish the program and tear down the VM context right away, and a retained
// binding imported from module rodata would make iree_vm_buffer_deinitialize
// abort.
//
// NOTE(fork): upstream reads and dispatches here without this step.
static void iree_hal_vulkan_native_semaphore_release_before_visible(
    iree_hal_vulkan_native_semaphore_t* semaphore, uint64_t value) {
  iree_async_semaphore_t* async_sem = &semaphore->async;
  if ((int64_t)value > iree_atomic_load(&async_sem->timeline_value,
                                        iree_memory_order_acquire)) {
    iree_hal_vulkan_completion_watcher_release_completed(
        semaphore->completion_watcher);
  }
}

// Every value read from Vulkan (a query, a wait, the completion watcher) only
// moves the host-side timeline forward; the thread that moves it dispatches.
// Waiter registration (iree_async_semaphore_acquire_timepoint) re-checks the
// timeline after inserting and dispatch is edge-triggered, which is only
// correct if the timeline never decreases.
//
// NOTE(fork): upstream (#23695) stored the observed value unconditionally in
// the query, the wait and the completion watcher. A thread that read an older
// value from Vulkan could then lower the timeline after another thread had
// raised it and dispatched, and a timepoint registered afterwards never fired
// (for example a CPU queue operation waiting on a Vulkan fence, so the host
// waits that depend on it hung). Same rule as upstream main's rewritten driver
// (semaphore.c iree_hal_vulkan_semaphore_observe_native_value, 9e2ac9142e).
uint64_t iree_hal_vulkan_native_semaphore_observe_value(
    iree_async_semaphore_t* semaphore, uint64_t value) {
  int64_t current_raw = 0;
  do {
    current_raw =
        iree_atomic_load(&semaphore->timeline_value, iree_memory_order_acquire);
    if (value <= (uint64_t)current_raw) return (uint64_t)current_raw;
  } while (!iree_atomic_compare_exchange_weak(
      &semaphore->timeline_value, &current_raw, (int64_t)value,
      iree_memory_order_release, iree_memory_order_relaxed));
  iree_async_semaphore_dispatch_timepoints(semaphore, value);
  return value;
}

static uint64_t iree_hal_vulkan_native_semaphore_query(
    iree_async_semaphore_t* base_semaphore) {
  iree_hal_vulkan_native_semaphore_t* semaphore =
      iree_hal_vulkan_native_semaphore_cast(
          iree_hal_semaphore_cast(base_semaphore));
  iree_async_semaphore_t* async_sem = base_semaphore;

  // Check for failure first so we can encode the actual failure status.
  iree_status_t failure_status = (iree_status_t)iree_atomic_load(
      &async_sem->failure_status, iree_memory_order_acquire);
  if (!iree_status_is_ok(failure_status)) {
    return iree_hal_status_as_semaphore_failure(failure_status);
  }

  // Query from Vulkan source-of-truth.
  uint64_t value = 0;
  VkResult result =
      semaphore->logical_device->syms()->vkGetSemaphoreCounterValue(
          *semaphore->logical_device, semaphore->handle, &value);
  if (result != VK_SUCCESS) {
    return iree_hal_status_as_semaphore_failure(
        iree_status_from_code(IREE_STATUS_INTERNAL));
  }

  // If the semaphore has failed return the encoded failure status.
  if (value >= IREE_HAL_SEMAPHORE_FAILURE_VALUE) {
    // Re-check the failure status as it may have been set between our first
    // check and the Vulkan query.
    failure_status = (iree_status_t)iree_atomic_load(&async_sem->failure_status,
                                                     iree_memory_order_acquire);
    if (!iree_status_is_ok(failure_status)) {
      return iree_hal_status_as_semaphore_failure(failure_status);
    }
    return IREE_HAL_SEMAPHORE_FAILURE_VALUE;
  }

  iree_hal_vulkan_native_semaphore_release_before_visible(semaphore, value);

  // Raise the host-side timeline to the Vulkan value (never lower it) and
  // dispatch the timepoints it satisfies. This keeps latency low on the
  // device-side signal path (Vulkan does not notify us of device-side
  // signals). Returns the host-side value, which another thread may already
  // have raised past what we read.
  return iree_hal_vulkan_native_semaphore_observe_value(base_semaphore, value);
}

static iree_status_t iree_hal_vulkan_native_semaphore_signal(
    iree_async_semaphore_t* base_semaphore, uint64_t new_value,
    const iree_async_frontier_t* frontier) {
  iree_hal_vulkan_native_semaphore_t* semaphore =
      iree_hal_vulkan_native_semaphore_cast(
          iree_hal_semaphore_cast(base_semaphore));

  // Advance the software timeline (CAS) and merge frontier first, then the
  // VkSemaphore. Each timeline value must be signaled exactly once — CAS
  // failure here indicates a structural error (duplicate signal or
  // non-monotonic scheduling). Observers (query, wait, completion watcher)
  // only raise the host-side timeline to values they read from Vulkan, so
  // with this order none of them can get ahead of this signal and make the
  // CAS fail.
  //
  // NOTE(fork): upstream (#23695) signaled the VkSemaphore first, leaving a
  // window in which an observer could publish |new_value| before the CAS here.
  // Same order as upstream main's rewritten driver (semaphore.c
  // iree_hal_vulkan_semaphore_signal, 9e2ac9142e).
  IREE_RETURN_IF_ERROR(iree_async_semaphore_advance_timeline(
      base_semaphore, new_value, frontier));

  VkSemaphoreSignalInfo signal_info;
  signal_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
  signal_info.pNext = NULL;
  signal_info.semaphore = semaphore->handle;
  signal_info.value = new_value;
  iree_status_t status =
      VK_RESULT_TO_STATUS(semaphore->logical_device->syms()->vkSignalSemaphore(
                              *semaphore->logical_device, &signal_info),
                          "vkSignalSemaphore");
  if (!iree_status_is_ok(status)) {
    // Vulkan signal failed — fail the async semaphore to notify waiters.
    // iree_async_semaphore_fail takes ownership of the clone.
    iree_async_semaphore_fail(base_semaphore, iree_status_clone(status));
    return status;
  }

  iree_async_semaphore_dispatch_timepoints(base_semaphore, new_value);
  return iree_ok_status();
}

static void iree_hal_vulkan_native_semaphore_on_fail(
    iree_async_semaphore_t* base_semaphore, iree_status_code_t status_code) {
  (void)status_code;
  iree_hal_vulkan_native_semaphore_t* semaphore =
      iree_hal_vulkan_native_semaphore_cast(
          iree_hal_semaphore_cast(base_semaphore));

  // Signal the VkSemaphore with the failure sentinel so Vulkan-side waiters
  // wake. We don't care about the result — we're already failing.
  VkSemaphoreSignalInfo signal_info;
  signal_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
  signal_info.pNext = NULL;
  signal_info.semaphore = semaphore->handle;
  signal_info.value = IREE_HAL_SEMAPHORE_FAILURE_VALUE;
  semaphore->logical_device->syms()->vkSignalSemaphore(
      *semaphore->logical_device, &signal_info);
}

static iree_status_t iree_hal_vulkan_native_semaphore_wait(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_timeout_t timeout, iree_async_wait_flags_t flags) {
  iree_hal_vulkan_native_semaphore_t* semaphore =
      iree_hal_vulkan_native_semaphore_cast(base_semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Convert timeout to Vulkan nanoseconds.
  iree_time_t deadline_ns = iree_timeout_as_deadline_ns(timeout);
  uint64_t timeout_ns;
  if (deadline_ns == IREE_TIME_INFINITE_FUTURE) {
    timeout_ns = UINT64_MAX;
  } else if (deadline_ns == IREE_TIME_INFINITE_PAST) {
    timeout_ns = 0;
  } else {
    iree_time_t now_ns = iree_time_now();
    timeout_ns = deadline_ns < now_ns ? 0 : (uint64_t)(deadline_ns - now_ns);
  }

  // Direct vkWaitSemaphores — no thread hop through the completion watcher.
  VkSemaphore vk_semaphore = semaphore->handle;
  VkSemaphoreWaitInfo wait_info;
  wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
  wait_info.pNext = NULL;
  wait_info.flags = 0;
  wait_info.semaphoreCount = 1;
  wait_info.pSemaphores = &vk_semaphore;
  wait_info.pValues = &value;
  VkResult result = semaphore->logical_device->syms()->vkWaitSemaphores(
      *semaphore->logical_device, &wait_info, timeout_ns);

  if (result == VK_SUCCESS) {
    iree_async_semaphore_t* async_sem = (iree_async_semaphore_t*)base_semaphore;

    // The wait saw the value straight from Vulkan; returning makes it visible
    // to the caller. Free the resources of every completed submission first
    // (see iree_hal_vulkan_native_semaphore_release_before_visible). A blocking
    // wait is not a hot path, so do it unconditionally.
    iree_hal_vulkan_completion_watcher_release_completed(
        semaphore->completion_watcher);

    // Check if the semaphore was failed (on_fail signals the VkSemaphore to
    // IREE_HAL_SEMAPHORE_FAILURE_VALUE which satisfies any vkWaitSemaphores
    // call). The failure_status check must happen after vkWaitSemaphores
    // returns to avoid racing with the fail path.
    iree_status_t failure_status = (iree_status_t)iree_atomic_load(
        &async_sem->failure_status, iree_memory_order_acquire);
    if (IREE_UNLIKELY(!iree_status_is_ok(failure_status))) {
      IREE_TRACE_ZONE_END(z0);
      return iree_status_clone(failure_status);
    }

    // Sync the async timeline so timepoints waiting on this value dispatch
    // immediately rather than waiting for the completion watcher's next pass.
    // Only raises it (see iree_hal_vulkan_native_semaphore_observe_value).
    uint64_t current_value = 0;
    VkResult query_result =
        semaphore->logical_device->syms()->vkGetSemaphoreCounterValue(
            *semaphore->logical_device, semaphore->handle, &current_value);
    if (query_result == VK_SUCCESS &&
        current_value < IREE_HAL_SEMAPHORE_FAILURE_VALUE) {
      iree_hal_vulkan_native_semaphore_observe_value(async_sem, current_value);
    }
  }

  IREE_TRACE_ZONE_END(z0);

  if (result == VK_SUCCESS) {
    return iree_ok_status();
  } else if (result == VK_TIMEOUT) {
    return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
  }
  return VK_RESULT_TO_STATUS(result, "vkWaitSemaphores");
}

IREE_API_EXPORT iree_status_t iree_hal_vulkan_semaphore_handle(
    iree_hal_semaphore_t* base_semaphore, VkSemaphore* out_handle) {
  IREE_ASSERT_ARGUMENT(base_semaphore);
  IREE_ASSERT_ARGUMENT(out_handle);
  iree_hal_vulkan_native_semaphore_t* semaphore =
      iree_hal_vulkan_native_semaphore_cast(base_semaphore);
  *out_handle = semaphore->handle;
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_semaphore_import_timepoint(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_external_timepoint_t external_timepoint) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "timepoint import is not yet implemented");
}

static iree_status_t iree_hal_vulkan_semaphore_export_timepoint(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_external_timepoint_type_t requested_type,
    iree_hal_external_timepoint_flags_t requested_flags,
    iree_hal_external_timepoint_t* IREE_RESTRICT out_external_timepoint) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "timepoint export is not yet implemented");
}

namespace {
const iree_hal_semaphore_vtable_t iree_hal_vulkan_native_semaphore_vtable = {
    /*.async=*/
    {
        /*.destroy=*/iree_hal_vulkan_native_semaphore_destroy,
        /*.query=*/iree_hal_vulkan_native_semaphore_query,
        /*.signal=*/iree_hal_vulkan_native_semaphore_signal,
        /*.on_fail=*/iree_hal_vulkan_native_semaphore_on_fail,
    },
    /*.wait=*/iree_hal_vulkan_native_semaphore_wait,
    /*.import_timepoint=*/iree_hal_vulkan_semaphore_import_timepoint,
    /*.export_timepoint=*/iree_hal_vulkan_semaphore_export_timepoint,
};
}  // namespace
