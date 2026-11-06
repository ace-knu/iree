// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/completion_watcher.h"

#include <cstddef>
#include <cstring>

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/threading/mutex.h"
#include "iree/base/threading/thread.h"
#include "iree/hal/drivers/vulkan/dynamic_symbols.h"
#include "iree/hal/drivers/vulkan/native_semaphore.h"
#include "iree/hal/drivers/vulkan/status_util.h"

using namespace iree::hal::vulkan;

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

// Initial capacity for the pending and active entry arrays.
#define IREE_HAL_VULKAN_COMPLETION_WATCHER_INITIAL_CAPACITY 64

// Timeout for vkWaitSemaphores in nanoseconds. Periodic wakeups serve as a
// liveness check (some drivers don't return VK_ERROR_DEVICE_LOST from wait).
#define IREE_HAL_VULKAN_COMPLETION_WATCHER_WAIT_TIMEOUT_NS \
  (5ull * 1000000000ull)  // 5 seconds

//===----------------------------------------------------------------------===//
// Completion entry
//===----------------------------------------------------------------------===//

// Resources of one registered submission, shared by the entries of all of its
// signal semaphores (one vkQueueSubmit signals them all).
typedef struct iree_hal_vulkan_completion_batch_t {
  // Resources retained until the submission completes. Guarded by the watcher
  // mutex. While non-NULL the batch is linked into the watcher's in-flight
  // list. Freed by the first thread that observes the submission complete in
  // iree_hal_vulkan_completion_watcher_release_completed (before anything
  // signals), by fail_all, or with the last entry if the submission never
  // completed (its semaphores were failed).
  iree_hal_resource_set_t* resource_set;
  // One of the submission's signal semaphores and its target value, used to
  // check whether the submission completed. All signal semaphores of one
  // vkQueueSubmit are signaled together, so one is enough.
  VkSemaphore vk_semaphore;
  uint64_t target_value;
  // In-flight list links. Guarded by the watcher mutex.
  struct iree_hal_vulkan_completion_batch_t* prev;
  struct iree_hal_vulkan_completion_batch_t* next;
  bool in_flight;
  // Entries still referencing this batch. The last one frees the batch. Only
  // the watcher thread changes it after registration.
  iree_host_size_t entry_count;
} iree_hal_vulkan_completion_batch_t;

typedef struct iree_hal_vulkan_completion_entry_t {
  // HAL semaphore being watched. Retained by the watcher.
  iree_hal_semaphore_t* semaphore;
  // VkSemaphore handle (cached from semaphore for direct use in wait arrays).
  VkSemaphore vk_semaphore;
  // Target timeline value that indicates this entry's work is complete.
  uint64_t target_value;
  // Resources of the submission this entry belongs to.
  iree_hal_vulkan_completion_batch_t* batch;
} iree_hal_vulkan_completion_entry_t;

//===----------------------------------------------------------------------===//
// Watcher state
//===----------------------------------------------------------------------===//

struct iree_hal_vulkan_completion_watcher_t {
  // The device holds one reference and each Vulkan semaphore holds one, so a
  // semaphore that outlives the device can still call
  // iree_hal_vulkan_completion_watcher_release_completed safely (it finds
  // nothing in flight after the device shut the watcher down).
  iree_atomic_ref_count_t ref_count;
  iree_thread_t* thread;
  iree_allocator_t host_allocator;

  // Logical device handle for Vulkan API calls. Borrowed — the device outlives
  // the watcher.
  VkDeviceHandle* logical_device;

  // Poke semaphore: a dedicated VkTimelineSemaphore used solely to wake the
  // watcher thread when new entries are registered or shutdown is requested.
  VkSemaphore poke_semaphore;
  // Next value to signal on the poke semaphore. Incremented atomically by
  // registration calls.
  iree_atomic_int64_t poke_next_value;
  // Value the watcher last observed on the poke semaphore. Used to compute
  // the wait target (poke_last_value + 1). Watcher-local, no sync needed.
  uint64_t poke_last_value;

  // Protects pending_entries, pending_count, do_exit, and the in-flight list
  // (including each batch's resource_set). Held briefly by submitters to append
  // entries, by the watcher to drain them, and by any thread that frees the
  // resources of completed submissions (the frees run under it).
  iree_slim_mutex_t mutex;
  // Submissions whose resources are still retained, newest first.
  iree_hal_vulkan_completion_batch_t* in_flight_head;
  iree_hal_vulkan_completion_entry_t* pending_entries;
  iree_host_size_t pending_count;
  iree_host_size_t pending_capacity;
  bool do_exit;

  // Active entries being watched. Owned exclusively by the watcher thread —
  // no lock needed.
  iree_hal_vulkan_completion_entry_t* active_entries;
  iree_host_size_t active_count;
  iree_host_size_t active_capacity;

  // Scratch arrays rebuilt each iteration for vkWaitSemaphores. Sized to
  // active_capacity + 1 (the extra slot is for the poke semaphore).
  VkSemaphore* wait_semaphores;
  uint64_t* wait_values;
};

//===----------------------------------------------------------------------===//
// Internal helpers
//===----------------------------------------------------------------------===//

// Ensures the active arrays have room for at least |required_capacity| entries
// plus 1 (for the poke semaphore slot in the wait arrays). Grows by doubling.
// All three arrays (active_entries, wait_semaphores, wait_values) are allocated
// together in a single allocation to avoid partial-update bugs.
static iree_status_t iree_hal_vulkan_completion_watcher_grow_active(
    iree_hal_vulkan_completion_watcher_t* watcher,
    iree_host_size_t required_capacity) {
  if (required_capacity <= watcher->active_capacity) return iree_ok_status();

  iree_host_size_t new_capacity =
      iree_max(watcher->active_capacity * 2,
               iree_max(required_capacity,
                        IREE_HAL_VULKAN_COMPLETION_WATCHER_INITIAL_CAPACITY));

  // Single allocation for all three arrays. Wait arrays have +1 for poke.
  iree_host_size_t wait_count = new_capacity + 1;
  iree_host_size_t entries_size =
      new_capacity * sizeof(iree_hal_vulkan_completion_entry_t);
  iree_host_size_t semaphores_size = wait_count * sizeof(VkSemaphore);
  iree_host_size_t values_size = wait_count * sizeof(uint64_t);
  iree_host_size_t total_size = entries_size + semaphores_size + values_size;
  uint8_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(watcher->host_allocator,
                                             total_size, (void**)&buffer));

  iree_hal_vulkan_completion_entry_t* new_active =
      (iree_hal_vulkan_completion_entry_t*)buffer;
  VkSemaphore* new_wait_semaphores = (VkSemaphore*)(buffer + entries_size);
  uint64_t* new_wait_values =
      (uint64_t*)(buffer + entries_size + semaphores_size);

  if (watcher->active_count > 0) {
    memcpy(new_active, watcher->active_entries,
           watcher->active_count * sizeof(iree_hal_vulkan_completion_entry_t));
  }

  // Free old combined allocation (active_entries is the base pointer).
  iree_allocator_free(watcher->host_allocator, watcher->active_entries);
  watcher->active_entries = new_active;
  watcher->wait_semaphores = new_wait_semaphores;
  watcher->wait_values = new_wait_values;
  watcher->active_capacity = new_capacity;
  return iree_ok_status();
}

// Drains pending entries into the active set. Caller must NOT hold the mutex.
static iree_status_t iree_hal_vulkan_completion_watcher_drain_pending(
    iree_hal_vulkan_completion_watcher_t* watcher) {
  iree_slim_mutex_lock(&watcher->mutex);
  iree_host_size_t pending_count = watcher->pending_count;
  if (pending_count == 0) {
    iree_slim_mutex_unlock(&watcher->mutex);
    return iree_ok_status();
  }

  // Ensure active has room.
  iree_host_size_t required = watcher->active_count + pending_count;
  iree_status_t status =
      iree_hal_vulkan_completion_watcher_grow_active(watcher, required);
  if (iree_status_is_ok(status)) {
    memcpy(&watcher->active_entries[watcher->active_count],
           watcher->pending_entries,
           pending_count * sizeof(iree_hal_vulkan_completion_entry_t));
    watcher->active_count += pending_count;
    watcher->pending_count = 0;
  }
  iree_slim_mutex_unlock(&watcher->mutex);
  return status;
}

// The watcher whose mutex this thread holds while freeing resources. A release
// path that re-enters iree_hal_vulkan_completion_watcher_release_completed on
// the same thread returns immediately instead of deadlocking; the outer call
// finishes the frees before its caller proceeds.
static thread_local iree_hal_vulkan_completion_watcher_t*
    iree_hal_vulkan_completion_watcher_releasing = NULL;

// Unlinks |batch| from the in-flight list and frees its resources. The watcher
// mutex must be held. No-op if the resources were already freed.
static void iree_hal_vulkan_completion_batch_retire_locked(
    iree_hal_vulkan_completion_watcher_t* watcher,
    iree_hal_vulkan_completion_batch_t* batch) {
  if (!batch->in_flight) return;
  if (batch->prev) {
    batch->prev->next = batch->next;
  } else {
    watcher->in_flight_head = batch->next;
  }
  if (batch->next) batch->next->prev = batch->prev;
  batch->prev = NULL;
  batch->next = NULL;
  batch->in_flight = false;
  iree_hal_resource_set_t* resource_set = batch->resource_set;
  batch->resource_set = NULL;
  iree_hal_vulkan_completion_watcher_t* outer =
      iree_hal_vulkan_completion_watcher_releasing;
  iree_hal_vulkan_completion_watcher_releasing = watcher;
  iree_hal_resource_set_free(resource_set);
  iree_hal_vulkan_completion_watcher_releasing = outer;
}

// Frees the resources of every in-flight submission. The watcher mutex must be
// held.
static void iree_hal_vulkan_completion_watcher_retire_all_locked(
    iree_hal_vulkan_completion_watcher_t* watcher) {
  while (watcher->in_flight_head) {
    iree_hal_vulkan_completion_batch_retire_locked(watcher,
                                                   watcher->in_flight_head);
  }
}

// Releases a single completion entry: releases the semaphore and, with the
// last entry of the submission, frees the batch. The submission resources are
// normally freed already (when the submission was observed complete, or in
// fail_all); if every entry was removed without the submission being observed
// complete (its semaphores were failed by another thread), the last entry frees
// them here, as before. |mutex_held| tells whether the caller holds the watcher
// mutex.
static void iree_hal_vulkan_completion_entry_release(
    iree_hal_vulkan_completion_watcher_t* watcher,
    iree_hal_vulkan_completion_entry_t* entry, bool mutex_held) {
  iree_hal_vulkan_completion_batch_t* batch = entry->batch;
  const bool is_last_entry = --batch->entry_count == 0;
  if (is_last_entry) {
    if (!mutex_held) iree_slim_mutex_lock(&watcher->mutex);
    iree_hal_vulkan_completion_batch_retire_locked(watcher, batch);
    if (!mutex_held) iree_slim_mutex_unlock(&watcher->mutex);
  }
  iree_hal_semaphore_release(entry->semaphore);
  if (is_last_entry) {
    iree_allocator_free(watcher->host_allocator, batch);
  }
  memset(entry, 0, sizeof(*entry));
}

// Swap-removes the entry at |index| from the active array, together with the
// value the scan observed for it (kept in the wait_values scratch array).
static void iree_hal_vulkan_completion_watcher_swap_remove(
    iree_hal_vulkan_completion_watcher_t* watcher, iree_host_size_t index) {
  IREE_ASSERT(index < watcher->active_count);
  watcher->active_count--;
  if (index < watcher->active_count) {
    watcher->active_entries[index] =
        watcher->active_entries[watcher->active_count];
    watcher->wait_values[index] = watcher->wait_values[watcher->active_count];
  }
}

// Fails all active and pending entries with the given status. Used during
// device loss or forced shutdown.
static void iree_hal_vulkan_completion_watcher_fail_all(
    iree_hal_vulkan_completion_watcher_t* watcher, iree_status_code_t code) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // As on completion, the submission resources are freed before a waiter can
  // observe a (failed) semaphore.
  iree_slim_mutex_lock(&watcher->mutex);
  iree_hal_vulkan_completion_watcher_retire_all_locked(watcher);
  iree_slim_mutex_unlock(&watcher->mutex);

  // Fail all active entries.
  for (iree_host_size_t i = 0; i < watcher->active_count; ++i) {
    iree_hal_semaphore_fail(watcher->active_entries[i].semaphore,
                            iree_status_from_code(code));
    iree_hal_vulkan_completion_entry_release(
        watcher, &watcher->active_entries[i], /*mutex_held=*/false);
  }
  watcher->active_count = 0;

  // Drain and fail pending entries.
  iree_slim_mutex_lock(&watcher->mutex);
  for (iree_host_size_t i = 0; i < watcher->pending_count; ++i) {
    iree_hal_semaphore_fail(watcher->pending_entries[i].semaphore,
                            iree_status_from_code(code));
    iree_hal_vulkan_completion_entry_release(
        watcher, &watcher->pending_entries[i], /*mutex_held=*/true);
  }
  watcher->pending_count = 0;
  iree_slim_mutex_unlock(&watcher->mutex);

  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Watcher thread
//===----------------------------------------------------------------------===//

static int iree_hal_vulkan_completion_watcher_thread_main(void* param) {
  iree_hal_vulkan_completion_watcher_t* watcher =
      (iree_hal_vulkan_completion_watcher_t*)param;
  auto* syms = watcher->logical_device->syms().get();
  VkDevice vk_device = *watcher->logical_device;

  while (true) {
    IREE_TRACE_ZONE_BEGIN_NAMED(z_iteration,
                                "iree_hal_vulkan_completion_watcher_iteration");

    // Drain pending entries into the active set.
    iree_status_t status =
        iree_hal_vulkan_completion_watcher_drain_pending(watcher);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      iree_hal_vulkan_completion_watcher_fail_all(watcher,
                                                  IREE_STATUS_INTERNAL);
      IREE_TRACE_ZONE_END(z_iteration);
      break;
    }

    // Check exit condition.
    iree_slim_mutex_lock(&watcher->mutex);
    bool do_exit = watcher->do_exit;
    iree_slim_mutex_unlock(&watcher->mutex);
    if (do_exit && watcher->active_count == 0) {
      IREE_TRACE_ZONE_END(z_iteration);
      break;
    }

    // Build wait set.
    iree_host_size_t wait_count = 0;
    VkSemaphoreWaitFlags wait_flags = 0;

    if (watcher->active_count == 0) {
      // No active entries — wait only on the poke semaphore.
      if (do_exit) {
        IREE_TRACE_ZONE_END(z_iteration);
        break;
      }
      watcher->wait_semaphores[0] = watcher->poke_semaphore;
      watcher->wait_values[0] = watcher->poke_last_value + 1;
      wait_count = 1;
      // No ANY flag needed for single semaphore.
    } else {
      // Active entries + poke semaphore.
      for (iree_host_size_t i = 0; i < watcher->active_count; ++i) {
        watcher->wait_semaphores[i] = watcher->active_entries[i].vk_semaphore;
        watcher->wait_values[i] = watcher->active_entries[i].target_value;
      }
      watcher->wait_semaphores[watcher->active_count] = watcher->poke_semaphore;
      watcher->wait_values[watcher->active_count] =
          watcher->poke_last_value + 1;
      wait_count = watcher->active_count + 1;
      wait_flags = VK_SEMAPHORE_WAIT_ANY_BIT;
    }

    VkSemaphoreWaitInfo wait_info;
    wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    wait_info.pNext = NULL;
    wait_info.flags = wait_flags;
    wait_info.semaphoreCount = (uint32_t)wait_count;
    wait_info.pSemaphores = watcher->wait_semaphores;
    wait_info.pValues = watcher->wait_values;

    {
      IREE_TRACE_ZONE_BEGIN_NAMED(z_wait,
                                  "iree_hal_vulkan_completion_watcher_wait");
      VkResult result = syms->vkWaitSemaphores(
          vk_device, &wait_info,
          IREE_HAL_VULKAN_COMPLETION_WATCHER_WAIT_TIMEOUT_NS);
      IREE_TRACE_ZONE_END(z_wait);

      if (result == VK_ERROR_DEVICE_LOST) {
        iree_hal_vulkan_completion_watcher_fail_all(watcher,
                                                    IREE_STATUS_UNAVAILABLE);
        IREE_TRACE_ZONE_END(z_iteration);
        break;
      }
      if (result == VK_TIMEOUT) {
        // Periodic liveness wake. If we're exiting and have active entries,
        // check if they've timed out (GPU may be hung).
        if (do_exit && watcher->active_count > 0) {
          // On shutdown, don't wait forever — fail remaining entries.
          iree_hal_vulkan_completion_watcher_fail_all(watcher,
                                                      IREE_STATUS_ABORTED);
          IREE_TRACE_ZONE_END(z_iteration);
          break;
        }
        IREE_TRACE_ZONE_END(z_iteration);
        continue;
      }
      // VK_SUCCESS — at least one semaphore made progress.
    }

    // Catch up poke value so next iteration waits at the right target.
    watcher->poke_last_value = (uint64_t)iree_atomic_load(
        &watcher->poke_next_value, iree_memory_order_acquire);

    // Scan active entries for completion in three steps:
    //   1. Observe every active entry's value.
    //   2. If any entry completed, free the resources of every submission that
    //      is complete by now (iree_hal_vulkan_completion_watcher_release_
    //      completed) before signaling anything.
    //   3. Sync host-side timelines and dispatch timepoints for the entries
    //      observed complete in step 1.
    //
    // The order matters: a waiter woken by a dispatch (the host or another
    // device's queue) may finish the program and tear down the VM context
    // right away. A retained command buffer keeps its bindings alive, and a
    // binding imported zero-copy from module rodata (hal.allocator.import)
    // keeps the rodata iree_vm_buffer_t referenced, so iree_vm_buffer_
    // deinitialize aborts when the module is destroyed. Step 2 frees not only
    // the signaled submission's resources but those of every completed
    // submission: a submission that the GPU finished earlier (and that the
    // signaled one may have waited on) can still be waiting in the active set
    // or in the pending list.
    //
    // NOTE(fork): upstream signals first and releases afterwards, here and in
    // the rewritten driver on main (queue.c iree_hal_vulkan_queue_drain_
    // completions: complete_submission before pending_submission_destroy).
    // Moving to that code drops this fix and brings the teardown abort back
    // unless it is re-applied.
    {
      IREE_TRACE_ZONE_BEGIN_NAMED(z_scan,
                                  "iree_hal_vulkan_completion_watcher_scan");
      // Step 1. The wait arrays are free scratch space until the next wait.
      bool any_completed = false;
      for (iree_host_size_t i = 0; i < watcher->active_count; ++i) {
        uint64_t value = 0;
        VkResult query_result = syms->vkGetSemaphoreCounterValue(
            vk_device, watcher->active_entries[i].vk_semaphore, &value);
        if (query_result != VK_SUCCESS) {
          // Query failure — treat as device loss.
          iree_hal_vulkan_completion_watcher_fail_all(watcher,
                                                      IREE_STATUS_UNAVAILABLE);
          IREE_TRACE_ZONE_END(z_scan);
          IREE_TRACE_ZONE_END(z_iteration);
          return 0;
        }
        watcher->wait_values[i] = value;
        if (value < IREE_HAL_SEMAPHORE_FAILURE_VALUE &&
            value >= watcher->active_entries[i].target_value) {
          any_completed = true;
        }
      }

      // Step 2.
      if (any_completed) {
        iree_hal_vulkan_completion_watcher_release_completed(watcher);
      }

      // Step 3.
      iree_host_size_t i = 0;
      while (i < watcher->active_count) {
        uint64_t value = watcher->wait_values[i];
        if (value >= IREE_HAL_SEMAPHORE_FAILURE_VALUE) {
          // Semaphore has been failed by another thread. The failing thread
          // already dispatched timepoints, so we just release the entry.
          iree_hal_vulkan_completion_entry_release(
              watcher, &watcher->active_entries[i], /*mutex_held=*/false);
          iree_hal_vulkan_completion_watcher_swap_remove(watcher, i);
          continue;
        }

        if (value >= watcher->active_entries[i].target_value) {
          // Completed; its resources were freed in step 2. Publish the value
          // only forward: a query or wait on another thread may already have
          // published it or a later one (see
          // iree_hal_vulkan_native_semaphore_observe_value).
          iree_hal_vulkan_native_semaphore_observe_value(
              (iree_async_semaphore_t*)watcher->active_entries[i].semaphore,
              value);
          iree_hal_vulkan_completion_entry_release(
              watcher, &watcher->active_entries[i], /*mutex_held=*/false);
          iree_hal_vulkan_completion_watcher_swap_remove(watcher, i);
          continue;
        }

        ++i;
      }
      IREE_TRACE_ZONE_END(z_scan);
    }

    IREE_TRACE_ZONE_END(z_iteration);
  }

  return 0;
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_vulkan_completion_watcher_create(
    VkDeviceHandle* logical_device, iree_allocator_t host_allocator,
    iree_hal_vulkan_completion_watcher_t** out_watcher) {
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(out_watcher);
  *out_watcher = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_vulkan_completion_watcher_t* watcher = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*watcher),
                                (void**)&watcher));
  memset(watcher, 0, sizeof(*watcher));
  iree_atomic_ref_count_init(&watcher->ref_count);
  watcher->host_allocator = host_allocator;
  watcher->logical_device = logical_device;
  iree_atomic_store(&watcher->poke_next_value, 0, iree_memory_order_release);
  watcher->poke_last_value = 0;
  iree_slim_mutex_initialize(&watcher->mutex);
  watcher->do_exit = false;

  // Create the poke semaphore (timeline, initial value 0).
  VkSemaphoreTypeCreateInfo timeline_create_info;
  timeline_create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
  timeline_create_info.pNext = NULL;
  timeline_create_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  timeline_create_info.initialValue = 0;
  VkSemaphoreCreateInfo create_info;
  create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  create_info.pNext = &timeline_create_info;
  create_info.flags = 0;
  iree_status_t status = VK_RESULT_TO_STATUS(
      logical_device->syms()->vkCreateSemaphore(*logical_device, &create_info,
                                                logical_device->allocator(),
                                                &watcher->poke_semaphore),
      "vkCreateSemaphore(poke)");

  // Allocate initial pending array.
  if (iree_status_is_ok(status)) {
    watcher->pending_capacity =
        IREE_HAL_VULKAN_COMPLETION_WATCHER_INITIAL_CAPACITY;
    status = iree_allocator_malloc(
        host_allocator,
        watcher->pending_capacity * sizeof(iree_hal_vulkan_completion_entry_t),
        (void**)&watcher->pending_entries);
  }

  // Allocate initial active + wait arrays.
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_completion_watcher_grow_active(
        watcher, IREE_HAL_VULKAN_COMPLETION_WATCHER_INITIAL_CAPACITY);
  }

  // Spawn the watcher thread.
  if (iree_status_is_ok(status)) {
    iree_thread_create_params_t params;
    memset(&params, 0, sizeof(params));
    params.name = iree_make_cstring_view("iree-hal-vulkan-watcher");
    params.priority_class = IREE_THREAD_PRIORITY_CLASS_NORMAL;
    status = iree_thread_create(
        (iree_thread_entry_t)iree_hal_vulkan_completion_watcher_thread_main,
        watcher, params, host_allocator, &watcher->thread);
  }

  if (iree_status_is_ok(status)) {
    *out_watcher = watcher;
  } else {
    // Cleanup on failure.
    if (watcher->poke_semaphore != VK_NULL_HANDLE) {
      logical_device->syms()->vkDestroySemaphore(*logical_device,
                                                 watcher->poke_semaphore,
                                                 logical_device->allocator());
    }
    iree_allocator_free(host_allocator, watcher->pending_entries);
    // active_entries is the base of the combined allocation that also contains
    // wait_semaphores and wait_values.
    iree_allocator_free(host_allocator, watcher->active_entries);
    iree_slim_mutex_deinitialize(&watcher->mutex);
    iree_allocator_free(host_allocator, watcher);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_vulkan_completion_watcher_destroy(
    iree_hal_vulkan_completion_watcher_t* watcher) {
  if (!watcher) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Signal the watcher to exit.
  iree_slim_mutex_lock(&watcher->mutex);
  watcher->do_exit = true;
  iree_slim_mutex_unlock(&watcher->mutex);

  // Poke the watcher to wake it from vkWaitSemaphores.
  int64_t poke_value = iree_atomic_fetch_add(&watcher->poke_next_value, 1,
                                             iree_memory_order_acq_rel) +
                       1;
  VkSemaphoreSignalInfo signal_info;
  signal_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
  signal_info.pNext = NULL;
  signal_info.semaphore = watcher->poke_semaphore;
  signal_info.value = (uint64_t)poke_value;
  watcher->logical_device->syms()->vkSignalSemaphore(*watcher->logical_device,
                                                     &signal_info);

  // Release joins the thread (ref count drops to 0 → internal join).
  iree_thread_release(watcher->thread);

  // Destroy the poke semaphore.
  watcher->logical_device->syms()->vkDestroySemaphore(
      *watcher->logical_device, watcher->poke_semaphore,
      watcher->logical_device->allocator());

  // Free the resources of any submission still in flight. Normally there are
  // none: the thread exits only after every active entry was processed.
  iree_slim_mutex_lock(&watcher->mutex);
  iree_hal_vulkan_completion_watcher_retire_all_locked(watcher);
  iree_slim_mutex_unlock(&watcher->mutex);

  // Free arrays. active_entries is the base of the combined allocation that
  // also contains wait_semaphores and wait_values.
  iree_allocator_free(watcher->host_allocator, watcher->pending_entries);
  watcher->pending_entries = NULL;
  watcher->pending_count = 0;
  iree_allocator_free(watcher->host_allocator, watcher->active_entries);
  watcher->active_entries = NULL;
  watcher->active_count = 0;

  // Drop the device's reference. Semaphores still alive keep the (now empty)
  // watcher until they are destroyed.
  iree_hal_vulkan_completion_watcher_release(watcher);

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_vulkan_completion_watcher_retain(
    iree_hal_vulkan_completion_watcher_t* watcher) {
  if (watcher) iree_atomic_ref_count_inc(&watcher->ref_count);
}

void iree_hal_vulkan_completion_watcher_release(
    iree_hal_vulkan_completion_watcher_t* watcher) {
  if (!watcher || iree_atomic_ref_count_dec(&watcher->ref_count) != 1) return;
  iree_slim_mutex_deinitialize(&watcher->mutex);
  iree_allocator_t host_allocator = watcher->host_allocator;
  iree_allocator_free(host_allocator, watcher);
}

void iree_hal_vulkan_completion_watcher_release_completed(
    iree_hal_vulkan_completion_watcher_t* watcher) {
  if (!watcher) return;
  // Re-entered from a release path on this thread: the outer call is freeing
  // under the mutex and finishes before its caller proceeds.
  if (iree_hal_vulkan_completion_watcher_releasing == watcher) return;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_slim_mutex_lock(&watcher->mutex);
  iree_hal_vulkan_completion_batch_t* batch = watcher->in_flight_head;
  // Nothing in flight (also after the device shut the watcher down): do not
  // touch the device.
  auto* syms = batch ? watcher->logical_device->syms().get() : nullptr;
  VkDevice vk_device = batch ? (VkDevice)*watcher->logical_device : VK_NULL_HANDLE;
  while (batch) {
    iree_hal_vulkan_completion_batch_t* next_batch = batch->next;
    uint64_t value = 0;
    VkResult result =
        syms->vkGetSemaphoreCounterValue(vk_device, batch->vk_semaphore, &value);
    // A failed semaphore does not mean the GPU work finished; leave those to
    // the failure paths.
    if (result == VK_SUCCESS && value < IREE_HAL_SEMAPHORE_FAILURE_VALUE &&
        value >= batch->target_value) {
      iree_hal_vulkan_completion_batch_retire_locked(watcher, batch);
    }
    batch = next_batch;
  }
  iree_slim_mutex_unlock(&watcher->mutex);
  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_vulkan_completion_watcher_register(
    iree_hal_vulkan_completion_watcher_t* watcher,
    const iree_hal_semaphore_list_t* signal_semaphores,
    iree_hal_resource_set_t* resource_set) {
  IREE_ASSERT_ARGUMENT(watcher);
  IREE_ASSERT_ARGUMENT(signal_semaphores);
  if (signal_semaphores->count == 0) {
    // Nothing to watch — free the resource set immediately.
    iree_hal_resource_set_free(resource_set);
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  // All entries of this submission share one batch that holds its resources,
  // so whichever of its semaphores is observed complete first frees them
  // before any of them is signaled.
  iree_hal_vulkan_completion_batch_t* batch = NULL;
  iree_status_t batch_status = iree_allocator_malloc(
      watcher->host_allocator, sizeof(*batch), (void**)&batch);
  if (!iree_status_is_ok(batch_status)) {
    iree_hal_resource_set_free(resource_set);
    IREE_TRACE_ZONE_END(z0);
    return batch_status;
  }
  batch->resource_set = resource_set;
  batch->vk_semaphore =
      iree_hal_vulkan_native_semaphore_handle(signal_semaphores->semaphores[0]);
  batch->target_value = signal_semaphores->payload_values[0];
  batch->prev = NULL;
  batch->next = NULL;
  batch->in_flight = false;
  batch->entry_count = signal_semaphores->count;

  iree_slim_mutex_lock(&watcher->mutex);

  // Check for failure/exit before accepting new work.
  if (watcher->do_exit) {
    iree_slim_mutex_unlock(&watcher->mutex);
    iree_allocator_free(watcher->host_allocator, batch);
    iree_hal_resource_set_free(resource_set);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_ABORTED,
                            "completion watcher is shutting down");
  }

  // Grow pending array if needed.
  iree_host_size_t required = watcher->pending_count + signal_semaphores->count;
  if (required > watcher->pending_capacity) {
    iree_host_size_t new_capacity =
        iree_max(watcher->pending_capacity * 2, required);
    iree_hal_vulkan_completion_entry_t* new_pending = NULL;
    iree_status_t status = iree_allocator_malloc(
        watcher->host_allocator,
        new_capacity * sizeof(iree_hal_vulkan_completion_entry_t),
        (void**)&new_pending);
    if (!iree_status_is_ok(status)) {
      iree_slim_mutex_unlock(&watcher->mutex);
      iree_allocator_free(watcher->host_allocator, batch);
      iree_hal_resource_set_free(resource_set);
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
    if (watcher->pending_count > 0) {
      memcpy(
          new_pending, watcher->pending_entries,
          watcher->pending_count * sizeof(iree_hal_vulkan_completion_entry_t));
    }
    iree_allocator_free(watcher->host_allocator, watcher->pending_entries);
    watcher->pending_entries = new_pending;
    watcher->pending_capacity = new_capacity;
  }

  // Append entries. They all share the batch holding the resource set.
  for (iree_host_size_t i = 0; i < signal_semaphores->count; ++i) {
    iree_hal_vulkan_completion_entry_t* entry =
        &watcher->pending_entries[watcher->pending_count++];
    entry->semaphore = signal_semaphores->semaphores[i];
    iree_hal_semaphore_retain(entry->semaphore);
    entry->vk_semaphore =
        iree_hal_vulkan_native_semaphore_handle(entry->semaphore);
    entry->target_value = signal_semaphores->payload_values[i];
    entry->batch = batch;
  }

  // Track the submission's resources until it is observed complete.
  if (resource_set) {
    batch->next = watcher->in_flight_head;
    if (batch->next) batch->next->prev = batch;
    watcher->in_flight_head = batch;
    batch->in_flight = true;
  }

  iree_slim_mutex_unlock(&watcher->mutex);

  // Poke the watcher to wake it from vkWaitSemaphores. The atomic increment
  // and vkSignalSemaphore are both thread-safe without the mutex.
  int64_t poke_value = iree_atomic_fetch_add(&watcher->poke_next_value, 1,
                                             iree_memory_order_acq_rel) +
                       1;
  VkSemaphoreSignalInfo signal_info;
  signal_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
  signal_info.pNext = NULL;
  signal_info.semaphore = watcher->poke_semaphore;
  signal_info.value = (uint64_t)poke_value;
  iree_status_t status =
      VK_RESULT_TO_STATUS(watcher->logical_device->syms()->vkSignalSemaphore(
                              *watcher->logical_device, &signal_info),
                          "vkSignalSemaphore(poke)");

  IREE_TRACE_ZONE_END(z0);
  return status;
}
