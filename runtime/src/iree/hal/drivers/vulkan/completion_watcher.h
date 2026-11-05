// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_COMPLETION_WATCHER_H_
#define IREE_HAL_DRIVERS_VULKAN_COMPLETION_WATCHER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/handle_util.h"
#include "iree/hal/utils/resource_set.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_vulkan_completion_watcher_t
    iree_hal_vulkan_completion_watcher_t;

// Creates a completion watcher thread that monitors outstanding
// VkTimelineSemaphore signals from GPU queue submissions. When any semaphore
// makes progress the watcher synchronizes the host-side async semaphore
// timelines, dispatches waiting timepoints, and frees retained resources.
//
// |logical_device| is borrowed — the device must outlive the watcher.
iree_status_t iree_hal_vulkan_completion_watcher_create(
    iree::hal::vulkan::VkDeviceHandle* logical_device,
    iree_allocator_t host_allocator,
    iree_hal_vulkan_completion_watcher_t** out_watcher);

// Shuts down the watcher: signals exit, joins the thread, and frees any
// remaining resource sets. All command queues should be drained before calling
// this so that outstanding GPU work has completed.
void iree_hal_vulkan_completion_watcher_destroy(
    iree_hal_vulkan_completion_watcher_t* watcher);

// Registers signal semaphores from a queue submission for completion tracking.
// The watcher will monitor each semaphore in |signal_semaphores| and, upon GPU
// completion, synchronize the host-side async timeline and dispatch timepoints.
//
// Ownership of |resource_set| transfers to the watcher, also when registration
// fails. It is freed by the first iree_hal_vulkan_completion_watcher_release_
// completed call that observes the submission complete, which the watcher makes
// before it signals any semaphore, so a waiter that wakes up never sees the
// submission's resources still retained. If |resource_set| is NULL, no
// resources are freed on completion.
//
// Thread-safe. Non-blocking. Wakes the watcher thread to include the new
// semaphores in its next wait set.
iree_status_t iree_hal_vulkan_completion_watcher_register(
    iree_hal_vulkan_completion_watcher_t* watcher,
    const iree_hal_semaphore_list_t* signal_semaphores,
    iree_hal_resource_set_t* resource_set);

// Frees the resources of every registered submission that has completed on the
// GPU. Rule: when a completion becomes visible (a semaphore is signaled or a
// waiter is woken), the resources of every submission completed by then are
// already released. The watcher calls this before it signals; a caller that
// learns of a completion without going through the watcher (for example a host
// fence wait before signaling other devices' semaphores) calls it before it
// makes that completion visible.
//
// Thread-safe. Does not wait on the GPU or on the watcher thread: it takes the
// watcher mutex, queries each in-flight submission once, and frees the
// completed ones under the mutex, so on return every submission that was
// complete when the call began has been released. A call re-entered from a
// release path on the same thread returns immediately. |watcher| may be NULL.
void iree_hal_vulkan_completion_watcher_release_completed(
    iree_hal_vulkan_completion_watcher_t* watcher);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_COMPLETION_WATCHER_H_
