// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_NATIVE_SEMAPHORE_H_
#define IREE_HAL_DRIVERS_VULKAN_NATIVE_SEMAPHORE_H_

#include <stdint.h>

#include "iree/async/semaphore.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/handle_util.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_vulkan_completion_watcher_t
    iree_hal_vulkan_completion_watcher_t;

// Creates a timeline semaphore implemented using the native VkSemaphore type.
// |proactor| is borrowed from the device's proactor pool and must outlive the
// semaphore. |completion_watcher| (may be NULL) is retained: when a host query
// or wait observes a value the watcher has not signaled yet, the semaphore
// first frees the resources of every completed submission through it, so a
// completion seen this way follows the same rule as the watcher's signals.
iree_status_t iree_hal_vulkan_native_semaphore_create(
    iree::hal::vulkan::VkDeviceHandle* logical_device,
    iree_async_proactor_t* proactor,
    iree_hal_vulkan_completion_watcher_t* completion_watcher,
    uint64_t initial_value, iree_hal_semaphore_t** out_semaphore);

// Returns true if |semaphore| is a Vulkan native semaphore.
bool iree_hal_vulkan_native_semaphore_isa(iree_hal_semaphore_t* semaphore);

// Returns the Vulkan timeline semaphore handle.
VkSemaphore iree_hal_vulkan_native_semaphore_handle(
    iree_hal_semaphore_t* semaphore);

// Raises the host-side timeline of the Vulkan semaphore |semaphore| to |value|,
// a value read from Vulkan, if it is higher, and dispatches the timepoints it
// satisfies. A lower or equal |value| (a stale observation, or one another
// thread already published) changes nothing. |value| must not be a failure
// value. Returns the host-side value after the call.
uint64_t iree_hal_vulkan_native_semaphore_observe_value(
    iree_async_semaphore_t* semaphore, uint64_t value);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_NATIVE_SEMAPHORE_H_
