#pragma once

#include "httplib.h"
#include <stdexcept>

// Inference executes on HTTP workers. A hardware-sized or expanding pool can
// spread successive host-KV allocations over many allocator arenas, retaining
// a document-sized high-water mark in each. Keep a fixed, lane-sized pool;
// two workers per lane leave room for control requests during generation.
inline int kvmem_configure_http_pool(httplib::Server & server, int parallel, int requested = -1) {
    if (parallel < 1 || parallel > 8) throw std::invalid_argument("HTTP pool requires 1..8 inference lanes");
    const int workers = requested > 0 ? requested : 2 * parallel;
    server.new_task_queue = [workers] {
        return new httplib::ThreadPool(static_cast<size_t>(workers), static_cast<size_t>(workers));
    };
    return workers;
}
