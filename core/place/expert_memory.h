/* The device boundary used while changing expert backing. Keeping mapping operations separate
 * from loan publication lets every rank finish (including translation setup) before any new
 * expert slot becomes usable. Backends/tests can report partial mapping failures explicitly. */
#pragma once
#include "device/device.h"
#include <cstddef>

namespace rad {
class ExpertMemoryOps {
public:
    virtual ~ExpertMemoryOps() = default;
    virtual size_t rank_count() const = 0;
    virtual VMem& memory(size_t rank) = 0;
    virtual int select(size_t rank) = 0;
    virtual int commit(size_t rank, int64_t offset, int64_t bytes) {
        return memory(rank).commit(offset, bytes);
    }
    virtual int settle() { return dev_settle_maps(); }
};
} // namespace rad
