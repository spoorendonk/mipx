#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <span>
#include <vector>

#include "mipx/core.h"

namespace mipx {

// Dense-addressable work vector with sparse touched-entry tracking.
// This is intended for hot simplex scratch state where most iterations touch
// only a small subset of entries, but O(1) random access is still needed.
class SparseWorkVector {
public:
    SparseWorkVector() = default;
    explicit SparseWorkVector(Index size) { setup(size); }

    void setup(Index size) {
        assert(size >= 0);
        values_.assign(static_cast<std::size_t>(size), 0.0);
        touched_indices_.clear();
        touched_indices_.reserve(static_cast<std::size_t>(size));
        touched_epoch_.assign(static_cast<std::size_t>(size), 0U);
        current_epoch_ = 1U;
    }

    [[nodiscard]] Index size() const {
        return static_cast<Index>(values_.size());
    }

    void clear() {
        for (Index i : touched_indices_) {
            values_[static_cast<std::size_t>(i)] = 0.0;
        }
        touched_indices_.clear();
        ++current_epoch_;
        if (current_epoch_ == 0U) {
            std::fill(touched_epoch_.begin(), touched_epoch_.end(), 0U);
            current_epoch_ = 1U;
        }
    }

    void clearAll() {
        std::fill(values_.begin(), values_.end(), 0.0);
        touched_indices_.clear();
        std::fill(touched_epoch_.begin(), touched_epoch_.end(), 0U);
        current_epoch_ = 1U;
    }

    void set(Index i, Real value) {
        assert(i >= 0 && i < size());
        if (value == 0.0 && !isTouched(i)) {
            return;
        }
        touch(i);
        values_[static_cast<std::size_t>(i)] = value;
    }

    void add(Index i, Real delta) {
        assert(i >= 0 && i < size());
        if (delta == 0.0) {
            return;
        }
        touch(i);
        values_[static_cast<std::size_t>(i)] += delta;
    }

    [[nodiscard]] Real operator[](Index i) const {
        assert(i >= 0 && i < size());
        return values_[static_cast<std::size_t>(i)];
    }

    [[nodiscard]] Real* data() { return values_.data(); }
    [[nodiscard]] const Real* data() const { return values_.data(); }

    [[nodiscard]] std::span<Real> dense() { return values_; }
    [[nodiscard]] std::span<const Real> dense() const { return values_; }

    [[nodiscard]] std::span<const Index> touched() const {
        return touched_indices_;
    }

    [[nodiscard]] bool isTouched(Index i) const {
        assert(i >= 0 && i < size());
        return touched_epoch_[static_cast<std::size_t>(i)] == current_epoch_;
    }

private:
    void touch(Index i) {
        auto& epoch = touched_epoch_[static_cast<std::size_t>(i)];
        if (epoch == current_epoch_) {
            return;
        }
        epoch = current_epoch_;
        touched_indices_.push_back(i);
    }

    std::vector<Real> values_;
    std::vector<Index> touched_indices_;
    std::vector<uint32_t> touched_epoch_;
    uint32_t current_epoch_ = 1U;
};

}  // namespace mipx
