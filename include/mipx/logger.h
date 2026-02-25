#pragma once

#include <atomic>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <mutex>

#ifdef __linux__
#include <unistd.h>
#endif

namespace mipx {

/// Fast thread-safe logger.
///
/// Design principles:
///   - **Zero-cost disable**: enabled_ check is a relaxed atomic load (single
///     instruction, no fence).  When disabled, log() is a no-op — no formatting,
///     no allocation, no syscall.  This makes it safe to leave log calls in
///     library code with negligible overhead.
///   - **Format outside lock**: vsnprintf writes to a stack buffer before any
///     synchronisation, so formatting never blocks other threads.
///   - **Atomic write fast path** (Linux): for messages ≤ PIPE_BUF (4096 bytes)
///     a single write() syscall is used. POSIX guarantees this is atomic, so no
///     mutex is needed and concurrent threads never interleave output.
///   - **Mutex fallback**: messages > PIPE_BUF (rare) acquire a mutex around
///     fwrite + fflush to prevent interleaving.
class Logger {
public:
    explicit Logger(std::FILE* out = stdout) : out_(out) {
#ifdef __linux__
        fd_ = fileno(out);
#endif
    }

    /// Check if logging is enabled.  Relaxed load — essentially free.
    [[nodiscard]] bool enabled() const noexcept {
        return enabled_.load(std::memory_order_relaxed);
    }

    /// Enable or disable all output.
    void setEnabled(bool e) noexcept {
        enabled_.store(e, std::memory_order_relaxed);
    }

    /// Printf-style logging.
    /// Fast path: format on stack, then atomic write().
    void log(const char* fmt, ...) __attribute__((format(printf, 2, 3))) {
        if (!enabled_.load(std::memory_order_relaxed)) return;

        // Format into stack buffer — no lock, no allocation.
        char buf[4096];
        std::va_list args;
        va_start(args, fmt);
        int len = std::vsnprintf(buf, sizeof(buf), fmt, args);
        va_end(args);

        if (len <= 0) return;
        if (len >= static_cast<int>(sizeof(buf))) len = sizeof(buf) - 1;

        writeBuf(buf, static_cast<std::size_t>(len));
    }

    /// Format a count with SI suffixes: 1234 -> "1234", 12345 -> "12k", 1234567 -> "1.2M".
    static void formatCount(int64_t count, char* buf, std::size_t buf_size) {
        if (count < 0) count = -count;
        if (count >= 1'000'000) {
            std::snprintf(buf, buf_size, "%.1fM", static_cast<double>(count) / 1e6);
        } else if (count >= 10'000) {
            std::snprintf(buf, buf_size, "%lldk",
                          static_cast<long long>(count / 1000));
        } else {
            std::snprintf(buf, buf_size, "%lld", static_cast<long long>(count));
        }
    }

private:
    void writeBuf(const char* buf, std::size_t len) {
#ifdef __linux__
        // POSIX guarantees write() is atomic for len ≤ PIPE_BUF (4096 on Linux).
        // No mutex needed — the kernel serialises concurrent writes.
        if (len <= 4096) {
            auto ret = ::write(fd_, buf, len);
            (void)ret;
            return;
        }
#endif
        // Fallback: mutex-protected fwrite for oversized messages.
        std::lock_guard<std::mutex> lock(mtx_);
        std::fwrite(buf, 1, len, out_);
        std::fflush(out_);
    }

    std::FILE* out_;
    std::atomic<bool> enabled_{true};
    std::mutex mtx_;  // only used for oversized messages
#ifdef __linux__
    int fd_ = STDOUT_FILENO;
#endif
};

}  // namespace mipx
