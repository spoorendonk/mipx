#pragma once

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <mutex>

namespace mipx {

/// Thread-safe logger using fprintf + fflush behind a mutex.
/// All output goes through this to prevent interleaved lines from
/// multiple threads.
class Logger {
public:
    explicit Logger(std::FILE* out = stdout) : out_(out) {}

    /// Printf-style logging.  Acquires the mutex, formats, writes, flushes.
    void log(const char* fmt, ...) __attribute__((format(printf, 2, 3))) {
        std::va_list args;
        va_start(args, fmt);
        {
            std::lock_guard<std::mutex> lock(mtx_);
            std::vfprintf(out_, fmt, args);
            std::fflush(out_);
        }
        va_end(args);
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
    std::FILE* out_;
    std::mutex mtx_;
};

}  // namespace mipx
