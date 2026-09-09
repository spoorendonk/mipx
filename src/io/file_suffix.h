#pragma once

// Filename-suffix helpers shared by the readers.
//
// The set of compression suffixes lives here so that the reader that
// decompresses them and the dispatcher that looks past them to find the format
// extension cannot drift apart: `Model.MPS.GZ` has to mean the same thing to
// both, or a file resolves to a format nobody can then read.

#include <algorithm>
#include <array>
#include <cctype>
#include <string_view>

namespace mipx::io_detail {

/// Compression suffixes the MPS line reader decompresses transparently. The
/// format extension sits in front of them: `model.mps.gz` is MPS.
inline constexpr std::array<std::string_view, 2> kCompressionSuffixes = {".gz", ".bz2"};

/// Case-insensitive suffix test. Filenames reach us from case-insensitive
/// filesystems too, where `Model.LP` is an ordinary name.
inline bool endsWithIgnoreCase(std::string_view text, std::string_view suffix) {
    if (text.size() < suffix.size()) {
        return false;
    }
    const std::string_view tail = text.substr(text.size() - suffix.size());
    return std::ranges::equal(tail, suffix, [](char a, char b) {
        return std::tolower(static_cast<unsigned char>(a)) ==
               std::tolower(static_cast<unsigned char>(b));
    });
}

/// Drop one trailing compression suffix, if present.
inline std::string_view stripCompressionSuffix(std::string_view filename) {
    for (const auto suffix : kCompressionSuffixes) {
        if (endsWithIgnoreCase(filename, suffix)) {
            return filename.substr(0, filename.size() - suffix.size());
        }
    }
    return filename;
}

}  // namespace mipx::io_detail
