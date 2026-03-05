#include "mipx/io.h"

#include <algorithm>
#include <cassert>
#include <charconv>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

#include <zlib.h>

#ifdef MIPX_HAS_BZIP2
#include <bzlib.h>
#endif

#ifdef __unix__
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace mipx {

namespace {

// Compressed files below this threshold are bulk-decompressed into memory
// so that getline() returns zero-copy string_view into the buffer.
constexpr size_t kBulkDecompressThreshold = 128 * 1024 * 1024;  // 128 MB

/// Unified line reader: mmap (plain), bulk-decompress (small .gz),
/// or buffered gzread (large .gz).  getline() returns string_view
/// valid until the next call.
class LineReader {
public:
    explicit LineReader(const std::string& filename) {
        bool is_gz = filename.size() >= 3 &&
                     filename.compare(filename.size() - 3, 3, ".gz") == 0;
        bool is_bz2 = filename.size() >= 4 &&
                      filename.compare(filename.size() - 4, 4, ".bz2") == 0;

        compressed_size_ = queryFileSize(filename);

        if (is_bz2) {
#ifdef MIPX_HAS_BZIP2
            initBz2(filename);
#else
            throw std::runtime_error(
                "bzip2 support not compiled in (need -DMIPX_USE_BZIP2=ON): " +
                filename);
#endif
        } else if (is_gz) {
            if (compressed_size_ <= kBulkDecompressThreshold) {
                initGzBulk(filename);
            } else {
                initGzStreaming(filename);
            }
        } else {
            initPlain(filename);
        }
    }

    ~LineReader() {
#ifdef __unix__
        if (mmap_addr_) {
            munmap(const_cast<char*>(mmap_addr_), mmap_len_);
            ::close(fd_);
        }
#endif
        if (gz_file_) gzclose(gz_file_);
#ifdef MIPX_HAS_BZIP2
        if (bz_handle_) {
            int bzerr;
            BZ2_bzReadClose(&bzerr, bz_handle_);
        }
        if (bz_fp_) fclose(bz_fp_);
#endif
    }

    LineReader(const LineReader&) = delete;
    LineReader& operator=(const LineReader&) = delete;

    /// Returns next line (valid until next call).  Returns false at EOF.
    bool getline(std::string_view& out) {
        if (data_) return getlineMemory(out);
#ifdef MIPX_HAS_BZIP2
        if (bz_handle_) return getlineBz2(out);
#endif
        return getlineBuffered(out);
    }

    [[nodiscard]] size_t compressedSize() const { return compressed_size_; }
    [[nodiscard]] bool isCompressed() const { return is_compressed_; }

private:
    static size_t queryFileSize(const std::string& filename) {
#ifdef __unix__
        struct stat st;
        if (::stat(filename.c_str(), &st) == 0)
            return static_cast<size_t>(st.st_size);
#endif
        std::ifstream f(filename, std::ios::binary | std::ios::ate);
        if (f) return static_cast<size_t>(f.tellg());
        return 0;
    }

    void initPlain(const std::string& filename) {
#ifdef __unix__
        fd_ = ::open(filename.c_str(), O_RDONLY);
        if (fd_ < 0)
            throw std::runtime_error("Cannot open file: " + filename);

        struct stat st;
        if (fstat(fd_, &st) < 0) {
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("Cannot stat file: " + filename);
        }
        auto len = static_cast<size_t>(st.st_size);
        if (len == 0) {
            ::close(fd_);
            fd_ = -1;
            data_ = "";
            data_len_ = 0;
            return;
        }
        void* addr = mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (addr == MAP_FAILED) {
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("mmap failed: " + filename);
        }
        madvise(addr, len, MADV_SEQUENTIAL);
        mmap_addr_ = static_cast<const char*>(addr);
        mmap_len_ = len;
        data_ = mmap_addr_;
        data_len_ = len;
#else
        std::ifstream f(filename, std::ios::binary);
        if (!f.is_open())
            throw std::runtime_error("Cannot open file: " + filename);
        f.seekg(0, std::ios::end);
        auto len = static_cast<size_t>(f.tellg());
        f.seekg(0, std::ios::beg);
        owned_buf_.resize(len);
        f.read(owned_buf_.data(), static_cast<std::streamsize>(len));
        data_ = owned_buf_.data();
        data_len_ = len;
#endif
    }

    void initGzBulk(const std::string& filename) {
        is_compressed_ = true;
        gzFile f = gzopen(filename.c_str(), "rb");
        if (!f) throw std::runtime_error("Cannot open file: " + filename);

        // Pre-allocate with heuristic estimate (MPS compresses ~12x).
        size_t est = std::max<size_t>(compressed_size_ * 14, 4096);
        owned_buf_.resize(est);
        size_t total = 0;

        for (;;) {
            size_t avail = owned_buf_.size() - total;
            if (avail == 0) {
                owned_buf_.resize(owned_buf_.size() * 2);
                avail = owned_buf_.size() - total;
            }
            auto to_read =
                static_cast<unsigned>(std::min<size_t>(avail, 1u << 30));
            int n = gzread(f, owned_buf_.data() + total, to_read);
            if (n <= 0) break;
            total += static_cast<size_t>(n);
        }
        gzclose(f);

        owned_buf_.resize(total);
        data_ = owned_buf_.data();
        data_len_ = total;
    }

    void initGzStreaming(const std::string& filename) {
        is_compressed_ = true;
        gz_file_ = gzopen(filename.c_str(), "rb");
        if (!gz_file_)
            throw std::runtime_error("Cannot open file: " + filename);
        gzbuffer(gz_file_, 1 << 17);  // 128 KB zlib internal buffer
    }

#ifdef MIPX_HAS_BZIP2
    void initBz2(const std::string& filename) {
        is_compressed_ = true;
        bz_fp_ = fopen(filename.c_str(), "rb");
        if (!bz_fp_)
            throw std::runtime_error("Cannot open file: " + filename);
        int bzerr = BZ_OK;
        bz_handle_ = BZ2_bzReadOpen(&bzerr, bz_fp_, 0, 0, nullptr, 0);
        if (bzerr != BZ_OK) {
            fclose(bz_fp_);
            bz_fp_ = nullptr;
            throw std::runtime_error("BZ2_bzReadOpen failed: " + filename);
        }
    }

    bool getlineBz2(std::string_view& out) {
        overflow_.clear();

        for (;;) {
            if (buf_pos_ < buf_len_) {
                const char* start = buf_ + buf_pos_;
                auto remaining =
                    static_cast<size_t>(buf_len_ - buf_pos_);
                const char* nl = static_cast<const char*>(
                    std::memchr(start, '\n', remaining));

                if (nl) {
                    auto len = static_cast<size_t>(nl - start);
                    buf_pos_ += static_cast<int>(len) + 1;

                    if (overflow_.empty()) {
                        out = std::string_view(start, len);
                    } else {
                        overflow_.append(start, len);
                        out = overflow_;
                    }
                    if (!out.empty() && out.back() == '\r')
                        out.remove_suffix(1);
                    return true;
                }

                overflow_.append(start, remaining);
                buf_pos_ = buf_len_;
            }

            if (bz_eof_) {
                if (!overflow_.empty()) {
                    out = overflow_;
                    if (!out.empty() && out.back() == '\r')
                        out.remove_suffix(1);
                    return true;
                }
                return false;
            }

            int bzerr = BZ_OK;
            buf_len_ = BZ2_bzRead(&bzerr, bz_handle_, buf_, sizeof(buf_));
            buf_pos_ = 0;
            if (bzerr == BZ_STREAM_END || buf_len_ == 0) {
                bz_eof_ = true;
            } else if (bzerr != BZ_OK) {
                bz_eof_ = true;
            }
        }
    }
#endif

    bool getlineMemory(std::string_view& out) {
        if (data_pos_ >= data_len_) return false;

        const char* start = data_ + data_pos_;
        const char* end = data_ + data_len_;
        const char* nl = static_cast<const char*>(
            std::memchr(start, '\n', static_cast<size_t>(end - start)));

        if (nl) {
            out = std::string_view(start, static_cast<size_t>(nl - start));
            data_pos_ = static_cast<size_t>(nl - data_) + 1;
        } else {
            out = std::string_view(start, static_cast<size_t>(end - start));
            data_pos_ = data_len_;
        }
        if (!out.empty() && out.back() == '\r') out.remove_suffix(1);
        return true;
    }

    bool getlineBuffered(std::string_view& out) {
        overflow_.clear();

        for (;;) {
            if (buf_pos_ < buf_len_) {
                const char* start = buf_ + buf_pos_;
                auto remaining =
                    static_cast<size_t>(buf_len_ - buf_pos_);
                const char* nl = static_cast<const char*>(
                    std::memchr(start, '\n', remaining));

                if (nl) {
                    auto len = static_cast<size_t>(nl - start);
                    buf_pos_ += static_cast<int>(len) + 1;

                    if (overflow_.empty()) {
                        out = std::string_view(start, len);
                    } else {
                        overflow_.append(start, len);
                        out = overflow_;
                    }
                    if (!out.empty() && out.back() == '\r')
                        out.remove_suffix(1);
                    return true;
                }

                // No newline — save tail and refill.
                overflow_.append(start, remaining);
                buf_pos_ = buf_len_;
            }

            int n = gzread(gz_file_, buf_, sizeof(buf_));
            if (n <= 0) {
                if (!overflow_.empty()) {
                    out = overflow_;
                    if (!out.empty() && out.back() == '\r')
                        out.remove_suffix(1);
                    return true;
                }
                return false;
            }
            buf_pos_ = 0;
            buf_len_ = n;
        }
    }

    // Memory-backed mode (mmap or bulk-decompressed).
    const char* data_ = nullptr;
    size_t data_len_ = 0;
    size_t data_pos_ = 0;

    // Owns decompressed / file data when not using mmap.
    std::vector<char> owned_buf_;

#ifdef __unix__
    const char* mmap_addr_ = nullptr;
    size_t mmap_len_ = 0;
    int fd_ = -1;
#endif

    // Streaming gz mode.
    gzFile gz_file_ = nullptr;
    char buf_[65536]{};
    int buf_pos_ = 0;
    int buf_len_ = 0;
    std::string overflow_;

    size_t compressed_size_ = 0;
    bool is_compressed_ = false;

#ifdef MIPX_HAS_BZIP2
    FILE* bz_fp_ = nullptr;
    BZFILE* bz_handle_ = nullptr;
    bool bz_eof_ = false;
#endif
};

/// Fixed-capacity token array (zero allocation).
struct Tokens {
    std::string_view data[6];
    int n = 0;

    [[nodiscard]] int size() const { return n; }
    [[nodiscard]] std::string_view operator[](int i) const { return data[i]; }
};

Tokens tokenize(std::string_view line) {
    Tokens t;
    size_t i = 0;
    const size_t len = line.size();
    while (i < len && t.n < 6) {
        while (i < len && (line[i] == ' ' || line[i] == '\t')) ++i;
        if (i >= len) break;
        size_t start = i;
        while (i < len && line[i] != ' ' && line[i] != '\t') ++i;
        t.data[t.n++] = line.substr(start, i - start);
    }
    return t;
}

Real parseReal(std::string_view s) {
    const char* begin = s.data();
    const char* end = s.data() + s.size();
    // std::from_chars may not accept a leading '+' on some implementations.
    if (begin < end && *begin == '+') ++begin;
    Real val = 0.0;
    auto [ptr, ec] = std::from_chars(begin, end, val);
    if (ec != std::errc{} || ptr != end) {
        throw std::runtime_error("Invalid number: " + std::string(s));
    }
    return val;
}

bool isSection(std::string_view line) {
    if (line.empty()) return false;
    return !std::isspace(static_cast<unsigned char>(line[0]));
}

enum class Section { None, Name, Rows, Columns, Rhs, Ranges, Bounds, Endata };

Section parseSection(std::string_view line) {
    auto tokens = tokenize(line);
    if (tokens.n == 0) return Section::None;
    auto s = tokens[0];
    if (s == "NAME") return Section::Name;
    if (s == "ROWS") return Section::Rows;
    if (s == "LAZYCONS" || s == "USERCUTS") return Section::Rows;
    if (s == "COLUMNS") return Section::Columns;
    if (s == "RHS") return Section::Rhs;
    if (s == "RANGES") return Section::Ranges;
    if (s == "BOUNDS") return Section::Bounds;
    if (s == "ENDATA") return Section::Endata;
    return Section::None;
}

/// Transparent hash/equal for string_view lookups on string-keyed maps.
struct StringHash {
    using is_transparent = void;
    size_t operator()(std::string_view sv) const noexcept {
        return std::hash<std::string_view>{}(sv);
    }
};

struct StringEqual {
    using is_transparent = void;
    bool operator()(std::string_view a, std::string_view b) const noexcept {
        return a == b;
    }
};

using StringMap =
    std::unordered_map<std::string, Index, StringHash, StringEqual>;

}  // namespace

LpProblem readMps(const std::string& filename) {
    LineReader reader(filename);

    LpProblem prob;

    // Transparent hash maps for name → index.
    StringMap row_map;
    StringMap col_map;

    // Row data during parsing.
    std::vector<char> row_sense;  // 'N', 'L', 'G', 'E'

    // Entry arrays for direct CSR construction (replaces triplets).
    std::vector<Index> entry_rows;
    std::vector<Index> entry_cols;
    std::vector<Real> entry_vals;

    // RHS and range values (indexed by row).
    std::vector<Real> rhs_values;
    std::vector<Real> range_values;

    // Reserve based on file-size heuristic (#4).
    {
        size_t fsz = reader.compressedSize();
        if (fsz > 1024) {
            size_t text_est =
                reader.isCompressed() ? fsz * 12 : fsz;
            size_t est_nnz = text_est / 40;
            size_t est_cols = text_est / 100;
            size_t est_rows = text_est / 500;

            entry_rows.reserve(est_nnz);
            entry_cols.reserve(est_nnz);
            entry_vals.reserve(est_nnz);
            prob.obj.reserve(est_cols);
            prob.col_lower.reserve(est_cols);
            prob.col_upper.reserve(est_cols);
            prob.col_type.reserve(est_cols);
            prob.col_semi_lower.reserve(est_cols);
            prob.col_names.reserve(est_cols);
            prob.row_names.reserve(est_rows);
            row_sense.reserve(est_rows);
            row_map.reserve(est_rows);
            col_map.reserve(est_cols);
        }
    }

    bool in_integer_section = false;
    Section section = Section::None;

    // Column name cache: MPS groups columns contiguously, so the same
    // column name repeats on consecutive lines.  Cache the last lookup.
    std::string_view cached_col_name;
    Index cached_col_idx = -1;

    auto getOrCreateCol = [&](std::string_view name) -> Index {
        if (name == cached_col_name && cached_col_idx >= 0)
            return cached_col_idx;
        auto it = col_map.find(name);
        if (it != col_map.end()) {
            cached_col_name = it->first;
            cached_col_idx = it->second;
            return it->second;
        }
        Index idx = prob.num_cols++;
        auto [ins_it, _] = col_map.emplace(std::string(name), idx);
        cached_col_name = ins_it->first;
        cached_col_idx = idx;
        prob.col_names.emplace_back(name);
        prob.obj.push_back(0.0);
        prob.col_lower.push_back(0.0);
        prob.col_upper.push_back(kInf);
        prob.col_type.push_back(VarType::Continuous);
        prob.col_semi_lower.push_back(0.0);
        return idx;
    };

    std::string_view line;
    while (reader.getline(line)) {
        // Skip empty lines and full-line comments.
        if (line.empty()) continue;
        if (line[0] == '*' || line[0] == '$') continue;

        // Strip inline '$' comments: '$' preceded by whitespace.
        for (size_t pos = 1; pos < line.size(); ++pos) {
            if (line[pos] == '$' &&
                (line[pos - 1] == ' ' || line[pos - 1] == '\t')) {
                line = line.substr(0, pos);
                break;
            }
        }

        if (isSection(line)) {
            section = parseSection(line);
            if (section == Section::Name) {
                auto tokens = tokenize(line);
                if (tokens.n >= 2) {
                    prob.name = std::string(tokens[1]);
                }
            }
            if (section == Section::Endata) break;
            continue;
        }

        auto tokens = tokenize(line);
        if (tokens.n == 0) continue;

        switch (section) {
            case Section::Rows: {
                if (tokens.n < 2) break;
                char sense = tokens[0][0];
                auto name = tokens[1];
                if (sense == 'N') {
                    // Objective row.
                    row_map.emplace(std::string(name), -1);
                } else {
                    Index idx = prob.num_rows++;
                    row_map.emplace(std::string(name), idx);
                    prob.row_names.emplace_back(name);
                    row_sense.push_back(sense);
                }
                break;
            }

            case Section::Columns: {
                // Check for integer markers.
                if (tokens.n >= 3 && tokens[1] == "'MARKER'") {
                    if (tokens[2] == "'INTORG'") {
                        in_integer_section = true;
                    } else if (tokens[2] == "'INTEND'") {
                        in_integer_section = false;
                    }
                    break;
                }

                if (tokens.n < 3) break;
                auto col_name = tokens[0];
                Index col_idx = getOrCreateCol(col_name);

                if (in_integer_section) {
                    prob.col_type[col_idx] = VarType::Integer;
                }

                // Process pairs: (row_name, value).
                for (int i = 1; i + 1 < tokens.n; i += 2) {
                    auto row_name = tokens[i];
                    Real val = parseReal(tokens[i + 1]);

                    auto it = row_map.find(row_name);
                    if (it == row_map.end()) {
                        throw std::runtime_error(
                            "MPS: unknown row '" + std::string(row_name) +
                            "'");
                    }
                    if (it->second == -1) {
                        // Objective row.
                        prob.obj[col_idx] = val;
                    } else {
                        entry_rows.push_back(it->second);
                        entry_cols.push_back(col_idx);
                        entry_vals.push_back(val);
                    }
                }
                break;
            }

            case Section::Rhs: {
                // First token is RHS name (ignored), then pairs.
                if (tokens.n < 3) break;
                for (int i = 1; i + 1 < tokens.n; i += 2) {
                    auto row_name = tokens[i];
                    Real val = parseReal(tokens[i + 1]);
                    auto it = row_map.find(row_name);
                    if (it == row_map.end()) continue;
                    if (it->second == -1) {
                        // Objective offset (RHS of N row).
                        prob.obj_offset = -val;
                        continue;
                    }
                    Index idx = it->second;
                    if (idx >= static_cast<Index>(rhs_values.size())) {
                        rhs_values.resize(idx + 1, 0.0);
                    }
                    rhs_values[idx] = val;
                }
                break;
            }

            case Section::Ranges: {
                if (tokens.n < 3) break;
                for (int i = 1; i + 1 < tokens.n; i += 2) {
                    auto row_name = tokens[i];
                    Real val = parseReal(tokens[i + 1]);
                    auto it = row_map.find(row_name);
                    if (it == row_map.end() || it->second == -1) continue;
                    Index idx = it->second;
                    if (idx >= static_cast<Index>(range_values.size())) {
                        range_values.resize(idx + 1, 0.0);
                    }
                    range_values[idx] = val;
                }
                break;
            }

            case Section::Bounds: {
                if (tokens.n < 3) break;
                auto bound_type = tokens[0];
                // tokens[1] is bound name (ignored).
                auto col_name = tokens[2];
                Index col_idx = getOrCreateCol(col_name);
                const bool has_value = tokens.n >= 4;

                if (bound_type == "LO" && has_value) {
                    prob.col_lower[col_idx] = parseReal(tokens[3]);
                } else if (bound_type == "UP" && has_value) {
                    prob.col_upper[col_idx] = parseReal(tokens[3]);
                } else if (bound_type == "FX" && has_value) {
                    Real v = parseReal(tokens[3]);
                    prob.col_lower[col_idx] = v;
                    prob.col_upper[col_idx] = v;
                } else if (bound_type == "FR") {
                    prob.col_lower[col_idx] = -kInf;
                    prob.col_upper[col_idx] = kInf;
                } else if (bound_type == "MI") {
                    prob.col_lower[col_idx] = -kInf;
                } else if (bound_type == "PL") {
                    prob.col_upper[col_idx] = kInf;
                } else if (bound_type == "BV") {
                    prob.col_lower[col_idx] = 0.0;
                    prob.col_upper[col_idx] = 1.0;
                    prob.col_type[col_idx] = VarType::Binary;
                } else if (bound_type == "LI" && has_value) {
                    prob.col_lower[col_idx] = parseReal(tokens[3]);
                    prob.col_type[col_idx] = VarType::Integer;
                } else if (bound_type == "UI" && has_value) {
                    prob.col_upper[col_idx] = parseReal(tokens[3]);
                    prob.col_type[col_idx] = VarType::Integer;
                } else if (bound_type == "SC") {
                    const Real semi_lb =
                        has_value ? parseReal(tokens[3]) : 1.0;
                    prob.col_type[col_idx] = VarType::SemiContinuous;
                    prob.col_lower[col_idx] = 0.0;
                    prob.col_semi_lower[col_idx] =
                        std::max<Real>(0.0, semi_lb);
                } else if (bound_type == "SI") {
                    const Real semi_lb =
                        has_value ? parseReal(tokens[3]) : 1.0;
                    prob.col_type[col_idx] = VarType::SemiInteger;
                    prob.col_lower[col_idx] = 0.0;
                    prob.col_semi_lower[col_idx] =
                        std::max<Real>(0.0, semi_lb);
                }
                break;
            }

            default:
                break;
        }
    }

    // Build CSR via counting sort (#2 — replaces triplet sort).
    {
        const Index nnz = static_cast<Index>(entry_rows.size());
        std::vector<Index> row_starts(prob.num_rows + 1, 0);

        // Count entries per row.
        for (Index k = 0; k < nnz; ++k) {
            ++row_starts[entry_rows[k] + 1];
        }

        // Prefix sum.
        for (Index i = 0; i < prob.num_rows; ++i) {
            row_starts[i + 1] += row_starts[i];
        }

        // Scatter into CSR arrays.
        std::vector<Index> col_indices(nnz);
        std::vector<Real> values(nnz);
        {
            std::vector<Index> pos(row_starts.begin(), row_starts.end());
            for (Index k = 0; k < nnz; ++k) {
                Index p = pos[entry_rows[k]]++;
                col_indices[p] = entry_cols[k];
                values[p] = entry_vals[k];
            }
        }

        // Free entry arrays early.
        { std::vector<Index>{}.swap(entry_rows); }
        { std::vector<Index>{}.swap(entry_cols); }
        { std::vector<Real>{}.swap(entry_vals); }

        // Sort within each row by column index and sum duplicates.
        Index write = 0;
        for (Index i = 0; i < prob.num_rows; ++i) {
            const Index start = row_starts[i];
            const Index end = row_starts[i + 1];

            // Insertion sort (typically already in column order from MPS).
            for (Index j = start + 1; j < end; ++j) {
                Index c = col_indices[j];
                Real v = values[j];
                Index k = j;
                while (k > start && col_indices[k - 1] > c) {
                    col_indices[k] = col_indices[k - 1];
                    values[k] = values[k - 1];
                    --k;
                }
                col_indices[k] = c;
                values[k] = v;
            }

            // Compact: sum duplicates, skip zeros.
            Index new_start = write;
            for (Index j = start; j < end;) {
                Index c = col_indices[j];
                Real v = 0.0;
                while (j < end && col_indices[j] == c) {
                    v += values[j];
                    ++j;
                }
                if (v != 0.0) {
                    col_indices[write] = c;
                    values[write] = v;
                    ++write;
                }
            }
            row_starts[i] = new_start;
        }
        row_starts[prob.num_rows] = write;
        col_indices.resize(write);
        values.resize(write);

        prob.matrix = SparseMatrix(prob.num_rows, prob.num_cols,
                                   std::move(values), std::move(col_indices),
                                   std::move(row_starts));
    }

    // Convert row_sense + rhs + ranges to row_lower/row_upper.
    rhs_values.resize(prob.num_rows, 0.0);
    range_values.resize(prob.num_rows, 0.0);
    prob.row_lower.resize(prob.num_rows);
    prob.row_upper.resize(prob.num_rows);

    for (Index i = 0; i < prob.num_rows; ++i) {
        Real rhs = rhs_values[i];
        Real range = range_values[i];
        char sense = row_sense[i];

        switch (sense) {
            case 'L':
                prob.row_upper[i] = rhs;
                prob.row_lower[i] =
                    (range != 0.0) ? rhs - std::abs(range) : -kInf;
                break;
            case 'G':
                prob.row_lower[i] = rhs;
                prob.row_upper[i] =
                    (range != 0.0) ? rhs + std::abs(range) : kInf;
                break;
            case 'E':
                if (range > 0.0) {
                    prob.row_lower[i] = rhs;
                    prob.row_upper[i] = rhs + range;
                } else if (range < 0.0) {
                    prob.row_lower[i] = rhs + range;
                    prob.row_upper[i] = rhs;
                } else {
                    prob.row_lower[i] = rhs;
                    prob.row_upper[i] = rhs;
                }
                break;
            default:
                prob.row_lower[i] = -kInf;
                prob.row_upper[i] = kInf;
                break;
        }
    }

    return prob;
}

}  // namespace mipx
