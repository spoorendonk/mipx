#include "file_suffix.h"
#include "mipx/io.h"

#include <algorithm>
#include <cassert>
#include <charconv>
#include <cmath>
#include <cstring>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
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
    explicit LineReader(const std::string& filename) : filename_(filename) {
        // Case-insensitive, and shared with the format dispatcher, so that
        // `Model.MPS.GZ` is decompressed by the same rule that resolved it to
        // MPS in the first place.
        const bool is_gz = io_detail::endsWithIgnoreCase(filename, ".gz");
        const bool is_bz2 = io_detail::endsWithIgnoreCase(filename, ".bz2");

        compressed_size_ = queryFileSize(filename);

        if (is_bz2) {
#ifdef MIPX_HAS_BZIP2
            initBz2(filename);
#else
            throw std::runtime_error("bzip2 support not compiled in (need -DMIPX_USE_BZIP2=ON): " +
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
        if (gz_file_) {
            gzclose(gz_file_);
        }
#ifdef MIPX_HAS_BZIP2
        if (bz_handle_) {
            int bzerr;
            BZ2_bzReadClose(&bzerr, bz_handle_);
        }
        if (bz_fp_)
            fclose(bz_fp_);
#endif
    }

    LineReader(const LineReader&) = delete;
    LineReader& operator=(const LineReader&) = delete;

    /// Returns next line (valid until next call).  Returns false at EOF.
    bool getline(std::string_view& out) {
        if (data_) {
            return getlineMemory(out);
        }
#ifdef MIPX_HAS_BZIP2
        if (bz_handle_)
            return getlineBz2(out);
#endif
        return getlineBuffered(out);
    }

    [[nodiscard]] size_t compressedSize() const { return compressed_size_; }
    [[nodiscard]] bool isCompressed() const { return is_compressed_; }

private:
    static size_t queryFileSize(const std::string& filename) {
#ifdef __unix__
        struct stat st;
        if (::stat(filename.c_str(), &st) == 0) {
            return static_cast<size_t>(st.st_size);
        }
#endif
        std::ifstream f(filename, std::ios::binary | std::ios::ate);
        if (f) {
            return static_cast<size_t>(f.tellg());
        }
        return 0;
    }

    void initPlain(const std::string& filename) {
#ifdef __unix__
        fd_ = ::open(filename.c_str(), O_RDONLY);
        if (fd_ < 0) {
            throw std::runtime_error("Cannot open file: " + filename);
        }

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
        if (!f) {
            throw std::runtime_error("Cannot open file: " + filename);
        }

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
            auto to_read = static_cast<unsigned>(std::min<size_t>(avail, 1u << 30));
            int n = gzread(f, owned_buf_.data() + total, to_read);
            if (n <= 0) {
                // gzread reports both EOF and failure as a non-positive return,
                // so a truncated or corrupt stream is indistinguishable from a
                // complete one unless zlib is asked. Left unasked, a damaged
                // archive yields a partial model that solves and is reported as
                // optimal -- the same silent wrong answer as issue #198.
                const std::string error = gzErrorText(f, filename);
                if (!error.empty()) {
                    gzclose(f);
                    throw std::runtime_error("Cannot read '" + filename + "': " + error);
                }
                break;
            }
            total += static_cast<size_t>(n);
        }
        gzclose(f);

        owned_buf_.resize(total);
        data_ = owned_buf_.data();
        data_len_ = total;
    }

    /// Describe a zlib failure on `file`, or "" when the stream is merely at
    /// its end. Z_STREAM_END is a clean finish, not an error.
    ///
    /// zlib prefixes its message with the path it opened, which the caller is
    /// about to name as well, so that prefix is trimmed.
    static std::string gzErrorText(gzFile file, const std::string& filename) {
        int code = Z_OK;
        const char* message = gzerror(file, &code);
        if (code == Z_OK || code == Z_STREAM_END) {
            return {};
        }
        if (message == nullptr || *message == '\0') {
            return "corrupt or truncated gzip stream";
        }
        std::string text = message;
        const std::string prefix = filename + ": ";
        if (text.starts_with(prefix)) {
            text.erase(0, prefix.size());
        }
        return text;
    }

    void initGzStreaming(const std::string& filename) {
        is_compressed_ = true;
        gz_file_ = gzopen(filename.c_str(), "rb");
        if (!gz_file_) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
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
                auto remaining = static_cast<size_t>(buf_len_ - buf_pos_);
                const char* nl = static_cast<const char*>(std::memchr(start, '\n', remaining));

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
            // Checked before the EOF test: a truncated archive reports
            // BZ_UNEXPECTED_EOF with nothing read, which must not pass for a
            // clean end of stream (see the gzread note in the gz path).
            if (bzerr != BZ_OK && bzerr != BZ_STREAM_END) {
                throw std::runtime_error("Cannot read '" + filename_ +
                                         "': bzip2 decompression failed (BZ2_bzRead error " +
                                         std::to_string(bzerr) +
                                         "); the file is truncated or corrupt.");
            }
            if (bzerr == BZ_STREAM_END || buf_len_ == 0) {
                bz_eof_ = true;
            }
        }
    }
#endif

    bool getlineMemory(std::string_view& out) {
        if (data_pos_ >= data_len_) {
            return false;
        }

        const char* start = data_ + data_pos_;
        const char* end = data_ + data_len_;
        const char* nl =
            static_cast<const char*>(std::memchr(start, '\n', static_cast<size_t>(end - start)));

        if (nl) {
            out = std::string_view(start, static_cast<size_t>(nl - start));
            data_pos_ = static_cast<size_t>(nl - data_) + 1;
        } else {
            out = std::string_view(start, static_cast<size_t>(end - start));
            data_pos_ = data_len_;
        }
        if (!out.empty() && out.back() == '\r') {
            out.remove_suffix(1);
        }
        return true;
    }

    bool getlineBuffered(std::string_view& out) {
        overflow_.clear();

        for (;;) {
            if (buf_pos_ < buf_len_) {
                const char* start = buf_ + buf_pos_;
                auto remaining = static_cast<size_t>(buf_len_ - buf_pos_);
                const char* nl = static_cast<const char*>(std::memchr(start, '\n', remaining));

                if (nl) {
                    auto len = static_cast<size_t>(nl - start);
                    buf_pos_ += static_cast<int>(len) + 1;

                    if (overflow_.empty()) {
                        out = std::string_view(start, len);
                    } else {
                        overflow_.append(start, len);
                        out = overflow_;
                    }
                    if (!out.empty() && out.back() == '\r') {
                        out.remove_suffix(1);
                    }
                    return true;
                }

                // No newline — save tail and refill.
                overflow_.append(start, remaining);
                buf_pos_ = buf_len_;
            }

            int n = gzread(gz_file_, buf_, sizeof(buf_));
            if (n <= 0) {
                const std::string error = gzErrorText(gz_file_, filename_);
                if (!error.empty()) {
                    throw std::runtime_error("Cannot read '" + filename_ + "': " + error);
                }
                if (!overflow_.empty()) {
                    out = overflow_;
                    if (!out.empty() && out.back() == '\r') {
                        out.remove_suffix(1);
                    }
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
    std::string filename_;

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
        while (i < len && (line[i] == ' ' || line[i] == '\t')) {
            ++i;
        }
        if (i >= len) {
            break;
        }
        size_t start = i;
        while (i < len && line[i] != ' ' && line[i] != '\t') {
            ++i;
        }
        t.data[t.n++] = line.substr(start, i - start);
    }
    return t;
}

Real parseReal(std::string_view s) {
    const char* begin = s.data();
    const char* end = s.data() + s.size();
    // std::from_chars may not accept a leading '+' on some implementations.
    if (begin < end && *begin == '+') {
        ++begin;
    }
    Real val = 0.0;
    auto [ptr, ec] = std::from_chars(begin, end, val);
    if (ec != std::errc{} || ptr != end) {
        throw std::runtime_error("Invalid number: " + std::string(s));
    }
    return val;
}

// Unsupported: an extension section that changes the model and that this
// reader does not read (quadratic, piecewise-linear, general constraints);
// a file carrying one is refused rather than solved as a different model
// (#254). Skipped: a section whose data lines are ignored -- BRANCH, which
// carries branching priorities and leaves the model as it is, and any
// section of a file that has not shown itself to be MPS yet (see readMps).
enum class Section {
    None,
    Name,
    ObjSense,
    ObjName,
    Rows,
    Columns,
    Rhs,
    Ranges,
    Bounds,
    Sos,
    Indicators,
    Endata,
    Unsupported,
    Skipped
};

Section keywordSection(std::string_view s);

bool isObjSenseWord(std::string_view s) {
    return s == "MAX" || s == "MIN" || s == "MAXIMIZE" || s == "MINIMIZE";
}

bool isNumber(std::string_view s) {
    double v = 0.0;
    const auto [end, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
    return end == s.data() + s.size() && ec != std::errc::invalid_argument;
}

// The section a line opens, or nullopt for a data line. A header starts in
// column 1. Free MPS also lets data lines start there (mcf's LP recorder
// writes them so, as HiGHS reads them), so a column-1 line is a header only
// when it is a section keyword with at most one more token, not a number:
// "RHS", "OBJSENSE MAX", "OBJNAME obj". A data line has more tokens, or a
// value second (a row named BOUNDS in an RHS without a set name), so an RHS
// set named RHS or a column named BOUNDS stays data. NAME is a header only
// before any section (it may carry a name with spaces). "LAZYCONS 1" after
// ROWS is a header with its level, where no data line has a number second.
// A lone unknown word is an unknown section, which readMps refuses, except
// where a lone word is data: a sense under OBJSENSE, the row OBJNAME names,
// an SOS member without a weight. The caller has skipped empty lines, so
// tokens.n >= 1 here.
std::optional<Section> headerOf(std::string_view line, const Tokens& tokens, Section current) {
    if (std::isspace(static_cast<unsigned char>(line[0]))) {
        return std::nullopt;
    }
    const Section s = keywordSection(tokens[0]);
    if (s == Section::Name && current == Section::None) {
        return s;
    }
    if (s != Section::None && s != Section::Name &&
        (tokens.n == 1 || (tokens.n == 2 && !isNumber(tokens[1])))) {
        return s;
    }
    if (tokens.n == 2 && tokens[0] == "LAZYCONS" && current == Section::Rows) {
        return s;
    }
    // A section readMps refuses is a header whatever follows it: Gurobi's
    // "SCENARIOS 3" and MOSEK's "CSECTION k 3e-1 PPOW" carry numbers, and
    // read as data they would rewrite the section before them. Only a column
    // or set named after one of them is misread, and refused, which is safe.
    if (s == Section::Unsupported) {
        return s;
    }
    // OBJNAME belongs before ROWS, where no data line starts in column 1 with
    // it, so a row named with a number ("OBJNAME 2") is still its header.
    if (s == Section::ObjName && tokens.n == 2 &&
        (current == Section::None || current == Section::Name || current == Section::ObjSense)) {
        return s;
    }
    const bool lone_word_is_data = (current == Section::ObjSense && isObjSenseWord(tokens[0])) ||
                                   current == Section::ObjName || current == Section::Sos;
    if (s == Section::None && tokens.n == 1 && !lone_word_is_data) {
        return Section::None;
    }
    return std::nullopt;
}

Section keywordSection(std::string_view s) {
    if (s == "NAME") {
        return Section::Name;
    }
    // CPLEX's documentation spells the section both ways.
    if (s == "OBJSENSE" || s == "OBJSEN") {
        return Section::ObjSense;
    }
    if (s == "ROWS") {
        return Section::Rows;
    }
    if (s == "LAZYCONS" || s == "USERCUTS") {
        return Section::Rows;
    }
    if (s == "COLUMNS") {
        return Section::Columns;
    }
    if (s == "RHS") {
        return Section::Rhs;
    }
    if (s == "RANGES") {
        return Section::Ranges;
    }
    if (s == "BOUNDS") {
        return Section::Bounds;
    }
    if (s == "ENDATA") {
        return Section::Endata;
    }
    if (s == "OBJNAME") {
        return Section::ObjName;
    }
    if (s == "SOS") {
        return Section::Sos;
    }
    if (s == "INDICATORS") {
        return Section::Indicators;
    }
    if (s == "BRANCH") {
        return Section::Skipped;
    }
    // REFROW gives SOS weights from a row's coefficients, so reading the SOS
    // section without it would weigh the members wrongly.
    for (const std::string_view unsupported :
         {"QUADOBJ", "QSECTION", "QMATRIX", "QCMATRIX", "CSECTION", "PWLOBJ", "GENCONS", "REFROW",
          "NLOBJ", "NLCONS", "SCENARIOS"}) {
        if (s == unsupported) {
            return Section::Unsupported;
        }
    }
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
    bool operator()(std::string_view a, std::string_view b) const noexcept { return a == b; }
};

using StringMap = std::unordered_map<std::string, Index, StringHash, StringEqual>;

/// The objective sense an OBJSENSE value names: MAX/MAXIMIZE or MIN/MINIMIZE.
Sense parseObjSense(std::string_view value) {
    if (value == "MAX" || value == "MAXIMIZE") {
        return Sense::Maximize;
    }
    if (value == "MIN" || value == "MINIMIZE") {
        return Sense::Minimize;
    }
    throw std::runtime_error("MPS: unknown OBJSENSE '" + std::string(value) + "'");
}

/// Row-map entries for free (N) rows, which are not constraints. Only the
/// first N row is the objective; any later one is dropped with everything
/// written on it, the convention CPLEX and HiGHS follow.
constexpr Index kObjectiveRow = -1;
constexpr Index kDroppedFreeRow = -2;

using SosConstraint = LpProblem::SosConstraint;
using IndicatorConstraint = LpProblem::IndicatorConstraint;

Index existingColumn(const StringMap& col_map, std::string_view name, std::string_view section) {
    const auto it = col_map.find(name);
    if (it == col_map.end()) {
        throw std::runtime_error("MPS: " + std::string(section) + " names unknown column '" +
                                 std::string(name) + "'");
    }
    return it->second;
}

/// One line of the SOS section, in CPLEX's and Gurobi's layout: a set opens
/// with "S1 name" or "S2 name" (older writers put the word SOS before the
/// name and a priority after it, both ignored here), and each member line is
/// "column weight". A member without a weight takes its position in the set,
/// the order CPLEX gives members when no REFROW weighs them.
void readSosLine(const Tokens& tokens, const StringMap& col_map, std::vector<SosConstraint>& sets) {
    const bool opens_set =
        (tokens[0] == "S1" || tokens[0] == "S2") && (tokens.n == 1 || !isNumber(tokens[1]));
    if (opens_set) {
        SosConstraint& set = sets.emplace_back();
        set.type = tokens[0] == "S1" ? SosConstraint::Type::Sos1 : SosConstraint::Type::Sos2;
        if (tokens.n >= 3 && tokens[1] == "SOS" && !isNumber(tokens[2])) {
            set.name = std::string(tokens[2]);
        } else if (tokens.n >= 2) {
            set.name = std::string(tokens[1]);
        } else {
            set.name = "sos" + std::to_string(sets.size());
        }
        return;
    }
    if (sets.empty()) {
        throw std::runtime_error("MPS: SOS member '" + std::string(tokens[0]) +
                                 "' comes before any S1 or S2 line");
    }
    if (tokens.n > 2) {
        throw std::runtime_error("MPS: SOS member line for '" + std::string(tokens[0]) +
                                 "' has more than a column and a weight");
    }
    SosConstraint& set = sets.back();
    set.vars.push_back(existingColumn(col_map, tokens[0], "SOS"));
    set.weights.push_back(tokens.n == 2 ? parseReal(tokens[1])
                                        : static_cast<Real>(set.vars.size()));
}

/// An INDICATORS line, "IF row column value": the row holds when the binary
/// column takes the value. The row stays in the row data until the bounds
/// are known (extractIndicatorRows).
struct PendingIndicator {
    Index row = -1;
    Index col = -1;
    bool active_value = true;
};

PendingIndicator readIndicatorLine(const Tokens& tokens, const StringMap& row_map,
                                   const StringMap& col_map) {
    if (tokens.n != 4 || tokens[0] != "IF") {
        throw std::runtime_error("MPS: INDICATORS line must read 'IF row column value', not '" +
                                 std::string(tokens[0]) + " ...'");
    }
    const auto row = row_map.find(tokens[1]);
    if (row == row_map.end()) {
        throw std::runtime_error("MPS: INDICATORS names unknown row '" + std::string(tokens[1]) +
                                 "'");
    }
    if (row->second < 0) {
        throw std::runtime_error("MPS: INDICATORS names free (N) row '" + std::string(tokens[1]) +
                                 "'");
    }
    const Real value = parseReal(tokens[3]);
    if (value != 0.0 && value != 1.0) {
        throw std::runtime_error("MPS: INDICATORS value for row '" + std::string(tokens[1]) +
                                 "' must be 0 or 1, not '" + std::string(tokens[3]) + "'");
    }
    return {row->second, existingColumn(col_map, tokens[2], "INDICATORS"), value == 1.0};
}

/// Move each indicator's row out of the linear rows into an indicator
/// constraint carrying the row's coefficients and bounds. A ranged row cannot
/// be one (CPLEX's rule), and a row can carry only one indicator.
void extractIndicatorRows(LpProblem& prob, const std::vector<PendingIndicator>& pending,
                          const std::vector<Real>& range_values) {
    if (pending.empty()) {
        return;
    }
    std::vector<char> is_indicator(static_cast<std::size_t>(prob.num_rows), 0);
    for (const PendingIndicator& p : pending) {
        const std::string& row_name = prob.row_names[p.row];
        if (is_indicator[p.row] != 0) {
            throw std::runtime_error("MPS: INDICATORS names row '" + row_name + "' twice");
        }
        if (range_values[p.row] != 0.0) {
            throw std::runtime_error("MPS: INDICATORS row '" + row_name + "' is a ranged row");
        }
        is_indicator[p.row] = 1;
        const SparseVectorView row = prob.matrix.row(p.row);
        IndicatorConstraint& ic = prob.indicator_constraints.emplace_back();
        ic.binary_var = p.col;
        ic.active_value = p.active_value;
        ic.indices.assign(row.indices.begin(), row.indices.end());
        ic.values.assign(row.values.begin(), row.values.end());
        ic.lower = prob.row_lower[p.row];
        ic.upper = prob.row_upper[p.row];
        ic.name = row_name;
    }

    std::vector<Real> values;
    std::vector<Index> col_indices;
    std::vector<Index> row_starts{0};
    Index kept = 0;
    for (Index i = 0; i < prob.num_rows; ++i) {
        if (is_indicator[i] != 0) {
            continue;
        }
        const SparseVectorView row = prob.matrix.row(i);
        values.insert(values.end(), row.values.begin(), row.values.end());
        col_indices.insert(col_indices.end(), row.indices.begin(), row.indices.end());
        row_starts.push_back(static_cast<Index>(values.size()));
        if (kept != i) {
            prob.row_lower[kept] = prob.row_lower[i];
            prob.row_upper[kept] = prob.row_upper[i];
            prob.row_names[kept] = std::move(prob.row_names[i]);
        }
        ++kept;
    }
    prob.num_rows = kept;
    prob.row_lower.resize(kept);
    prob.row_upper.resize(kept);
    prob.row_names.resize(kept);
    prob.matrix = SparseMatrix(kept, prob.num_cols, std::move(values), std::move(col_indices),
                               std::move(row_starts));
}

}  // namespace

LpProblem readMps(const std::string& filename, ReadDiagnostics* diag) {
    if (diag != nullptr) {
        *diag = ReadDiagnostics{};
    }

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
            size_t text_est = reader.isCompressed() ? fsz * 12 : fsz;
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
    bool have_objective_row = false;
    bool saw_mps_section = false;
    std::string deferred_refusal;
    Section section = Section::None;

    // OBJNAME picks the objective among the N rows; without it the first is.
    // CPLEX puts it before ROWS, the only place it can still decide.
    std::string obj_name;
    auto setObjName = [&](std::string_view name) {
        if (!row_map.empty()) {
            throw std::runtime_error("MPS: OBJNAME '" + std::string(name) + "' comes after ROWS");
        }
        obj_name = std::string(name);
    };

    std::vector<PendingIndicator> pending_indicators;

    // Column name cache: MPS groups columns contiguously, so the same
    // column name repeats on consecutive lines.  Cache the last lookup.
    std::string_view cached_col_name;
    Index cached_col_idx = -1;

    auto getOrCreateCol = [&](std::string_view name) -> Index {
        if (name == cached_col_name && cached_col_idx >= 0) {
            return cached_col_idx;
        }
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
        if (line.empty()) {
            continue;
        }
        // Recorded before comments are skipped: this answers "was there
        // anything in the file", not "was there anything we understood".
        if (diag != nullptr) {
            diag->saw_content = true;
        }
        if (line[0] == '*' || line[0] == '$') {
            continue;
        }

        // Strip inline '$' comments: '$' preceded by whitespace.
        for (size_t pos = 1; pos < line.size(); ++pos) {
            if (line[pos] == '$' && (line[pos - 1] == ' ' || line[pos - 1] == '\t')) {
                line = line.substr(0, pos);
                break;
            }
        }

        auto tokens = tokenize(line);
        if (const auto header = headerOf(line, tokens, section)) {
            section = *header;
            // Only headers that are exclusive to MPS count as evidence that
            // this really is an MPS file. NAME carries no model data, and
            // BOUNDS is a section keyword in CPLEX LP too -- counting either
            // would let a wrong-format file pass as MPS.
            if (section == Section::Rows || section == Section::Columns ||
                section == Section::Rhs || section == Section::Ranges ||
                section == Section::Endata) {
                saw_mps_section = true;
                if (diag != nullptr) {
                    diag->saw_format_section = true;
                }
            }
            // An extension section this reader cannot read, or an unknown
            // one, is refused by name (#254). Before any MPS-exclusive header
            // the file may be another format altogether (CPLEX LP spells a
            // section SOS too), where readModel's format error says more, so
            // the refusal waits for the end and the section is skipped.
            const bool extension = section == Section::Unsupported || section == Section::None ||
                                   section == Section::Sos || section == Section::Indicators;
            if (extension) {
                const std::string word(tokens[0]);
                std::string refusal =
                    section == Section::Unsupported ? "MPS: section '" + word + "' is not supported"
                    : section == Section::None      ? "MPS: unknown section '" + word + "'"
                                               : "MPS: section '" + word + "' comes before ROWS";
                if (saw_mps_section && section != Section::Sos && section != Section::Indicators) {
                    throw std::runtime_error(refusal);
                }
                if (!saw_mps_section) {
                    if (deferred_refusal.empty()) {
                        deferred_refusal = std::move(refusal);
                    }
                    section = Section::Skipped;
                }
            }
            if (section == Section::Name && tokens.n >= 2) {
                prob.name = std::string(tokens[1]);
            }
            // Free MPS may put the sense on the section line itself, and the
            // objective's name on the OBJNAME line.
            if (section == Section::ObjSense && tokens.n >= 2) {
                prob.sense = parseObjSense(tokens[1]);
            }
            if (section == Section::ObjName && tokens.n >= 2) {
                setObjName(tokens[1]);
            }
            if (section == Section::Endata) {
                break;
            }
            continue;
        }

        if (tokens.n == 0) {
            continue;
        }

        switch (section) {
            case Section::ObjSense: {
                prob.sense = parseObjSense(tokens[0]);
                break;
            }

            case Section::ObjName: {
                setObjName(tokens[0]);
                break;
            }

            case Section::Rows: {
                if (tokens.n < 2) {
                    break;
                }
                char sense = tokens[0][0];
                auto name = tokens[1];
                if (sense == 'N') {
                    // Gurobi declares each of several objectives as an N row
                    // with priority, weight and tolerance fields; read as
                    // one, the model would lose all but the first.
                    if (tokens.n > 2) {
                        throw std::runtime_error("MPS: N row '" + std::string(name) +
                                                 "' carries multi-objective fields, which "
                                                 "are not supported");
                    }
                    const bool is_objective =
                        obj_name.empty() ? !have_objective_row : name == obj_name;
                    row_map.emplace(std::string(name),
                                    is_objective ? kObjectiveRow : kDroppedFreeRow);
                    have_objective_row = have_objective_row || is_objective;
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
                    } else {
                        // CPLEX's 'SOSORG'/'SOSEND' form of SOS, and any
                        // other marker, would otherwise be dropped (#254).
                        throw std::runtime_error("MPS: marker " + std::string(tokens[2]) +
                                                 " is not supported");
                    }
                    break;
                }
                // The SOS marker form may put S1 or S2 before the marker's
                // name, which moves 'MARKER' one field on.
                if (tokens.n >= 4 && tokens[2] == "'MARKER'") {
                    throw std::runtime_error("MPS: marker " + std::string(tokens[3]) +
                                             " is not supported");
                }

                if (tokens.n < 3) {
                    break;
                }
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
                        throw std::runtime_error("MPS: unknown row '" + std::string(row_name) +
                                                 "'");
                    }
                    if (it->second == kObjectiveRow) {
                        prob.obj[col_idx] = val;
                    } else if (it->second == kDroppedFreeRow) {
                        continue;
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
                if (tokens.n < 3) {
                    break;
                }
                for (int i = 1; i + 1 < tokens.n; i += 2) {
                    auto row_name = tokens[i];
                    Real val = parseReal(tokens[i + 1]);
                    auto it = row_map.find(row_name);
                    if (it == row_map.end()) {
                        continue;
                    }
                    if (it->second == kDroppedFreeRow) {
                        continue;
                    }
                    if (it->second == kObjectiveRow) {
                        // An RHS on the objective row is the negated objective
                        // constant: obj_offset = -RHS, so the model reads as
                        // c'x - RHS. Same convention as HiGHS and CPLEX, and the
                        // one tests/data/netlib.solu follows (e226's RHS of
                        // -7.113 gives -11.638929066, not -18.751929066; issue
                        // #207). mps_writer.cpp writes it back negated.
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
                if (tokens.n < 3) {
                    break;
                }
                for (int i = 1; i + 1 < tokens.n; i += 2) {
                    auto row_name = tokens[i];
                    Real val = parseReal(tokens[i + 1]);
                    auto it = row_map.find(row_name);
                    if (it == row_map.end() || it->second < 0) {
                        continue;
                    }
                    Index idx = it->second;
                    if (idx >= static_cast<Index>(range_values.size())) {
                        range_values.resize(idx + 1, 0.0);
                    }
                    range_values[idx] = val;
                }
                break;
            }

            case Section::Bounds: {
                if (tokens.n < 3) {
                    break;
                }
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
                    const Real semi_lb = has_value ? parseReal(tokens[3]) : 1.0;
                    prob.col_type[col_idx] = VarType::SemiContinuous;
                    prob.col_lower[col_idx] = 0.0;
                    prob.col_semi_lower[col_idx] = std::max<Real>(0.0, semi_lb);
                } else if (bound_type == "SI") {
                    const Real semi_lb = has_value ? parseReal(tokens[3]) : 1.0;
                    prob.col_type[col_idx] = VarType::SemiInteger;
                    prob.col_lower[col_idx] = 0.0;
                    prob.col_semi_lower[col_idx] = std::max<Real>(0.0, semi_lb);
                }
                break;
            }

            case Section::Sos: {
                readSosLine(tokens, col_map, prob.sos_constraints);
                break;
            }

            case Section::Indicators: {
                pending_indicators.push_back(readIndicatorLine(tokens, row_map, col_map));
                break;
            }

            default:
                break;
        }
    }

    if (saw_mps_section && !deferred_refusal.empty()) {
        throw std::runtime_error(deferred_refusal);
    }
    if (!obj_name.empty() && !have_objective_row) {
        throw std::runtime_error("MPS: OBJNAME '" + obj_name + "' names no free (N) row");
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
        {
            std::vector<Index>{}.swap(entry_rows);
        }
        {
            std::vector<Index>{}.swap(entry_cols);
        }
        {
            std::vector<Real>{}.swap(entry_vals);
        }

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

        prob.matrix = SparseMatrix(prob.num_rows, prob.num_cols, std::move(values),
                                   std::move(col_indices), std::move(row_starts));
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
                prob.row_lower[i] = (range != 0.0) ? rhs - std::abs(range) : -kInf;
                break;
            case 'G':
                prob.row_lower[i] = rhs;
                prob.row_upper[i] = (range != 0.0) ? rhs + std::abs(range) : kInf;
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

    extractIndicatorRows(prob, pending_indicators, range_values);
    return prob;
}

}  // namespace mipx
