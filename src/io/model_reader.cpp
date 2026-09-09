#include "file_suffix.h"
#include "mipx/io.h"

#include <string>
#include <string_view>

namespace mipx {

namespace {

bool isCompressed(const std::string& filename) {
    return io_detail::stripCompressionSuffix(filename).size() != filename.size();
}

const char* formatName(ModelFormat format) {
    return format == ModelFormat::Lp ? "CPLEX LP" : "MPS";
}

}  // namespace

ModelFormat detectModelFormat(const std::string& filename) {
    const std::string_view stem = io_detail::stripCompressionSuffix(filename);
    if (io_detail::endsWithIgnoreCase(stem, ".lp")) {
        return ModelFormat::Lp;
    }
    // MPS is the historical default for every other name, including files with
    // no extension at all (Netlib instances are commonly stored that way).
    return ModelFormat::Mps;
}

LpProblem readModel(const std::string& filename, ModelFormat format) {
    const ModelFormat resolved =
        (format == ModelFormat::Auto) ? detectModelFormat(filename) : format;

    if (resolved == ModelFormat::Lp && isCompressed(filename)) {
        throw ModelFormatError("Cannot read '" + filename +
                               "' as CPLEX LP: the LP reader does not support compressed "
                               "input (only the MPS reader decompresses .gz/.bz2). "
                               "Decompress the file first.");
    }

    ReadDiagnostics diag;
    LpProblem problem =
        (resolved == ModelFormat::Lp) ? readLp(filename, &diag) : readMps(filename, &diag);

    // A file with content that carries no header exclusive to the format it was
    // read as is not a model in that format.
    //
    // Checking the headers rather than the parsed size matters both ways round:
    // a wrong-format file can parse to an empty model (issue #198's LP-as-MPS
    // reproducer), and it can just as well parse to a nonsense one, because the
    // shared BOUNDS section makes either reader manufacture columns out of the
    // other format's bound records.
    //
    // "Has content" likewise comes from the reader rather than from the file
    // size, which cannot answer it through a compression layer: an empty model
    // still occupies a gzip header's worth of bytes on disk.
    if (diag.saw_content && !diag.saw_format_section) {
        const std::string expected =
            (resolved == ModelFormat::Lp)
                ? "an objective (Minimize/Maximize), Subject To, General, Binary or End header"
                : "a ROWS, COLUMNS, RHS, RANGES or ENDATA header";
        throw ModelFormatError("Cannot read '" + filename + "' as " + formatName(resolved) +
                               ": the file is not empty but contains no header exclusive to " +
                               formatName(resolved) + " (expected " + expected +
                               "). The file is not in " + formatName(resolved) +
                               " format, or is truncated. Note that a BOUNDS section alone does "
                               "not identify a format -- both MPS and LP use it.");
    }

    return problem;
}

}  // namespace mipx
