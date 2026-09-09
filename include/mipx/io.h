#pragma once

#include "mipx/lp_problem.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace mipx {

/// Structural diagnostics collected while parsing a model file.
///
/// `saw_format_section` distinguishes "this file is in the format we parsed it
/// as" from "this file is not in that format at all". Readers ignore lines
/// belonging to no section they know, so a file in the wrong format parses
/// without any error -- to an empty model, or, where the two formats share a
/// section keyword, to a nonsense one. The flag is what makes that detectable.
/// See readModel().
struct ReadDiagnostics {
    /// True when the reader saw at least one section header that only its own
    /// format has: ROWS / COLUMNS / RHS / RANGES / ENDATA for MPS, or an
    /// objective (Minimize/Maximize) / Subject To / General / Binary / End
    /// header for LP.
    ///
    /// Two headers are deliberately excluded. MPS NAME carries no model data,
    /// so a truncated file holding only NAME is no evidence of an MPS body.
    /// BOUNDS belongs to both formats, so counting it would let each reader
    /// mistake the other's files for its own.
    bool saw_format_section = false;

    /// True when the reader saw at least one non-empty line of input.
    ///
    /// This is what "the input file was not empty" has to mean: the file size
    /// on disk cannot answer it through a compression layer, where an empty
    /// model still occupies a gzip header's worth of bytes.
    bool saw_content = false;
};

/// Thrown by readModel() when a file cannot be read as the format it was given.
///
/// Distinct from the plain std::runtime_error a reader raises for an
/// unreadable file, so callers can offer format-specific advice without
/// attaching it to every I/O failure.
class ModelFormatError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/// Model file formats understood by readModel().
enum class ModelFormat {
    Auto,  ///< Choose from the filename extension.
    Mps,
    Lp
};

/// Read an MPS file (fixed or free format). Detects .gz for gzip.
/// Pass `diag` to learn whether the file actually looked like MPS.
LpProblem readMps(const std::string& filename, ReadDiagnostics* diag = nullptr);

/// Write an MPS file (free format).
void writeMps(const std::string& filename, const LpProblem& problem);

/// Read a CPLEX-style LP file.
/// Pass `diag` to learn whether the file actually looked like an LP file.
LpProblem readLp(const std::string& filename, ReadDiagnostics* diag = nullptr);

/// Resolve the reader to use for `filename`.
///
/// Any single trailing compression suffix understood by the MPS reader (.gz,
/// .bz2) is stripped first, so `model.mps.gz` resolves to Mps. A `.lp`
/// extension selects Lp; every other extension, and no extension at all,
/// resolves to Mps. Extensions are matched case-insensitively, so `Model.LP`
/// resolves to Lp.
ModelFormat detectModelFormat(const std::string& filename);

/// Read a model file, picking the reader from `format` (or from the filename
/// when `format` is Auto).
///
/// Throws ModelFormatError, naming the file and the format problem, when a file
/// with content yields no section header exclusive to the format it was read as
/// -- that is a wrong-format, truncated or otherwise unparseable file, not a
/// model, whatever it happened to parse to. An empty file, and a format-valid
/// file that genuinely describes an empty model, are both read without error,
/// compressed or not.
LpProblem readModel(const std::string& filename, ModelFormat format = ModelFormat::Auto);

/// Entry from a .solu file.
struct SoluEntry {
    std::string name;
    Real value;
    bool is_infeasible = false;
};

/// Read a .solu file with known optimal values.
std::vector<SoluEntry> readSolu(const std::string& filename);

}  // namespace mipx
