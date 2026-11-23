#pragma once

#include "mipx/core.h"
#include "mipx/sparse_matrix.h"

#include <cstddef>
#include <vector>

namespace mipx {

/// Presolve's working matrix (#326): rows and columns both current under
/// inserts, deletes and coefficient changes, so a substitution can add the
/// entries it fills in. It is presolve's own structure; the solver's matrices
/// stay CSR-primary with a lazy CSC, and the reduced problem is exported as
/// CSR (toSparseMatrix).
///
/// An entry is a position, not a nonzero: a value may be exactly 0 and stays
/// an entry until erased, as the input's explicit zeros and a coefficient
/// changed to 0 do, so every row keeps the input's pattern (#338 review).
///
/// Entries live in an append-only pool with 32-bit ids, never reused or
/// reclaimed: the pool holds the input's nonzeros plus every insert of a new
/// position, an insert after an erase included. Each row keeps its entry ids
/// sorted by column and each column by row, the order of the CSR and CSC
/// presolve reads today, so pattern walks and hashes over a row are unchanged
/// by inserts. Erasing only marks an entry dead, so erasing while iterating is
/// safe; a dead id is never reused, so a list not yet compacted cannot alias a
/// new entry. Inserting into a list compacts it once it holds at least 8 more
/// dead ids than live ones; lists are not compacted otherwise. So do not
/// insert a *new* entry into a row or column while iterating that same row or
/// column; changing a value is fine. clearRow and clearCol empty that row's or
/// column's id list, so a range-for over rowIds(r) must not clear row r (the
/// index-based forEach walks may).
class PresolveMatrix {
public:
    PresolveMatrix() = default;
    /// Every stored entry of `matrix`, explicit zeros included. Each row's
    /// columns must be sorted and unique (asserted), or find() may miss them.
    explicit PresolveMatrix(const SparseMatrix& matrix);

    [[nodiscard]] Index numRows() const { return static_cast<Index>(row_size_.size()); }
    [[nodiscard]] Index numCols() const { return static_cast<Index>(col_size_.size()); }
    /// Live entries.
    [[nodiscard]] Index numEntries() const { return num_entries_; }
    [[nodiscard]] Index rowSize(Index row) const { return row_size_[row]; }
    [[nodiscard]] Index colSize(Index col) const { return col_size_[col]; }

    /// The value at (row, col), 0 when there is no entry.
    [[nodiscard]] Real coeff(Index row, Index col) const;
    [[nodiscard]] bool contains(Index row, Index col) const { return find(row, col) >= 0; }

    /// Inserts (row, col) or changes its value. A value of 0 is kept as an
    /// entry; erase removes one.
    void set(Index row, Index col, Real value);
    /// Erases (row, col); false when there was no entry.
    bool erase(Index row, Index col);
    /// Erases every entry of a row, or of a column.
    void clearRow(Index row);
    void clearCol(Index col);

    /// Random access, for walks that step through two rows at once or stop
    /// early: a row's or column's entry ids in order, dead ones included
    /// (skip them with isLive).
    [[nodiscard]] const std::vector<Index>& rowIds(Index row) const { return row_ids_[row]; }
    [[nodiscard]] const std::vector<Index>& colIds(Index col) const { return col_ids_[col]; }
    [[nodiscard]] bool isLive(Index entry) const { return entry_live_[entry] != 0; }
    [[nodiscard]] Index entryRow(Index entry) const { return entry_row_[entry]; }
    [[nodiscard]] Index entryCol(Index entry) const { return entry_col_[entry]; }
    [[nodiscard]] Real entryValue(Index entry) const { return entry_val_[entry]; }

    /// A live entry seen from its row (index = column) or its column (index =
    /// row), by value.
    struct Term {
        Index index;
        Real value;
    };

    /// The live entries of one row or column in order, as Terms: `for (const
    /// auto [col, a] : m.row(i))`. Reads the pool through the matrix at each
    /// step, so changing a value, erasing an entry, or inserting an entry
    /// that is not in the line being walked (walking row r: any entry of
    /// another row; walking column c: any entry of another column) is fine;
    /// inserting into, or clearing, the line being walked invalidates it
    /// (its id list may grow, compact or empty).
    class Line {
    public:
        class Iterator {
        public:
            Iterator(const PresolveMatrix* m, const Index* it, const Index* end, bool by_row)
                : m_(m), it_(it), end_(end), by_row_(by_row) {
                skipDead();
            }
            Term operator*() const {
                return {by_row_ ? m_->entry_col_[*it_] : m_->entry_row_[*it_],
                        m_->entry_val_[*it_]};
            }
            Iterator& operator++() {
                ++it_;
                skipDead();
                return *this;
            }
            bool operator==(const Iterator& other) const { return it_ == other.it_; }

        private:
            void skipDead() {
                while (it_ != end_ && m_->entry_live_[*it_] == 0) {
                    ++it_;
                }
            }
            const PresolveMatrix* m_;
            const Index* it_;
            const Index* end_;
            bool by_row_;
        };

        Line(const PresolveMatrix* m, const std::vector<Index>& ids, bool by_row)
            : m_(m), ids_(&ids), by_row_(by_row) {}
        [[nodiscard]] Iterator begin() const {
            const Index* end = ids_->data() + ids_->size();
            return {m_, ids_->data(), end, by_row_};
        }
        [[nodiscard]] Iterator end() const {
            const Index* end = ids_->data() + ids_->size();
            return {m_, end, end, by_row_};
        }

    private:
        const PresolveMatrix* m_;
        const std::vector<Index>* ids_;
        bool by_row_;
    };

    [[nodiscard]] Line row(Index r) const { return {this, row_ids_[r], true}; }
    [[nodiscard]] Line col(Index c) const { return {this, col_ids_[c], false}; }

    /// Calls f(col, value) for each live entry of a row, by column.
    template <typename F>
    void forEachInRow(Index row, F&& f) const {
        const auto& list = row_ids_[row];
        for (std::size_t k = 0; k < list.size(); ++k) {
            const Index e = list[k];
            if (entry_live_[e] != 0) {
                // Copies: an insert elsewhere from f may move the pool.
                const Index col = entry_col_[e];
                const Real value = entry_val_[e];
                f(col, value);
            }
        }
    }

    /// Calls f(row, value) for each live entry of a column, by row.
    template <typename F>
    void forEachInCol(Index col, F&& f) const {
        const auto& list = col_ids_[col];
        for (std::size_t k = 0; k < list.size(); ++k) {
            const Index e = list[k];
            if (entry_live_[e] != 0) {
                const Index row = entry_row_[e];
                const Real value = entry_val_[e];
                f(row, value);
            }
        }
    }

    /// The live entries as CSR, columns sorted within each row.
    [[nodiscard]] SparseMatrix toSparseMatrix() const;

private:
    /// The live entry at (row, col), or -1.
    [[nodiscard]] Index find(Index row, Index col) const;
    void kill(Index entry);
    void compact(std::vector<Index>& list) const;

    // Pool. A dead entry keeps its row and column, so lists stay sorted.
    std::vector<Index> entry_row_;
    std::vector<Index> entry_col_;
    std::vector<Real> entry_val_;
    std::vector<unsigned char> entry_live_;
    std::vector<std::vector<Index>> row_ids_;
    std::vector<std::vector<Index>> col_ids_;
    std::vector<Index> row_size_;
    std::vector<Index> col_size_;
    Index num_entries_ = 0;
};

}  // namespace mipx
