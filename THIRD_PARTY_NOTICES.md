# Third-party notices

mipx contains code translated from the projects below. Each translated file
carries the project's notice at its head and names the source files and
version it follows.

## HiGHS

<https://github.com/ERGO-Code/HiGHS>

Translated in mipx:

- `src/lp/rank_revealing_lu.h`, `src/lp/rank_revealing_lu.cpp`: the
  elimination of `HFactor::build()` (HiGHS 1.16: `highs/util/HFactor.cpp`,
  `highs/util/HFactor.h`, `highs/util/HFactorConst.h`).
- `src/presolve/dependent_equations.h`, `src/presolve/dependent_equations.cpp`:
  `HPresolve::removeDependentEquations()` up to the removal (HiGHS 1.16:
  `highs/presolve/HPresolveDependent.cpp`).
- `src/presolve/dual_bounds.h`, `src/presolve/dual_bounds.cpp`: the implied
  column and row-dual bounds of `HPresolve` and the sums of
  `HighsLinearSumBounds` (HiGHS 1.16: `highs/presolve/HPresolve.cpp`,
  `highs/util/HighsLinearSumBounds.h`, `highs/util/HighsLinearSumBounds.cpp`).
- `src/presolve/presolve.cpp`, in part: single-equation dual fixing,
  the selection loop of `HPresolve::aggregator()` and
  `HPresolve::detectDominatedCol()` (HiGHS 1.16:
  `highs/presolve/HPresolveDualFixing.cpp`,
  `highs/presolve/HPresolveAggregator.cpp`, `highs/presolve/HPresolve.cpp`).
- `src/lp/dual_simplex.cpp`, in part: the dual ratio test of `HEkkDualRow`
  and the cost perturbation of `HEkk::initialiseCost()`, with the deviations
  the file's head names (first translated from an unrecorded version,
  following HiGHS 1.16:
  `highs/simplex/HEkkDualRow.cpp`, `highs/simplex/HEkk.cpp`).

```
MIT License

Copyright (c) 2026 HiGHS

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
