# Grid1 LP sequence fixture

Run these commands from the repository root to fetch the checksum-pinned
public CommaLAB archives and regenerate the fixture.

```sh
./tests/data/download_commalab.sh
python3 tests/perf/generate_grid1_lpseq.py --out tests/data/lpseq/grid1
```

The generator uses one shortest path per demand, ordered by decreasing
demand volume with ties in archive order. It starts with artificial
columns and ten paths and then raises artificial costs, adds capacity
rows and batches of paths, and removes two rows and two paths before the
last path enters. The nine snapshots have 50–62 rows and 60–99 columns.
The Tier 0 replays cover the same solver arms and edit types as the
original sequence. Solve 3 still adds 21 columns and triggers warm
primal pivots and bound perturbation. Solve 4 adds another seven columns
and checks that the per-solve column count resets. Reference objectives
were cross-checked with independent simplex solves. The two trailing
zeroes in each solve record are placeholders for iteration and time
metadata. The generator checks all three file hashes before writing them.
