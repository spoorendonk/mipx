# Grid2 master fixture

`grid2_master_0001.mps` is the checksum-pinned v1 master for the public
CommaLAB grid2 instance after one pricing sweep. Set `scratch` to a writable
directory, then run these commands from the repository root.

```sh
./tests/data/download_commalab.sh
python3 tests/perf/generate_cg_masters.py --out "$scratch/cg-masters"
cp "$scratch/cg-masters/grid2.mps" tests/data/grid2_master_0001.mps
```

This coupled demand and capacity model has 169 rows, 200 columns and 529
nonzeros. It exercises the three CPU barrier backends with presolve on and
off and crossover disabled. From the slack basis it exercises hyper-sparse
row pricing, forced dense row pricing and the cost-triggered DSE fallback.
The reference objective is 3,040,780.
