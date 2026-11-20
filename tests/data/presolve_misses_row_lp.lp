\ An LP whose presolved answer misses its one row (issue #257). Doubleton
\ aggregation eliminates xe = 10000 - 10000 xk; xe >= 0 implies xk <= 1, a
\ projection within presolve's 1e-8 of xk's own bound, so it is skipped. The
\ reduced LP then takes xk = 1 + 5e-9, postsolve clamps xe = -5e-5 to 0, and
\ the row misses by 5e-5. The optimum is xe = 0, xk = 1, objective 0.
Minimize
 obj: xe
Subject To
 r1: xe + 10000 xk = 10000
Bounds
 0 <= xe <= 10000
 0 <= xk <= 1.000000005
End
