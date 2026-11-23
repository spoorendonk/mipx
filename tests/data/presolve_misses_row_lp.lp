\ An LP whose presolved answer misses a row. Doubleton aggregation of r1
\ substitutes x2 = 1 + 1000000 x1 into r0, leaving 999999 x1 = 9999999; x1 is
\ set from that row's bound, off its exact value by about 3e-10, and postsolve
\ rebuilds x2 through the 1e6 coefficient, so r0 misses by about 3e-4. (Issue
\ #257's model missed through a skipped projection instead, fixed by #325.)
\ The optimum is x1 = 9999999 / 999999, objective 10.000009000009.
Minimize
 obj: x1
Subject To
 r0: x2 - x1 = 10000000
 r1: x2 - 1000000 x1 = 1
End
