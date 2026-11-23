\ An LP for mipx-presolve-bench's tests (#287), on which several of mipx's LP
\ reductions fire: a singleton row (s1) that fixes x1, a doubleton equality
\ (d1), a forcing row (f1) that fixes x6 and x7 at their lower bounds, a row
\ no point can violate (big), a pair of parallel rows (p1, p2), and a column
\ in no row (x9). The optimum: x1 = 2, x2 = 1, x3 = x4 = 0, x5 = 1, x6 = x7 =
\ 0, x8 = 10, x9 = 0, objective 2 + 2 + 1 - 10 = -5.
Minimize
 obj: x1 + 2 x2 + 3 x3 + x4 + x5 + x6 + x7 - x8 + x9
Subject To
 s1: x1 = 2
 d1: x3 - 2 x4 = 0
 f1: x6 + x7 <= 0
 big: x5 + x6 + x3 <= 1000
 p1: x5 + x2 >= 2
 p2: 2 x5 + 2 x2 >= 4
 r1: x8 + x3 + x5 <= 11
 r2: x2 + x4 >= 1
Bounds
 0 <= x1 <= 5
 0 <= x2 <= 10
 0 <= x3 <= 10
 0 <= x4 <= 10
 0 <= x5 <= 10
 0 <= x6 <= 10
 0 <= x7 <= 10
 0 <= x8 <= 10
 0 <= x9 <= 10
End
