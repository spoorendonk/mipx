\ An integer program with no feasible point at all: x + y >= 5 but x, y <= 2.
\ The CLI must print "Objective: none", not 0 (issue #224).
Minimize
 obj: x + y
Subject To
 c1: x + y >= 5
Bounds
 0 <= x <= 2
 0 <= y <= 2
General
 x y
End
