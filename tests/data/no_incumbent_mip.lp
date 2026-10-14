\ An integer program with no integer point: 2 (x + y + z) = 3 has no integer
\ solution, but its LP relaxation does. With presolve and cuts off and a node
\ limit of 1 the solver stops before it can prove that, holding no incumbent.
\ The CLI must print "Objective: none", not 0 (issue #224).
Minimize
 obj: x + y + z
Subject To
 c1: 2 x + 2 y + 2 z = 3
Bounds
 0 <= x <= 1
 0 <= y <= 1
 0 <= z <= 1
General
 x y z
End
