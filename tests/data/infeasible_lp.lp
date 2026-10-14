\ A continuous LP with no feasible point: x + y >= 5 but x, y <= 2.
\ The CLI must not print an objective for it (issue #206).
Minimize
 obj: x + y
Subject To
 c1: x + y >= 5
Bounds
 0 <= x <= 2
 0 <= y <= 2
End
