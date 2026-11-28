\ The cap on x + y implies x <= 2. An independent fixed column makes
\ the CLI use the presolved model and report the reduction counts.
Minimize
 obj: - x - y - z + fixed
Subject To
 cap: x + y <= 7
 xz: x + z <= 12
 yz: y + z <= 15
Bounds
 0 <= x <= 10
 5 <= y <= 10
 0 <= z <= 10
 fixed = 1
End
