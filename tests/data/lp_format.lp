\ Reproducer from issue #198: an LP-format file whose optimum is 3.
\ Read as MPS this parses to a 0x0 model and used to report Optimal 0.0.
Maximize
 obj: x + y
Subject To
 c1: x + y <= 3
Bounds
 0 <= x <= 2
 0 <= y <= 2
End
