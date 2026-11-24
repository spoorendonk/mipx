* SOS section (#254), CPLEX/Gurobi layout. All columns are continuous, so
* only the sets make this a MIP. S1 on X, Y, Z: one nonzero, Z = 1 gives 3
* (5 without the set). S2 on A, B, C, unweighted (positions 1, 2, 3): A and C
* are not adjacent, so A + C is at most 1 (2 without the set). Optimum 4.
NAME          MPSSOS
OBJSENSE
    MAX
ROWS
 N  OBJ
 L  CAP
COLUMNS
    X         OBJ       1.0          CAP       1.0
    Y         OBJ       2.0          CAP       1.0
    Z         OBJ       3.0          CAP       1.0
    A         OBJ       1.0
    B         OBJ       0.0
    C         OBJ       1.0
RHS
    RHS       CAP       2.0
BOUNDS
 UP BND       X         1.0
 UP BND       Y         1.0
 UP BND       Z         1.0
 UP BND       A         1.0
 UP BND       B         1.0
 UP BND       C         1.0
SOS
 S1 SOS       s1        1
    X         1
    Y         2
    Z         3
 S2 s2
    A
    B
    C
ENDATA
