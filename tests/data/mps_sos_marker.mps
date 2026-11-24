* CPLEX's marker form of SOS inside COLUMNS (#254): refused rather than
* dropped with the markers.
NAME          MPSSOSMARKER
ROWS
 N  OBJ
 L  CAP
COLUMNS
    S1        SOS       'MARKER'                 'SOSORG'
    X         OBJ       -1.0         CAP       1.0
    Y         OBJ       -2.0         CAP       1.0
    S1        SOS       'MARKER'                 'SOSEND'
RHS
    RHS       CAP       2.0
BOUNDS
 UP BND       X         1.0
 UP BND       Y         1.0
ENDATA
