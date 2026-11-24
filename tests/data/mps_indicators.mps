* INDICATORS section (#254). B = 1 forces RX (X <= 2), B = 0 forces RY
* (Y <= 3). B = 1 gives 2 + 10 + 5 = 17, B = 0 gives 10 + 3 = 13, so the
* optimum is 17. Read as linear rows it would be 10, with the indicators
* dropped 25, and with the trigger values swapped 18.
NAME          MPSIND
OBJSENSE
    MAX
ROWS
 N  OBJ
 L  RX
 L  RY
COLUMNS
    X         OBJ       1.0          RX        1.0
    Y         OBJ       1.0          RY        1.0
    B         OBJ       5.0
RHS
    RHS       RX        2.0          RY        3.0
BOUNDS
 UP BND       X         10.0
 UP BND       Y         10.0
 BV BND       B
INDICATORS
 IF RX        B         1
 IF RY        B         0
ENDATA
