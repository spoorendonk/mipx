NAME          CLIFEATURES
ROWS
 N  COST
 L  C12
 L  C23
 L  C13
 L  P45
 L  P46
 L  P56
 L  KNAP
COLUMNS
    X1        COST      -1.0         C12       1.0
    X1        C13       1.0
    X2        COST      -1.0         C12       1.0
    X2        C23       1.0
    X3        COST      -1.0         C23       1.0
    X3        C13       1.0
    X4        COST      -2.0         P45       1.0
    X4        P46       1.0
    X5        COST      -1.0         P45       -1.0
    X5        P56       1.0
    X6        COST      -1.0         P46       -1.0
    X6        P56       1.0
    Y1        COST      -4.0         KNAP      3.0
    Y2        COST      -6.0         KNAP      5.0
    Y3        COST      -8.0         KNAP      7.0
    Y4        COST      -10.0        KNAP      9.0
    Y5        COST      -12.0        KNAP      11.0
RHS
    RHS       C12       1.0          C23       1.0
    RHS       C13       1.0          P56       1.0
    RHS       KNAP      17.0
BOUNDS
 BV BND       X1
 BV BND       X2
 BV BND       X3
 BV BND       X4
 BV BND       X5
 BV BND       X6
 BV BND       Y1
 BV BND       Y2
 BV BND       Y3
 BV BND       Y4
 BV BND       Y5
ENDATA
