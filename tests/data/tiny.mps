NAME          tiny
ROWS
 N  obj
 L  c1
 G  c2
 E  c3
COLUMNS
    x1        obj           1.0   c1            2.0
    x1        c2            1.0
    INT1      'MARKER'                 'INTORG'
    x2        obj           3.0   c1            1.0
    x2        c3            1.0
    INT1END   'MARKER'                 'INTEND'
    x3        obj           2.0   c2            1.0
    x3        c3            1.0
RHS
    rhs       c1            10.0  c2            5.0
    rhs       c3            7.0
BOUNDS
 UP bnd       x1            8.0
 BV bnd       x2
 LO bnd       x3            1.0
 UP bnd       x3            5.0
ENDATA
