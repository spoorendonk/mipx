* OBJNAME (#254) picks the second free row, SECOND, as the objective; the
* first, FIRST, is dropped. Minimizing SECOND = -X with X <= 4 gives -4;
* minimizing FIRST = X would give 0.
NAME          MPSOBJNAME
OBJNAME
    SECOND
ROWS
 N  FIRST
 N  SECOND
 L  CAP
COLUMNS
    X         FIRST     1.0          SECOND    -1.0
    X         CAP       1.0
RHS
    RHS       CAP       4.0
ENDATA
