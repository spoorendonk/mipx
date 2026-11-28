execute_process(
    COMMAND "${REPLAY}" "${SEQUENCE}" --arm dual-warm --arm dual-cold --arm barrier
        --primal-bound-perturbation --dual-stats-csv "${STATS}"
    RESULT_VARIABLE result OUTPUT_VARIABLE replay_output ERROR_VARIABLE replay_error)
if(NOT result STREQUAL "0")
    message(FATAL_ERROR "grid1 replay failed (${result}): ${replay_error}")
endif()
if(NOT replay_output MATCHES "^seq,arm,status,objective,recorded,rel_err,iterations,seconds,reloaded\n")
    message(FATAL_ERROR "The main replay CSV header changed")
endif()

file(STRINGS "${STATS}" rows)
list(POP_FRONT rows header)
if(NOT header STREQUAL "seq,arm,added_columns,dual_pivots,primal_pivots,primal_degenerate_pivots,primal_bound_perturbation_activations")
    message(FATAL_ERROR "Unexpected dual stats CSV header")
endif()
list(LENGTH rows count)
if(NOT count EQUAL 18)
    message(FATAL_ERROR "Expected two dual rows for each of the nine solves, got ${count}")
endif()
foreach(row IN LISTS rows)
    if(NOT row MATCHES "^[0-9]+,dual-(warm|cold),[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+$")
        message(FATAL_ERROR "Unexpected dual stats row: ${row}")
    endif()
endforeach()

# Solve 3 adds 21 columns and exercises primal bound perturbation.
# Solve 4 adds 7 more. Both arms see the new columns and the count resets.
string(JOIN "\n" data ${rows})
if(NOT data MATCHES "\n3,dual-warm,21,[0-9]+,[1-9][0-9]*,[0-9]+,[1-9][0-9]*\n"
   OR NOT data MATCHES "\n3,dual-cold,21,"
   OR NOT data MATCHES "\n4,dual-warm,7,"
   OR NOT data MATCHES "\n4,dual-cold,7,")
    message(FATAL_ERROR "Missing grid1 added columns, primal pivots, or perturbation activation")
endif()

file(READ "${STATS}" saved_stats)
execute_process(
    COMMAND "${REPLAY}" "${SEQUENCE}" --csv "${STATS}" --dual-stats-csv "${STATS}"
    RESULT_VARIABLE result ERROR_VARIABLE replay_error)
if(NOT result STREQUAL "2" OR NOT replay_error MATCHES "must name different files")
    message(FATAL_ERROR "The replay must reject a shared output destination")
endif()
file(READ "${STATS}" retained_stats)
if(NOT retained_stats STREQUAL saved_stats)
    message(FATAL_ERROR "The rejected replay modified the existing CSV")
endif()
