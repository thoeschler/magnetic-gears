#!/bin/bash
# import mult procs from subdirectory
DIRNAME=$(dirname -- "$0")
source "$DIRNAME/../mult_procs.sh"

# set current directory as python path
export PYTHONPATH=.

n_procs=$1
open_sem $n_procs
for par_nb in {1..114}; do
    echo "ITERATION" $((par_nb-1))
    run_with_lock python3 parameter_study_source/parameter_study_d_p/parameter_study_d.py $(( par_nb-1 )) 2>>"$DIRNAME/log.txt"
done
