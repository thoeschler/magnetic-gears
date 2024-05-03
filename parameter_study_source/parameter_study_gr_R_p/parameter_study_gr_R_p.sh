#!/bin/bash
# import mult procs from subdirectory
DIRNAME=$(dirname -- "$0")
source "$DIRNAME/../mult_procs.sh"

# set current directory as python path
export PYTHONPATH=.

magnet=$1
n_procs=$2
open_sem $n_procs
for par_nb in {17..400}; do
    echo "ITERATION" $((par_nb-1))>>"$DIRNAME/log.txt"
    run_with_lock python3 parameter_study_source/parameter_study_gr_R_p/parameter_study_gr_R_p.py $magnet $(( par_nb-1 )) 2>>"$DIRNAME/log.txt"
done
