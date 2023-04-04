#!/bin/bash

# initialize a semaphore with a given number of tokens
open_sem(){
    mkfifo pipe-$$
    exec 3<>pipe-$$
    rm pipe-$$
    local i=$1
    for((;i>0;i--)); do
        printf %s 000 >&3
    done
}

# run the given command asynchronously and pop/push tokens
run_with_lock(){
    local x
    # this read waits until there is something to read
    read -u 3 -n 3 x && ((0==x)) || exit $x
    (
     ( "$@"; )
    # push the return code of the command to the semaphore
    printf '%.3d' $? >&3
    )&
}

magnet=$1
n_procs=$2
open_sem $n_procs
for par_nb in {1..456}; do
    echo "ITERATION" $((par_nb-1))
    run_with_lock python3 sample_torque.py $magnet $(( par_nb-1 )) 2>> log.txt
done