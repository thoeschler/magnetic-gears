#!/bin/bash

proc_num=$1
nb_procs=$2
magnet=$3

for ((par_nb=$proc_num;par_nb<=500;par_nb+=nb_procs)); do
    echo -e $'\n'"ITERATION:" $(($par_nb-1))
    python3 sample_torque.py $magnet $(($par_nb-1))
done
