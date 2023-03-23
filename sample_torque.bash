#!/bin/bash

: '
for nb_it in {0..199}; do
    echo -e $'\n'"ITERATION:" $(($nb_it + 1)) "\n"
    python3 spur_gears/sample_torque.py "ball" $nb_it
done
'
: '
for nb_it in {0..200}; do
    echo -e $'\n'"ITERATION" $(($nb_it + 1))
    python3 spur_gears/sample_torque.py "bar" $nb_it
done
'
proc_num=$1
nb_procs=$2

for ((par_nb=$proc_num;par_nb<=8;par_nb+=nb_procs)); do
    echo -e $'\n'"ITERATION:" $(($par_nb + 1))
    python3 sample_torque.py "cylinder_segment" $par_nb
done
