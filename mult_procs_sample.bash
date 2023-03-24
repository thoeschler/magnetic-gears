#!/bin/bash

# number of processes
nb_procs=$1
magnet=$2

for ((proc_num=1;proc_num<=$nb_procs;proc_num++)); do
    ./sample_torque.bash $proc_num $nb_procs $magnet &
done