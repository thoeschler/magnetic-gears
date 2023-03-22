#!/bin/bash

: '
for nb_it in {0..199}; do
    echo -e $'\n'"ITERATION:" $(($nb_it + 1)) "\n"
    python3 spur_gears/sample_torque.py "ball" $nb_it
done
'

for nb_it in {0..200}; do
    echo -e $'\n'"ITERATION" $(($nb_it + 1))
    python3 spur_gears/sample_torque.py "bar" $nb_it
done

for nb_it in {0..199}; do
    echo -e $'\n'"ITERATION:" $(($nb_it + 1))
    python3 spur_gears/sample_torque.py "cylinder_segment" $nb_it
done
