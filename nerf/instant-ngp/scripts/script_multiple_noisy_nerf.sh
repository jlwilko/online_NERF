#!/bin/bash

#12 was not good 
seq_list="6 20 33 36 43" 

for i in $seq_list; do
    echo "Processing sequence $i"
    ./scripts/do_noisy_nerf.sh $i
done