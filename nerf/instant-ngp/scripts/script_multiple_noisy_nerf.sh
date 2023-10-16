#!/bin/bash

#pretty sequences
# seq_list="6 20 33 36 43" 
# all trained sequences
# seq_list="2 3 4 5 6 9 10 11 12 13 14 16 17 18 19 20 21 22 23 24 25 26 27 31 32 33 34 35 36 37 38 39 40 41 43"
seq_list="37 38 39 40 41"
#still to do  


for i in $seq_list; do
    echo "Processing sequence $i"
    ./scripts/do_noisy_nerf.sh $i
done