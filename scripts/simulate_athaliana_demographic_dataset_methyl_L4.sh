#!/bin/bash

# make sure directory exists

max_processes=500
for index in {1..500}; do
    python3 ~/aTMi/aTMi/dataset.py \
		 --population_time arabidopsis_methyl \
		 --population_size arabidopsis_methyl \
		 --simulation arabidopsis_L4_n10 \
		 --dataset arabidopsis_L4_n10 \
		 --ith_chunk "$index" &
 
 	if [ $((index % max_processes)) -eq 0 ]; then
		wait
    fi
done
wait

for index in {501..600}; do
    python3 ~/aTMi/aTMi/dataset.py \
		 --population_time arabidopsis_methyl \
		 --population_size arabidopsis_methyl_constant \
		 --simulation arabidopsis_L4_n10 \
		 --dataset arabidopsis_L4_n10 \
		 --ith_chunk "$index" &
 
 	if [ $((index % max_processes)) -eq 0 ]; then
		wait
    fi
done
wait
 
 
 
