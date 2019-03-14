#!/bin/bash

declare -a case_array=("6rv_1e_1_1_2" "6rv_1e_1_2_2" "6rv_1e_1_3_2" "6rv_1e_1_4_2" "6rv_1e_1_5_2" "6rv_mc_full_2" "deterministic")
declare -a rv_array=("Mach_number" "CT" "W0" "R" "load_factor" "mrho")

for i in "${case_array[@]}"
do
  for j in "${rv_array[@]}"
  do
    echo $i $j
    python plot_fburn_vs_rvs.py generate_data $j $i 2
  done
done
