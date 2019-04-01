#!/bin/bash

# declare -a case_array=("6rv_1e_1_1_2" "6rv_1e_1_2_2" "6rv_1e_1_3_2" "6rv_1e_1_4_2" "6rv_1e_1_5_2" "6rv_mc_full_2" "deterministic")
declare -a case_array=("deterministic_7rv" "7rv_1e_1_2_2")
declare -a rv_array=("Mach_number" "CT" "W0" "R" "load_factor" "mrho")
declare -a dir_arr=("V0")

for i in "${case_array[@]}"
do
  : '
  for j in "${rv_array[@]}"
  do
    echo $i $j
    python plot_fburn_vs_rvs.py generate_data $j $i 3
    # python generate_se.py $j $i 3
  done
  '
  for j in {0..6}
  do
    echo $i $j
    python plot_fburn_vs_rvs.py generate_data_dom_dir $i $j
  done
done

python plot_fburn_vs_rvs.py plot_dom_dir 0
python plot_fburn_vs_rvs.py plot_dom_dir 1
python plot_fburn_vs_rvs.py plot_dom_dir 2
python plot_fburn_vs_rvs.py plot_dom_dir 3
python plot_fburn_vs_rvs.py plot_dom_dir 4
python plot_fburn_vs_rvs.py plot_dom_dir 5
python plot_fburn_vs_rvs.py plot_dom_dir 6
