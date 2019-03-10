#!/bin/bash

#python scaneagle_check_krylov_pert.py full

declare -a ndirection_array=(1 2 3 4 5 6)

for i in {0..5}
do
  for j in "${ndirection_array[@]}"
  do
    python -W ignore scaneagle_check_krylov_pert.py reduced $i $j
  done
done
