#!/bin/bash

kernel_list="0 3 4 5 6 7 8 9 10 11 12"

mkdir -p result/log

for kernel in ${kernel_list}
do
  ./hgemm ${kernel} --skip_ver | tee result/log/kernel${kernel}.log
done

