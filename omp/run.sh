#!/bin/bash

for x in {1..20}
do
    export OMP_NUM_THREADS=${x} && taskset fffff00000 ./main.x 1920
    echo ""
done

# for x in 1 2 4 8 10 20
# do
#     export OMP_NUM_THREADS=${x} && taskset fffff00000 ./main.x 3840
#     echo ""
# done
