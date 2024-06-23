import os
import re
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

kernel_name_map = {
    0: 'cublas',
    3: 'wmma_smem',
    4: 'wmma_smem_opt',
    5: 'wmma_stage',
    6: 'wmma_stage_dbreg',
    7: 'mma_padding',
    8: 'mma_permute',
    9: 'mma_stage',
    10: 'mma_stage_dbreg',
    11: 'mma_swizzle',
    12: 'mma_swizzle_opt',
}

def parse_digits(line: str):
    return re.findall(r'\d*\.?\d+', line)


def parse_log(file_path: str):
    size_list = []
    tflops_list = []

    with open(file_path, 'r') as f:
        for line in f.readlines():
            if line.startswith('Size'):
                digits = parse_digits(line)
                assert len(digits) > 0
                size = int(digits[0])
                size_list.append(size)
            if line.startswith('TFLOPS'):
                digits = parse_digits(line)
                mid_tflops = float(digits[2])
                p95_tflops = float(digits[4])
                tflops_list.append(mid_tflops)
    return size_list, tflops_list


if __name__ == '__main__':
    log_path = './result/log'
    if len(sys.argv) > 1:
        log_path = sys.argv[1]

    result_dict = {}
    size_list = []
    for file in sorted(os.listdir(log_path), key=lambda x: int(parse_digits(x)[0])):
        file_name = file.split('.')[0]
        kernel_id = int(parse_digits(file)[0])
        kernel_name = kernel_name_map[kernel_id]
        kernel_name = f'k{kernel_id:02d}-{kernel_name}'
        if len(file_name.split('_')) > 1:
            kernel_name += '-' + file_name.split('_')[-1]
        file_path = os.path.join(log_path, file)
        size_list, tflops_list = parse_log(file_path)
        result_dict[kernel_name] = tflops_list

    df = pd.DataFrame(result_dict)
    df.index = size_list
    print(df)

    df.to_csv('./result/performance.csv')

    df.plot(marker='x', linestyle='-', figsize=(15, 12))
    plt.title('HGEMM on A100 GPU')
    plt.xlabel('size (M=N=K)')
    plt.ylabel('TFLOPS')

    plt.savefig('./result/performance.png', dpi=150)
