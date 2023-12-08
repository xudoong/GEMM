import sys
from collections import defaultdict
import re
import numpy as np


def parse_log_file(file_path):
    with open(file_path, 'r') as file:
        log_content = file.read()

    # Define the pattern to extract relevant information
    # pattern = re.compile(r'Running kernel (\d+) on device (\d+).*?Average elapsed time: \(([\d.]+)\) s, performance: \(([\d.]+)\) GFLOPS. size: \((\d+)\)')
    pattern1 = re.compile(r'Running kernel (\d*) on device')
    pattern2 = re.compile(r'Average elapsed time: \(([\d.]+)\) s, performance: \(([\d.]+)\) GFLOPS. size: \((\d+)\)')

    # Find all matches in the log content
    matches1 = pattern1.findall(log_content)
    matches2 = pattern2.findall(log_content)

    # Organize the results into a list of dictionaries
    results = []
    for match1, match2 in zip(matches1, matches2):
        result_dict = {
            'kernel': int(match1),
            'performance': float(match2[1]),
        }
        results.append(result_dict)

    return results

# Example usage:
log_file_path = sys.argv[1]
parsed_results = parse_log_file(log_file_path)

kernel_performance_dict = defaultdict(lambda: [])
for result in parsed_results:
    kernel = result['kernel']
    gflops = result['performance']
    tflops = round(gflops / 1000, 1)
    kernel_performance_dict[kernel].append(tflops)

for k, v_list in kernel_performance_dict.items():
    mid = np.median(v_list)
    std = round(np.std(v_list), 2)
    print(f'kernel={k} mid={mid} TFLOPS std={std}')
