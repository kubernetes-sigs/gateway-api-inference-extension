#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source ${SCRIPT_DIR}/env.sh
regex=${regex:-".*.pbtxt"}

while read -r file; do
    filename=$(basename "$file")
    # Extract the filename without extension using parameter expansion
    filename_without_ext="${filename%.*}"
    echo "Running benchmark for $filename_without_ext"
    benchmarks=${filename_without_ext} ${SCRIPT_DIR}/run_benchmarks_file.bash
done < <(find "${SCRIPT_DIR}/../catalog/" -regex "$regex" -print | sort )