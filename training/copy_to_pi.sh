#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: bash $0 RUN_DIRECTORY" >&2
    exit 2
fi

run_dir=${1%/}
run_name=${run_dir##*/}
if [[ ! "$run_name" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
    echo "Use a run directory with a name like run2 or run-003." >&2
    exit 2
fi

if [[ "$1" != */* && ( -d "runs/$run_dir" || ! -d "$run_dir" ) ]]; then
    run_dir="runs/$run_dir"
fi
bundle="$run_dir/bundle"
files=(detector.tflite classifier.tflite objects.txt species.txt manifest.json)
for file in "${files[@]}"; do
    if [[ ! -s "$bundle/$file" ]]; then
        echo "Missing or empty bundle file: $bundle/$file" >&2
        exit 1
    fi
done

dataset=$(python3 -c 'import json, sys; print(json.load(sys.stdin)["dataset"])' < "$bundle/manifest.json")
if [[ ! "$dataset" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
    echo "Invalid dataset name in $bundle/manifest.json" >&2
    exit 1
fi

host=ferdi@192.168.2.117
destination="/home/ferdi/pigeon/pigeon_detector/data/models/pigeon-models-$dataset-$run_name"
ssh "$host" "mkdir -p -- '$destination'"
scp -- "${files[@]/#/$bundle/}" "$host:$destination/"
echo "Copied bundle to $host:$destination"
