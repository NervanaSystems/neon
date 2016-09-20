#!/bin/bash

usage() {
    cat <<EOM
    Usage:
    $(basename $0) input_dir out_dir
    input_dir: path to whale_data.zip
    out_dir: directory into which to output the processed files and manifests
EOM
    exit 0
}

[ -z $2 ] && { usage; }

# Ingest the data (won't rerun if it has already been done)
python data.py --input_dir $1 --out_dir $2

# Run in evaluation mode
python train.py

# Run in submission mode
python make_submission.py

echo Done
