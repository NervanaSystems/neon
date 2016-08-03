#!/bin/bash
usage() {
    cat <<EOM
    Usage:
    $(basename $0) train_directory test_directory label_directory class_map
    train_directory: where ingested video clips from training set have been written
    test_directory: where ingested video clips from test set have been written
    label_directory: where to write the label files for each category
    class_map: file containing action to integer mappings (ucfTrainTestlist/classInd.txt)
EOM
    exit 0
}

[ -z $4 ] && { usage; }

TRAINDIR=$1
TESTDIR=$2
LABELDIR=$3
CLASSMAP=$4


# Make the labels file
mkdir -p $LABELDIR
idx=0
awk '{print $2}' $CLASSMAP | col -b | while read ACTION; do echo $idx > $LABELDIR/$ACTION.txt; idx=$((idx+1)); done


find $TRAINDIR -name '*.mp4' | sed "s|\(^.*\)/v_\([a-Z]*\)\(_g.*\)|\1/v_\2\3,$LABELDIR/\2.txt|" > $TRAINDIR/manifest.csv
find $TESTDIR -name '*.mp4' | sed "s|\(^.*\)/v_\([a-Z]*\)\(_g.*\)|\1/v_\2\3,$LABELDIR/\2.txt|" > $TESTDIR/manifest.csv


