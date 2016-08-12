#!/bin/bash

usage() {
    cat <<EOM
    Usage:
    $(basename $0) input_rar input_zip
    input_rar: raw video files, can be obtained from http://crcv.ucf.edu/data/UCF101/UCF101.rar
    input_zip: zip of train/test partitions, UCF101TrainTestSplits-RecognitionTask.zip
EOM
    exit 0
}

# This function rescales the input video and breaks it up into clips of 16 frames each
split_vid() {
    VIDPATH=$1
    ffmpeg -v quiet -i $VIDPATH \
           -an -vf scale=171:128 -framerate 25 \
           -c:v mjpeg -q:v 3 \
           -f segment -segment_time 0.64 -reset_timestamps 1 \
           -segment_list ${VIDPATH%.avi}.csv \
           -segment_list_entry_prefix `dirname $VIDPATH`/ \
           -y ${VIDPATH%.avi}_%02d.avi
}

require() {
    type $1 > /dev/null 2>&1 || { echo >&2 $1 "required but not installed. Aborting."; exit 1; }
}


export -f split_vid

require ffmpeg
require unrar-nonfree
require parallel

if [ -z ${V3D_DATA_PATH+x} ]; then echo "V3D_DATA_PATH not set. Aborting."; exit 1; fi

[ -z $2 ] && { usage; }

RARFILE=$1
ZIPFILE=$2

OUTDIR=${V3D_DATA_PATH%/}/ucf-extracted
VIDDIR=$OUTDIR/UCF-101

mkdir -p $OUTDIR

# Extract the video files
unrar-nonfree x -o- $RARFILE $OUTDIR

# Make label index files
rm -f $OUTDIR/category-index.csv
idx=0
for i in $(find $VIDDIR -maxdepth 1 -mindepth 1 | sort); do
    echo $idx > $i/label.txt;
    echo `(basename $i)`,$idx >> $OUTDIR/category-index.csv
    idx=$((idx+1));
done


for setn in train test; do
    echo Processing $setn data...
    listfile=$OUTDIR/${setn}list.txt
    unzip -q -c $ZIPFILE ucfTrainTestlist/${setn}list01.txt | col -b | awk -vO=$VIDDIR/ '{print O$1}' > $listfile
    # Split the raw video files
    cat $listfile | parallel --joblog $OUTDIR/${setn}split.log --progress split_vid {}
    cat $(cat $listfile | sed "s/.avi$/.csv/") |
        awk -vFS=',' '{if ($3-$2>=0.63) {print $1}}' |
        sed "s|^\(.*\)/\(.*.avi\)|\1/\2,\1/label.txt|" > $OUTDIR/${setn}-index.csv
done

echo Done
