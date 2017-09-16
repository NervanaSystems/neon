#!/bin/bash

usage() {
    cat <<EOM
    Usage:
    $(basename $0) input_dir out_dir
    input_dir: location of raw video files and meta_data, can be obtained from
    http://crcv.ucf.edu/data/UCF101/UCF101.rar
    out_dir: directory into which to output the processed files and manifests
EOM
    exit 0
}

require() {
    type $1 > /dev/null 2>&1 || { echo >&2 $1 "required but not installed. Aborting."; exit 1; }
}

require_input() {
    [ -e $1 ] || { echo >&2 "Missing input file " $1; exit 1; }
}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

[ -z $2 ] && { usage; }

INPUT_DIR=$1
V3D_DATA_PATH=$2
OUTDIR=${V3D_DATA_PATH%/}/ucf-extracted
VIDDIR=$OUTDIR/UCF-101

mkdir -p $OUTDIR

# Save train config
echo "manifest = [test:$OUTDIR/test-index.csv, train:$OUTDIR/train-index.csv]" > $SCRIPT_DIR/train.cfg
echo "manifest_root = $OUTDIR" >> $SCRIPT_DIR/train.cfg
echo -e "epochs = 18\nbatch_size = 32\nrng_seed = 0\nverbose = True" >> $SCRIPT_DIR/train.cfg
echo "eval_freq = 1" >> $SCRIPT_DIR/train.cfg
echo "log = $V3D_DATA_PATH/train.log" >> $SCRIPT_DIR/train.cfg
echo "save_path = $V3D_DATA_PATH/UCF101-C3D.p"  >> $SCRIPT_DIR/train.cfg

# Save test config
echo "manifest = [categories:$OUTDIR/category-index.csv, test:$OUTDIR/test-index.csv]" > $SCRIPT_DIR/test.cfg
echo "manifest_root = $OUTDIR" >> $SCRIPT_DIR/test.cfg
echo -e "batch_size = 32\nrng_seed = 0\nverbose = True" >> $SCRIPT_DIR/test.cfg
echo "log = $V3D_DATA_PATH/test.log" >> $SCRIPT_DIR/test.cfg
echo "model_file = $V3D_DATA_PATH/UCF101-C3D.p"  >> $SCRIPT_DIR/test.cfg

if [ -f $OUTDIR/test-index.csv ] && [ -f $OUTDIR/train-index.csv ]; then
    echo "Ingestion already completed, delete manifests to re-ingest"
    exit 0
fi

RARFILE=${INPUT_DIR%/}/UCF101.rar
ZIPFILE=${INPUT_DIR%/}/UCF101TrainTestSplits-RecognitionTask.zip

require_input $RARFILE
require_input $ZIPFILE

require ffmpeg
require unrar-nonfree
require parallel

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

export -f split_vid

for setn in train test; do
    echo Processing $setn data...
    outfile=$OUTDIR/${setn}-index.csv
    listfile=$OUTDIR/${setn}list.txt
    unzip -q -c $ZIPFILE ucfTrainTestlist/${setn}list01.txt | col -b | awk -vO=$VIDDIR/ '{print O$1}' > $listfile
    # Split the raw video files
    cat $listfile | parallel --joblog $OUTDIR/${setn}split.log --progress split_vid {}
    cat $(cat $listfile | sed "s/.avi$/.csv/") |
        awk -vFS=',' '{if ($3-$2>=0.63) {print $1}}' |
        sed "s|^$OUTDIR/\(.*\)/\(.*.avi\)|\1/\2,\1/label.txt|" > $outfile
    tmpfile=$(mktemp)
    paste -d$'\t' <(cut -d',' -f1 $outfile) <(cut -d',' -f2 $outfile | sed "s|\(.*\)|$OUTDIR/\1|" | xargs cat) > $tmpfile
    mv $tmpfile $outfile
    rm -rf $tmpfile
    sed -i "1s/^/@FILE\\tASCII_INT\n/" $outfile
done

unset split_vid


echo Done
