#!/bin/bash

usage() {
    cat <<EOM
Usage:  $(basename $0) zip_file train_percentage
    zip_file: full path to whale_data.zip
    train_percentage: percentage of train set to use for training vs validation (default 80)
EOM
    exit 0
}

type sox >/dev/null 2>&1 || { echo >&2 "sox required but not installed."; exit 1; }
type parallel >/dev/null 2>&1 || { echo >&2 "parallel required but not installed."; exit 1; }

[ -z $1 ] && { usage; }
ZIP=$1

if [ ! -e $ZIP ]; then
    echo "$ZIP" not found
    exit 1
fi

# Use default 80 percent train
if [ ! -z $2 ]; then
    TRAINPCT=$2
else
    TRAINPCT=80
fi

if [ -z ${WHALE_DATA_PATH+x} ]; then echo "WHALE_DATA_PATH not set. Aborting."; exit 1; fi

ODIR=${WHALE_DATA_PATH%/}/whale-extracted

# Unzips whale_data.zip and converts to 16-bit PCM
unzip -q -d $ODIR $ZIP && find $ODIR -name '*.aiff' | parallel --progress 'sox {} {.}.wav && rm {}'

# Create label files
mkdir -p $ODIR/lbl && echo 1 > $ODIR/lbl/1.txt && echo 0 > $ODIR/lbl/0.txt

# Create test manifest
find $ODIR/data/test -name '*.wav' -printf '%p\n' > $ODIR/test-index.csv

# Create all train manifest
TDIR=$ODIR/data/train/
LDIR=$ODIR/lbl/
tail -n +2 $ODIR/data/train.csv | sed -e "{s|^|$TDIR|; s|.aiff|.wav|; s|$|.txt|; s|,|,$LDIR|;}" > $ODIR/all-index.csv

# Split the pos and neg examples
grep '0.txt$' $ODIR/all-index.csv > $ODIR/neg_train.csv
grep '1.txt$' $ODIR/all-index.csv > $ODIR/pos_train.csv

nn=$(cat $ODIR/neg_train.csv | wc -l)
np=$(cat $ODIR/pos_train.csv | wc -l)

head -$((nn * TRAINPCT / 100)) $ODIR/neg_train.csv > $ODIR/train-index.csv
head -$((np * TRAINPCT / 100)) $ODIR/pos_train.csv >> $ODIR/train-index.csv

head -$((nn * TRAINPCT / 100)) $ODIR/neg_train.csv | sed "s|,.*||" > $ODIR/noise-index.csv

tail -n +$((nn * TRAINPCT / 100)) $ODIR/neg_train.csv > $ODIR/val-index.csv
tail -n +$((np * TRAINPCT / 100)) $ODIR/pos_train.csv >> $ODIR/val-index.csv
