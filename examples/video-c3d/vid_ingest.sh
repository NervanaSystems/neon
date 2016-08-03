#!/bin/bash
len=16  # number of frames per video

usage() {
    cat <<EOM
    Usage:
    $(basename $0) list_file video_directory output_directory
    list_file: either trainlist01.txt or testlist01.txt
    video_directory: location where UCF101.rar was extracted to
    output_directory: location to write segmented clips
EOM
    exit 0
}

[ -z $3 ] && { usage; }

VIDLIST=$1
INPATH=$2
OUTDIR=$3
mkdir -p $OUTDIR

type ffmpeg >/dev/null 2>&1 || { echo >&2 "ffmpeg required but not installed.  Aborting."; exit 1; }

function split_vid {
    VIDPATH=$1
    VID=`basename $VIDPATH`
    frame_dir=$OUTDIR/${VID%.avi}
    mkdir -p $frame_dir
    ffmpeg -v quiet -i $VIDPATH -vf scale=171:128 $frame_dir/f%04d.jpg
    frames=($frame_dir/*.jpg)
    nvids=$((${#frames[@]}/len))
    for i in $(seq 0 $((nvids-1))); do
        ofile=`printf "%s/%s_%02d.mp4" $OUTDIR ${VID%.avi} $i`
        ffmpeg -v quiet -f image2 -start_number $((i*len+1)) -i $frame_dir/f%04d.jpg -framerate 25 -c:v mjpeg -q:v 3 -vframes 16 $ofile
    done
    rm -rf $frame_dir
}

TOTAL=`cat $VIDLIST | wc -l`
STARTTIME=`date +%s`
curvid=0
for FF in $(cat $VIDLIST | sed "s|.*/\(.*.avi\).*$|\1|"); do
    split_vid $INPATH/$FF;
    echo -n "$curvid / $TOTAL   Elapsed Time: $(($(date +%s) - STARTTIME))" seconds $'\r'
    curvid=$((curvid+1))
done

echo Done
