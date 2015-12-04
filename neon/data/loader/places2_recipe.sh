work_dir=$1
src_dir=$2
batch_size=5000

die_with_msg() {
    echo $1
    exit 1
}

die_if_missing() {
    if ! [[ -e $1 ]]; then
        die_with_msg "Required source file $1 not found"
    fi
}

die_if_unable_to_write() {
    if ! [ -e $1 ]; then
        mkdir -p $1
        if [ $? -ne 0 ]; then
            die_with_msg "Unable to create $1 exiting ..."
        fi
    else
        if ! [ -w $1 ]; then
            die_with_msg "$1 not writeable..."
        fi
    fi
}

write_meta_file() {
    local macro_dir=$1
    local nclass=$2
    macro_meta=$macro_dir/macrobatch_meta
    nset=0
    echo "Writing metafile to $macro_meta"
    echo "nclass      $nclass"       > $macro_meta
    for settype in train val; do
        nrec=`cat   $macro_dir/${settype}_list* | wc -l`
        strt=$nset
        nset=`ls -1 $macro_dir/${settype}_list* | wc -l`
        echo "${settype}_nrec  $nrec" >> $macro_meta
        echo "n${settype}      $nset" >> $macro_meta
        echo "${settype}_start $strt" >> $macro_meta

    done

    echo "R_mean      104.412277"    >> $macro_meta
    echo "G_mean      119.213318"    >> $macro_meta
    echo "B_mean      126.806091"    >> $macro_meta

    nbatch=`ls -1 $macro_dir/{val_list,train_list}* | wc -l`

    # Now we need to get the largest individual datasize across all macrobatches
    maxval=`for i in $(seq $nbatch); do
        xxd -p -s 24 -l 4 $macro_dir/macrobatch_$i |
        awk '{for (i=7; i>0; i-=2) s = (256 * s) + strtonum("0x"substr($1,i,2))} END {print s}'
    done | sort -n -r | head -1`

    echo "item_max_size" $maxval >> $macro_meta
}

write_macrobatches() {
    local macrowriter=$1
    local macro_dir=$2
    echo "Writing macrobatches to $macro_dir"
    batchidx=0
    for list in $(\ls -1 $macro_dir/{train_list*,val_list*}); do
        $macrowriter $list $macro_dir/macrobatch_$batchidx
        ((batchidx++))
    done
}


train_tar=$src_dir/train.tar
val_tar=$src_dir/val.tar
meta_src=$src_dir/Places2_devkit.zip
macro_writer=./img_macrowriter

# Make sure the input files exist
src_files=($train_tar $val_tar $meta_src $macro_writer)
for f in ${src_files[*]}; do
    die_if_missing $f
done

# Make sure we can write to output directory
die_if_unable_to_write $work_dir

for outdir in train val batchout; do
    mkdir -p $work_dir/$outdir
done

# Check that we can use macrowriter
if [[ -x $macro_writer ]]; then
    die_with_msg "File '$macro_writer' is not executable"
fi

#output directories
train_dir=$work_dir/train
val_dir=$work_dir/val
macro_dir=$work_dir/batchout

metadir=$work_dir/`basename ${meta_src%%.zip}`
train_meta=$metadir/data/train.txt
val_meta=$metadir/data/val.txt

extract_imgs() {
    echo "Extracting training files..."
    for i in $(tar -tf $train_tar | grep tar); do
        tfile=$work_dir/$i
        tar -xf $train_tar -C $work_dir $i
        tar -xf $tfile
        rm $tfile
    done

    echo "Extracting validation files..."
    tar -xf $val_tar -C $work_dir
}

resize_imgs() { echo "Places2 already scaled.  Not resizing"; }

# Untar the top level files
echo "Extracting metadata files to " $i1kmetadir
unzip $meta_src -d $work_dir

resize_imgs

# places2 data is already resized for shortest dim <= 512
echo "Creating macrobatch lists"
cat $train_meta | shuf | awk -vF=$train_dir '{$0=F"/"$0}' | split -d -l $batch_size - $macro_dir/train_list
# We use 4010 instead of 5000 to get even validation batches
cat $val_meta | sort | awk -vF=$val_dir '{$0=F"/"$0}' | split -d -l 4010 - $macro_dir/val_list

