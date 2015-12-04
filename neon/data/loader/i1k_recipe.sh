work_dir=$1
src_dir=$2
batch_size=5000

train_tar=$src_dir/ILSVRC2012_img_train.tar
val_tar=$src_dir/ILSVRC2012_img_val.tar
meta_src=$src_dir/ILSVRC2012_devkit_t12.tar.gz
synset_map=$work_dir/i1k_synset.map

# Make sure the input files exist
src_files=($train_tar $val_tar $meta_src $synset_map)
for f in ${src_files[*]}; do
    if [[ -e $f ]]; then
        echo "Required source file $f not found"
        exit 1
    fi
done

train_dir=$work_dir/train
val_dir=$work_dir/validation
macro_dir=$work_dir/batchout
macro_writer=./img_macrowriter

if [[ -x $macro_writer ]]
then
    echo "File '$macro_writer' is executable"
else
    echo "File '$macro_writer' is not executable or found"
    exit 1
fi

# Untar the top level files
if ! [ -e $work_dir ]; then
    mkdir -p $work_dir
    if [ $? -ne 0 ]; then
        echo "Unable to create" $work_dir "exiting ..."
        exit 1
    fi
else
    if ! [ -w $work_dir ]; then
        echo $work_dir "not writeable..."
        exit 1
    fi
fi

echo "Extracting metadata files"
i1kmetadir=$work_dir/`basename ${meta_src%%.tar.gz}`
tar -xzf $meta_src -C $work_dir
val_labels=$i1kmetadir/ILSVRC2012_validation_ground_truth.txt

echo "Extracting training files..."
mkdir -p $train_dir
for i in $(tar -tf $train_tar); do
    tfile=$train_dir/$i
    subdir=${tfile%%.tar}
    #Extract the category tar
    echo $train_tar $tfile $subdir
    tar -xf $train_tar -C $train_dir $i

    #Untar it to a subdirectory
    mkdir -p $subdir
    tar -xf $tfile -C $subdir

    #Now delete the subdir tar
    rm $tfile
done


echo "Extracting validation files..."
mkdir -p $val_dir
tar -xf $val_tar -C $val_dir

# Optionally do the resizing in place using graphicsmagick (have to repeat for validation)
# for tdir in $(\ls -1 $train_dir); do
#     echo $tdir;
#     time gm mogrify -filter Triangle \
#     -define filter:support=2 \
#     -thumbnail "512x512^>" \
#     -unsharp 0.25x0.25+0.065 \
#     -quality 90 \
#     -interlace none \
#     -define jpeg:fancy-upsampling=off $tdir/*.JPEG
# done

echo "========================"
echo "Creating macrobatch lists"
echo "========================"

mkdir -p $macro_dir

# Make the shuffled lists for creating training macrobatches
n=0
for i in $(zcat $synset_map); do
    find $train_dir/$i -name "*.JPEG" | awk -vLL=$n '{print $0, LL}';
    n=$((n+1));
done | \
shuf | \
split -d -l $batch_size - $macro_dir/train_list

nclass=$n

# Make the lists for creating validation macrobatches
find $val_dir -name "*.JPEG" | \
    sort | paste - $val_labels | \
    awk '{print $1, $2-1}' | \
    split -d -l $batch_size - $macro_dir/val_list


echo "========================"
echo "Writing macrobatches"
echo "========================"

batchidx=0
for list in $(\ls -1 $macro_dir/{train_list*,val_list*}); do
    img_macrowriter $list $macro_dir/macrobatch_$batchidx
    ((batchidx++))
done


echo "========================"
echo "Writing metafile"
echo "========================"

train_nrec=`cat $macro_dir/train_list* | wc -l`
val_nrec=`cat $macro_dir/val_list* | wc -l`
train_start=0
ntrain=$(( (train_nrec + batch_size - 1)/batch_size))
val_start=$ntrain
nval=$(( (val_nrec + batch_size - 1)/batch_size))
nbatch=$((nval + ntrain))

macro_meta=$macro_dir/macrobatch_meta
echo "train_nrec  $train_nrec"   >  $macro_meta
echo "train_start $train_start"  >> $macro_meta
echo "ntrain      $ntrain"       >> $macro_meta
echo "val_nrec    $val_nrec"     >> $macro_meta
echo "val_start   $val_start"    >> $macro_meta
echo "nval        $nval"         >> $macro_meta
echo "nclass      $nclass"       >> $macro_meta
echo "R_mean      104.412277"    >> $macro_meta
echo "G_mean      119.213318"    >> $macro_meta
echo "B_mean      126.806091"    >> $macro_meta

# Now we need to get the largest individual datasize across all macrobatches
maxval=`for i in $(seq $nbatch); do
    xxd -p -s 24 -l 4 $macro_dir/macrobatch_$i |
    awk '{for (i=7; i>0; i-=2) s = (256 * s) + strtonum("0x"substr($1,i,2))} END {print s}'
done | sort -n -r | head -1`

echo "item_max_size" $maxval >> $macro_meta

