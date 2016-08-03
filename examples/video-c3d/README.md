##Model

This is an implementation of [C3D](http://arxiv.org/pdf/1412.0767v4.pdf) trained on the [UCF101](http://crcv.ucf.edu/data/UCF101.php) dataset.

### Model script
The model training script train.py will train an action recogition model from scratch on the UCF101 dataset.

### Dependencies
The preprocessor and data loader require ffmpeg which can be installed by following instructions [here](https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu)

### Instructions
The first step is to preprocess the UCF101 dataset using preprocess.py which splits videos into smaller clips. Preprocessed videos need to be created for both the training and test splits.  Grab the necessary files (`UCF101.rar` and `UCF101TrainTestSplits-RecognitionTask`), unpack them, then run the following commands to preprocess the data.

```bash
# Where UCF101.rar was unpacked
UCF_DATA_DIR=/usr/local/data/UCF-101

# Where UCF101TrainTestSplits-RecognitionTask.zip was unpacked
UCF_META_DIR=/usr/local/data/ucfTrainTestlist

# Where the processed clips from each split are saved
OUT_DIR=/usr/local/data/UCF-ingested

# Split the videos into clips
vid_ingest.sh ${UCF_DATA_DIR} ${UCF_META_DIR}/trainlist01.txt $OUT_DIR/train
vid_ingest.sh ${UCF_DATA_DIR} ${UCF_META_DIR}/testlist01.txt $OUT_DIR/test

# Create manifest files
create_manifests.sh $OUT_DIR/train $OUT_DIR/test $OUT_DIR/labels ${UCF_META_DIR}/classInd.txt
```

Once the preprocessed video directories are created for both the training and test splits, the model can be trained with the following:
```
python examples/video-c3d/train.py --data_dir <preprocessed_dir> --batch_size 32 --epochs 18 --save_path UCF101-C3D.p
```
The data loader will automatically create index files mapping each video clip to its label. Alternatively, an index file can be provided to the data loader.

After the model converges, the demo can be run which predicts the most probable class for each clip and aggregates them into one output video which displays the class labels and their probabilities.
```
python examples/video-c3d/demo.py --data_dir <preprocessed_dir/split_dir> --model_weights UCF101-C3D.p --class_ind_file ~/data/ucfTrainTestlist/classInd.txt
```

### Trained weights
The weights file for the trained C3D model on the UCF101 action recognition training split 1 can be downloaded from AWS using the following link: [trained model weights](https://s3-us-west-1.amazonaws.com/nervana-modelzoo/video-c3d/UCF101-C3D.p).

### Performance
The model achieves 46.3% clip accuracy on the action recognition test split 1 when trained on training split 1.

## Citation
```
Learning Spatiotemporal Features with 3D Convolutional Networks
http://arxiv.org/pdf/1412.0767v4.pdf
```
```
http://vlg.cs.dartmouth.edu/c3d/
```
```
https://github.com/facebook/C3D
```
