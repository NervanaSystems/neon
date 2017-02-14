#Model

This is an implementation of [C3D][c3d] trained on the [UCF101][ucf101] dataset.

## Model script
The model training script train.py will train an action recogition model from scratch on the UCF101 dataset.  All of the scripts here have defaults set to use batch size of 32 inside the script.

## Dependencies
* [unrar-nonfree][unrar]: used for extracting videos from the provided rar file.
* [ffmpeg][ffmpeg]: used for transcoding, scaling, splitting videos in ingestion, as well as for overlaying category hypotheses on original videos during demo
* [gnu parallel][parallel]: used for speeding up the video ingestion process by distributing across all available cores on the local machine

## Instructions
### Preprocessing (Ingestion)
The ingestion script will scale the input videos down to uniform 171x128 framesizes, strip out audio, transcode them to use the MJPEG codec and break the input video into uniform length clips.  This sets the data up so that it is in a form that can be used for training in `neon`.

Preprocessing needs to be done for both training and test partitions of the dataset.  Grab the necessary files (`UCF101.rar` and `UCF101TrainTestSplits-RecognitionTask.zip` from [here][ucf101]), then run the `vid_ingest.sh` script which preprocesses the entire dataset and generates the manifest files that can be used by neon for training and testing.

For convenience, we use the local shell variable `V3D_DATA_PATH` to indicate where ingested files will be written to and read from.  The files `UCF101.rar` and `UCF101TrainTestSplits-RecognitionTask.zip` are assumed to be present in the directory indicated by the shell variable `INPUT_PATH`. 

```bash
INPUT_PATH=/usr/local/data
V3D_DATA_PATH=/usr/local/data
./examples/video-c3d/vid_ingest.sh $INPUT_PATH $V3D_DATA_PATH
```
The split files will be written into the output directory `$V3D_DATA_PATH/ucf-extracted` along with the necessary manifest files (list of records for training and validation).

### Training
To train, just run the following command, which will (by default) train for 18 epochs with a batch size of 32.  The model weights will be saved to `$V3D_DATA_PATH/UCF101-C3D.p`:

```bash
python examples/video-c3d/train.py
```


### Testing
The classification performance displayed during execution of this script will be on the individual subclips of each test video.  To get the aggregated predictions over each video, one can get final error rates with the testing script:
```
python examples/video-c3d/inference.py
```

### Inference Demo
Finally, we also provide a demo script which performs inference on a given input video.  After the model converges, the demo can be run which takes an input video (any format), does the necessary resizing, transcoding, and splitting, to perform inference on the subclips, then generates an output video with the five likeliest predictions overlaid on the video.
```
python examples/video-c3d/demo.py --input_video <in.avi> --output_video <out.avi>
```

### Trained weights
The weights file for the trained C3D model on the UCF101 action recognition training split 1 can be downloaded from AWS using the following link: [trained model weights][awswts].

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

   [c3d]: <http://arxiv.org/pdf/1412.0767v4.pdf>
   [ucf101]: <http://crcv.ucf.edu/data/UCF101.php>
   [ffmpeg]: <https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu>
   [parallel]: <https://savannah.gnu.org/projects/parallel/>
   [unrar]: <https://launchpad.net/ubuntu/+source/unrar-nonfree>
   [awswts]: <https://s3-us-west-1.amazonaws.com/nervana-modelzoo/video-c3d/UCF101-C3D.p>
