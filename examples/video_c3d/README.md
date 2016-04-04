##Model

This is an implementation of [C3D](http://arxiv.org/pdf/1412.0767v4.pdf) trained on the [UCF101](http://crcv.ucf.edu/data/UCF101.php) dataset.

### Model script
The model training script [train.py](https://gist.github.com/SNagappan/304446c6c2f7afe29629#file-video_c3d-py) is included below.

### Instructions
The first step is to preprocess the UCF101 dataset using preprocess.py which splits videos into smaller clips. Preprocessed videos need to be created for both the training and test splits.

```
python preprocess.py <video_dir> <data_split_file> <class_ind_file> <preprocessed_dir>
```

`<video_dir>` is the location of the raw video data, `<data_split_file>` is the name of the training or test split, and `<class_ind_file>` is a text file containing the mapping from class labels to indices. `<preprocessed_dir>` is the output directory of the preprocessed videos.

An example of running this program is:
```
python examples/video_c3d/preprocess.py --video_dir ~/data/UCF-101/ --data_split_file ~/data/ucfTrainTestlist/trainlist01.txt --class_ind_file ~/data/ucfTrainTestlist/classInd.txt --preprocessed_dir ~/data/ucf_preprocessed/train1

python examples/video_c3d/preprocess.py --video_dir ~/data/UCF-101/ --data_split_file ~/data/ucfTrainTestlist/testlist01.txt --class_ind_file ~/data/ucfTrainTestlist/classInd.txt --preprocessed_dir ~/data/ucf_preprocessed/test1
```

Once the preprocessed video directories are created for both the training and test splits, the model can be trained with the following:
```
python examples/video_c3d/train.py --data_dir <preprocessed_dir> --epochs 18 --save_path UCF101-C3D.p
```

### Trained weights
The weights file for the trained C3D model on the UCF101 action detection training split 1 can be downloaded from AWS using the following link: [trained model weights](https://s3-us-west-1.amazonaws.com/nervana-modelzoo/video_c3d/UCF101-C3D.p).

### Performance
The model achieves 45.6% clip accuracy on the action detection test split 1 when trained on training split 1.

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