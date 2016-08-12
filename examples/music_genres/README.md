##Model
This example demonstrates how to classify music clips according to genre from between 10 different genres.

Download the example dataset from [here](http://marsyasweb.appspot.com/download/data_sets/).

First, set an environment variable to a local directory for extracting the files and saving any cached outputs used during training.  All scripts in this training process require that `$MUSIC_DATA_PATH` has been set.

The training script will unpack the data, ingest it into a format that can be loaded into neon, and then begin training.  The ingestion component requires the `sox` audio processing tool, so After downloading the tar, point the script at the file

Usage:
```bash
sudo apt-get -y install sox              # to install sox tool

export MUSIC_DATA_PATH=/usr/local/data   # or your preferred local directory

python examples/music_genres/train.py -e 16 --tar_file </path/to/genres.tar.gz> 
```

The `--tar_file` option only needs to provided on the first run.  Subsequent runs will pick up the pre-ingested files.

### Performance
Upon training, the model should achieve better than 50 percent misclassification accuracy.
