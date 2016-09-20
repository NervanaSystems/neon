##Model
This example demonstrates how to classify music clips according to genre from between 10 different genres.

Download the example dataset from [here](http://marsyasweb.appspot.com/download/data_sets/).

First, use the included ingest script to unpack the data and create manifest files
```bash
python examples/music_genres/data.py --tar_file </path/to/genres.tar.gz> --out_dir </path/to/extract/files>
```

Once the files have been extracted and the manifest files have been created, call the training script, providing the manifest files.

Usage:
```bash
python examples/music_genres/train.py
```

### Performance
Upon training, the model should achieve better than 50 percent misclassification accuracy.
