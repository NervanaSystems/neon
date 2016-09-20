# Deep Residual Network on CIFAR10 Data

This example demonstrates how to train a deep residual network as first described by [He et. al.][msra1].  The example has been updated to use the "preactivation" structure explored in [the followup paper][msra2].

## Usage
The first time the training script is run, the data must be retrieved and preprocessed into a local directory (this step includes padding the images out so that random cropping can be done).  The directory where these ingested files are extracted to should be local to your machine, as the files will be read from during training.

Training a 1001 layer network can be accomplished with the following command:

```bash
CIFAR_DATA_PATH=</some/local/directory>

#Ingestion
python examples/cifar10_msra/data.py --out_dir $CIFAR_DATA_PATH

#Training
python examples/cifar10_msra/train.py \
        --no_progress_bar \
        --depth 111 \
        --save_path <save-path>
```
This setting should get to ~4.84% top-1 error. (Could be as low as 4.7)


## Citation
```
Deep Residual Learning for Image Recognition
http://arxiv.org/abs/1512.03385
```
```
Identity Mappings in Deep Residual Networks
http://arxiv.org/abs/1603.05027
```

   [msra1]: <http://arxiv.org/abs/1512.03385>
   [msra2]: <http://arxiv.org/abs/1603.05027>

