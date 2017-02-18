# DCGAN in tensorflow-slim

Implementation of [DCGAN](https://arxiv.org/abs/1511.06434) with [TensorFlow slim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim). Base codes and models are from [DCGAN in Tensorflow](https://github.com/carpedm20/DCGAN-tensorflow) made by Taehoon Kim.

At this time, this code only support **Flower** dataset, but maybe with some tweaks you can train/evaluate in other dataset.

I know there are lots of code of DCGAN, especially made by Taehoon Kim. However, this code implement DCGAN with the bleeding edges features of TensorFlow such as TF-Slim, `tf.train.Supervisor` and `TFRecords` etc.

## Requirements

- TensorFlow
- SciPy
- NumPy
- Only tested in Python 3.3

## Basic usages
To train model,
```shell
$ python3 dcgan/train.py
```
See `dcgan/train.py` and `dcgan/config.py` to modify arguments like `logdir` or `batch_size` etc (Someday I will provide argument parse codes).

Test (or sample) with trained model using below code.
```shell
$ python3 dcgan/sample.py
```
Note that checkpoint files must be located in `logdir` directory. See `dcgan/train.py` and `dcgan/config.py`. By default, `logdir` is set to `log/` directory.

## Results
Below examples are randomly selected from model with trained around 60k steps.

![img1](assets/img1.jpg)
![img2](assets/img2.jpg)

Some flowers looks fine, but most of images are bad. I believe that this is because Flower dataset has lots of noises, however on the other side, DCGAN's capacity doesn't enough to handle noisy images.

## Plot in TensorBoard
Oops.. I accidentally deleted TensorBoard log file. :(

## TODO

1. Provide pre-trained model
2. Apply `argparse` in `train.py` and `sample.py`
3. TODOs
