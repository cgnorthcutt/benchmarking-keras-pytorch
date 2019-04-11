# Benchmarking Keras and PyTorch Pre-Trained Models

Benchmarks for **every** pre-trained model in PyTorch and Keras-Tensorflow. Benchmarks are reproducible.

## Why this is helpful

Combining Keras and PyTorch benchmarks into a single framework lets researchers decide which platform is best for a given model. For example `resnet` architectures perform better in PyTorch and `inception` architectures perform better in Keras (see below). These benchmarks serve as a standard from which to start new projects or debug current implementations. 

For researchers exploring Keras and PyTorch models, these benchmarks serve as a standard from which to start new projects or debug current implementations. 

Many researchers struggle with reproducible accuracy benchmarks of pre-trained Keras (Tensorflow) models on ImageNet. Examples of issues are [here1](https://github.com/keras-team/keras/issues/10040), [here2](https://github.com/keras-team/keras/issues/10979), [here3](http://blog.datumbox.com/the-batch-normalization-layer-of-keras-is-broken/), [here4](https://github.com/keras-team/keras/issues/8672), and [here5](https://github.com/keras-team/keras/issues/7848). 

In Keras, the [published benchmarks](https://keras.io/applications/#documentation-for-individual-models) on [Keras Applications](https://keras.io/applications/) cannot be reproduced by exactly copying the associated code. In fact, the reported accuracies are usually higher than the actual accuries.

I dive slightly deeper into the reproducibility issues of Keras in the associated [blog post](http://l7.curtisnorthcutt.com/towards-reproducibility-benchmarking-keras-pytorch).

## Benchmark Results on ImageNet

The **actual** validation set accuracies on ImageNet for all Keras and PyTorch models (verified on macOS 10.11.6, Linux Debian 9, and Ubuntu 18.04).

| Platform    | Model             | Acc@1 | Acc@5 | Rank@1 | Rank@5 |
|-------------|-------------------|-------|-------|--------|--------|
| Keras 2.2.4 | nasnetlarge       | 80.83 | 95.27 | 1      | 1      |
| Keras 2.2.4 | inceptionresnetv2 | 78.93 | 94.45 | 2      | 2      |
| PyTorch 1.0 | resnet152         | 77.62 | 93.81 | 3      | 3      |
| Keras 2.2.4 | xception          | 77.18 | 93.49 | 4      | 4      |
| PyTorch 1.0 | densenet161       | 76.92 | 93.49 | 5      | 5      |
| PyTorch 1.0 | resnet101         | 76.64 | 93.30 | 6      | 6      |
| PyTorch 1.0 | densenet201       | 76.41 | 93.18 | 7      | 7      |
| Keras 2.2.4 | inceptionv3       | 76.02 | 92.90 | 8      | 8      |
| PyTorch 1.0 | densenet169       | 75.59 | 92.69 | 9      | 9      |
| PyTorch 1.0 | resnet50          | 75.06 | 92.48 | 10     | 10     |
| Keras 2.2.4 | densenet201       | 74.77 | 92.32 | 11     | 11     |
| PyTorch 1.0 | densenet121       | 74.07 | 91.94 | 12     | 12     |
| Keras 2.2.4 | densenet169       | 73.92 | 91.76 | 13     | 13     |
| PyTorch 1.0 | vgg19_bn          | 72.90 | 91.35 | 14     | 14     |
| PyTorch 1.0 | resnet34          | 72.34 | 90.84 | 15     | 16     |
| PyTorch 1.0 | vgg16_bn          | 72.29 | 91.01 | 16     | 15     |
| Keras 2.2.4 | densenet121       | 72.09 | 90.70 | 17     | 17     |
| Keras 2.2.4 | nasnetmobile      | 71.59 | 90.19 | 18     | 19     |
| PyTorch 1.0 | vgg19             | 71.19 | 90.40 | 19     | 18     |
| PyTorch 1.0 | vgg16             | 70.66 | 89.93 | 20     | 20     |
| Keras 2.2.4 | resnet50          | 70.35 | 89.55 | 21     | 22     |
| PyTorch 1.0 | vgg13_bn          | 70.12 | 89.56 | 22     | 21     |
| Keras 2.2.4 | mobilenetV2       | 69.98 | 89.49 | 23     | 23     |
| PyTorch 1.0 | vgg11_bn          | 69.36 | 89.06 | 24     | 24     |
| PyTorch 1.0 | inception_v3      | 69.25 | 88.69 | 25     | 25     |
| Keras 2.2.4 | mobilenet         | 69.02 | 88.48 | 26     | 27     |
| PyTorch 1.0 | vgg13             | 68.69 | 88.65 | 27     | 28     |
| PyTorch 1.0 | resnet18          | 68.37 | 88.56 | 28     | 26     |
| PyTorch 1.0 | vgg11             | 67.85 | 88.11 | 29     | 29     |
| Keras 2.2.4 | vgg19             | 65.58 | 86.54 | 30     | 30     |
| Keras 2.2.4 | vgg16             | 65.24 | 86.20 | 31     | 31     |
| PyTorch 1.0 | squeezenet1_0     | 56.49 | 79.06 | 32     | 33     |
| PyTorch 1.0 | squeezenet1_1     | 56.42 | 79.21 | 33     | 32     |
| PyTorch 1.0 | alexnet           | 54.50 | 77.64 | 34     | 34     |



## To Reproduce

### Get the ImageNet validation dataset 

* Download
  * Download the Imagenet 2012 Validation dataset from [http://image-net.org/download-images](http://image-net.org/download-images) or [other options](http://academictorrents.com). This dataset contains 50000 images

* Preprocess/Extract validation data
  * Once `ILSVRC2012_img_val.tar` is downloaded, run:
  ```bash
  # Credit to Soumith: https://github.com/soumith/imagenet-multiGPU.torch
  $ cd ../ && mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
  $ wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
  ```

### Reproduce in 10 seconds

The top 5 predictions for every example in the ImageNet validation set have been pre-computed for you [here for Keras models](https://github.com/cgnorthcutt/benchmarking-keras-pytorch/tree/master/keras_imagenet) and [here for PyTorch models](https://github.com/cgnorthcutt/benchmarking-keras-pytorch/tree/master/pytorch_imagenet). These are automatically used by the following code which takes a few seconds to run:

```bash
$ git clone https://github.com:cgnorthcutt/imagenet-benchmarking.git
$ cd benchmarking-keras-pytorch
$ python imagenet_benchmarking.py /path/to/imagenet_val_data
```

### Reproduce model outputs (hours)

You can also reproduce the inference-time output of each Keras and PyTorch model without using the pre-computed data. Inference for Keras takes a long time (5-10 hours) because I compute the forward pass on each example one at a time and avoid vectorized operations: this was the only approach I found would reliably reproduce the same accuracies. PyTorch is fairly quick (less than one hour). To reproduce:

```bash
$ git clone https://github.com:cgnorthcutt/imagenet-benchmarking.git
$ cd benchmarking-keras-pytorch
$ # Compute outputs of PyTorch models (1 hour)
$ ./imagenet_pytorch_get_predictions.py /path/to/imagenet_val_data
$ # Compute outputs of Keras models (5-10 hours)
$ ./imagenet_keras_get_predictions.py /path/to/imagenet_val_data
$ # View benchmark results
$ ./imagenet_benchmarking.py /path/to/imagenet_val_data
```

You can control *GPU usage*, *batch size*, *output storage directories*, and more. Run the files with the `-h` flag to see command line argument options.

#### Tips for Keras

One of the goals of this project is to help reconcile issues with reproducibility in Keras pre-trained models. The way I deal with these issues is three-fold. In Keras I 
1. avoid batches during inference.
2. run each example one at a time. This is silly slow, but yields a reproducible output for every model.
3. only run models in local functions or use `with` clauses to ensure no aspects of a previous model persist in memory when the next model is loaded.

## License

Copyright (c) 2019 Curtis Northcutt. Released under the MIT License. See [LICENSE](https://github.com/cgnorthcutt/imagenet_benchmarking/blob/master/LICENSE) for details.
