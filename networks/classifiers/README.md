<img width="30%" src="https://upload.wikimedia.org/wikipedia/commons/4/46/Logo-KALRAY.png"></a>

## List of Classifiers Neural networks
This repository gives access to following classifiers Neural Networks main architecture:
* DenseNet
* EfficientNet
* Inception
* MobileNet
* NasNet
* Resnet
* RegNet
* SqueezeNet
* VGG

All models have been trained on:
**ImageNet Large Scale Visual Recognition Challenge 2012 ([ILSVRC2012](https://www.image-net.org/challenges/LSVRC/2012/))** dataset</br>

Please find the complete list here: </br>
![ModelsPass](https://img.shields.io/badge/ACE5.4-28%20pass-g)
![Models](https://img.shields.io/badge/Total%20model-28-blue)

<!-- START AUTOMATED TABLE -->
| NAME                                         | FLOPs    | Params   | accTop1 | accTop5 | Framework   | Dataset     | ACE status                                          |
|:---------------------------------------------|:---------|:---------|:--------|:--------|:------------|-------------|:----------------------------------------------------|
| [alexNet](./alexnet)                         | 1.335 G  | 60.9 M   | 56.52 % | 79.06 % | onnx, qonnx | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g) |
| [denseNet-121](./densenet-121)               | 5.718 G  | 8.04 M   | 74.43 % | 91.97 % | onnx, qonnx | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g) |
| [denseNet-169](./densenet-169)               | 6.777 G  | 14.27 M  | 75.6 %  | 92.81 % | onnx, qonnx | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g) |
| [efficientNet-B0](./efficientnet-b0)         | 1.004 G  | 5.26 M   | 77.69 % | 93.53 % | onnx, qonnx | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g) |
| [efficientNet-B4](./efficientnet-b4)         | 11.727 G | 16.83 M  | 83.38 % | 96.59 % | onnx, qonnx | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g) |
| [efficientNetLite-B4](./efficientnetlite-b4) | 2.785 G  | 12.96 M  | 80.4 %  | -       | onnx, qonnx | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g) |
| [googleNet](./googlenet)                     | 3.014 G  | 6.62 M   | 69.8 %  | 89.5 %  | onnx, qonnx | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g) |
| [inception-resnetv2](./inception-resnetv2)   | 13.27 G  | 55.9 M   | 80.3 %  | 95.3 %  | onnx        | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g) |
| [inception-V3](./inception-v3)               | 11.42 G  | 27.16 M  | 77.2 %  | 93.4 %  | onnx        | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g) |
| [mobileNet-V1](./mobilenet-v1)               | 1.124 G  | 4.16 M   | 70.9 %  | 89.9 %  | onnx        | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g) |
| [mobileNet-V2](./mobilenet-v2)               | 0.893 G  | 3.54 M   | 71.88 % | 90.29 % | onnx, qonnx | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g) |
| [mobileNet-V3-large](./mobilenet-v3-large)   | 0.465 G  | 5.47 M   | 74.04 % | 91.34 % | onnx, qonnx | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g) |
| [nasnet](./nasnet)                           | 0.650 G  | 4.36 M   | 73.45 % | 91.51 % | onnx, qonnx | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g) |
| [regNet-x-1.6g](./regnet-x-1.6g)             | 3.240 G  | 9.17 M   | 77.04 % | 93.44 % | onnx, qonnx | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g) |
| [regNet-x-8.0g](./regnet-x-8.0g)             | 16.052 G | 39.53 M  | 79.34 % | 94.68 % | onnx, qonnx | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g) |
| [resnet18](./resnet18)                       | 3.642 G  | 11.70 M  | 69.75 % | 89.07 % | onnx, qonnx | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g) |
| [resnet34](./resnet34)                       | 7.348 G  | 21.81 M  | 73.31 % | 91.42 % | onnx, qonnx | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g) |
| [resnet50](./resnet50)                       | 7.770 G  | 25.63 M  | 74.93 % | 92.38 % | onnx, qonnx | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g) |
| [resnet50v1.5](./resnet50v1.5/)              | 8.234 G  | 25.53 M  | 76.13 % | 92.86 % | onnx, qonnx | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g) |
| [resnet50v2](./resnet50v2)                   | 8.209 G  | 25.5 M   | 75.81 % | 92.82 % | onnx, qonnx | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g) |
| [resnext50](./resnext50)                     | 8.436 G  | 25.0 M   | 77.62 % | 93.69 % | onnx, qonnx | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g) |
| [resnet101](./resnet101)                     | 15.221 G | 44.70 M  | 77.37 % | 93.54 % | onnx, qonnx | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g) |
| [resnet152](./resnet152)                     | 22.680 G | 60.4 M   | 78.31 % | 94.04 % | onnx, qonnx | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g) |
| [squeezeNet](./squeezenet)                   | 0.714 G  | 1.23 M   | 58.17 % | 80.62 % | onnx, qonnx | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g) |
| [vgg-16](./vgg-16)                           | 31.006 G | 138.36 M | 71.3 %  | 90.1 %  | onnx        | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g) |
| [vgg-19](./vgg-19)                           | 37.683 G | 12.85 M  | 71.3 %  | 90.0 %  | onnx        | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g) |
| [xception](./xception)                       | 9.07 G   | 22.9 M   | 79.0 %  | 94.5 %  | onnx        | ILSVRC 2012 | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g) |
<!-- END AUTOMATED TABLE -->

![Ext](https://img.shields.io/badge/ACE5.4-ext-yellow) means that you have to use Custom layer
feature to run the model (and would be integrated in next ACE release).

Please see [WIKI.md](../../WIKI.md) to use it.
