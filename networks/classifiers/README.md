<img width="30%" src="https://upload.wikimedia.org/wikipedia/commons/4/46/Logo-KALRAY.png"></a>

## List of Classification Neural Networks

This repository gives access to following classification neural networks by architecture:
* DenseNet, EfficientNet, Inception, MobileNet, NasNet, Resnet, RegNet, SqueezeNet, VGG

## Important notes

* Neural networks are available on our **Hugging face plateform** 🤗 [HERE](https://huggingface.co/Kalray).
  Do not hesitate to check model card for details of implementation, sources or license.

* All models have been trained on **ImageNet Large Scale Visual Recognition Challenge 2012**
  [ILSVRC2012](https://www.image-net.org/challenges/LSVRC/2012/) dataset

* To generate a neural network with KaNN :
  + in FP16, please refer to ONNX model (pointed by network_f16.yaml)
  + in INT8/FP16, please use QDQ-model (pointed by network_i8.yaml)
  + Please refer to [WIKI.md](../../WIKI.md) for instructions on how to use any of these models

## Neural Networks

The models are listed below, according:
  + the accuracy metrics (TopK accuracy)
  + MPPA performance at batch 1 in :
    * Frame per seconds (according that MPPA frequency is 1.0GHz along inference) from MPPA device
    * Total number of cycles to compute 1 frames (averaged on 10 first frames) in mega cycles (Mc)
    * K Floating operations (FLOPS) per machine cycles (c)

*NB: MPPA Coolidge V2 processor default frequency is 1.0 GHz*

<!-- START AUTOMATED TABLE -->
| NAME                                                                     |  FLOPs   |  Params  | accTop1 | accTop5 | Framework |  Input  | 🤗 HF repo-id                                                          | FPS(MPPA) | TOTAL(Mc) | KFLOPS/c |
| :----------------------------------------------------------------------- | :------: | :------: | :-----: | :-----: | :-------- | :-----: | :------------------------------------------------------------------------------ | --------: | --------: | -------: |
| [alexNet Q-INT8](./alexnet/onnx/network_i8.yaml)                         | 1.335 G  |  60.9 M  | 56.5 %  | 79.0 %  | QDQ-ONNX  | 224x224 | [Kalray/alexnet](https://huggingface.co/Kalray/alexnet)                         |         - |         - |        - |
| [alexNet F16](./alexnet/onnx/network_f16.yaml)                           | 1.335 G  |  60.9 M  | 56.52 % | 79.06 % | ONNX      | 224x224 | [Kalray/alexnet](huggingface.co/Kalray/alexnet)                                 |     234.2 |      4.27 |    1.007 |
| [denseNet-121 Q-INT8](./densenet-121/onnx/network_i8.yaml)               | 5.718 G  |  8.04 M  | 74.43 % | 91.9 %  | QDQ-ONNX  | 224x224 | [Kalray/densenet-121](https://huggingface.co/Kalray/densenet-121)               |     199.8 |      5.00 |    1.139 |
| [denseNet-121 F16](./densenet-121/onnx/network_f16.yaml)                 | 5.718 G  |  8.04 M  | 74.43 % | 91.97 % | ONNX      | 224x224 | [Kalray/densenet-121](https://huggingface.co/Kalray/densenet-121)               |     230.5 |      4.35 |    1.317 |
| [denseNet-169 Q-INT8](./densenet-169/onnx/network_i8.yaml)               | 6.777 G  | 14.27 M  | 75.6 %  | 92.8 %  | QDQ-ONNX  | 224x224 | [Kalray/densenet-169](https://huggingface.co/Kalray/densenet-169)               |     154.9 |      6.45 |    1.048 |
| [denseNet-169 F16](./densenet-169/onnx/network_f16.yaml)                 | 6.777 G  | 14.27 M  | 75.6 %  | 92.81 % | ONNX      | 224x224 | [Kalray/densenet-169](https://huggingface.co/Kalray/densenet-169)               |     172.4 |      5.78 |    1.168 |
| [efficientNet-B0 Q-INT8](./efficientnet-b0/onnx/network_i8.yaml)         | 1.004 G  |  5.26 M  | 77.6 %  | 93.5 %  | QDQ-ONNX  | 224x224 | [Kalray/efficientnet-b0](https://huggingface.co/Kalray/efficientnet-b0)         |     135.6 |      7.38 |    0.131 |
| [efficientNet-B0 F16](./efficientnet-b0/onnx/network_f16.yaml)           | 1.004 G  |  5.26 M  | 77.69 % | 93.53 % | ONNX      | 224x224 | [Kalray/efficientnet-b0](https://huggingface.co/Kalray/efficientnet-b0)         |     145.7 |      6.87 |    0.140 |
| [efficientNet-B4 Q-INT8](./efficientnet-b4/onnx/network_i8.yaml)         | 11.727 G | 16.83 M  | 83.3 %  | 96.5 %  | QDQ-ONNX  | 224x224 | [Kalray/efficientnet-b4](https://huggingface.co/Kalray/efficientnet-b4)         |      25.4 |     39.29 |    0.298 |
| [efficientNet-B4 F16](./efficientnet-b4/onnx/network_f16.yaml)           | 11.727 G | 16.83 M  | 83.38 % | 96.59 % | ONNX      | 224x224 | [Kalray/efficientnet-b4](https://huggingface.co/Kalray/efficientnet-b4)         |      25.4 |     39.33 |    0.298 |
| [efficientNetLite-B4 Q-INT8](./efficientnetlite-b4/onnx/network_i8.yaml) | 2.785 G  | 12.96 M  | 80.4 %  |    -    | QDQ-ONNX  | 224x224 | [Kalray/efficientNetLite-B4](https://huggingface.co/Kalray/efficientnetlite-b4) |   2.785 G |   12.96 M |   80.4 % | - | QDQ-ONNX | 224x224 | []() | 177.7 | 5.62 | 0.447 | ) | 177.7 | 5.62 | 0.447 |
| [efficientNetLite-B4 F16](./efficientnetlite-b4/onnx/network_f16.yaml)   | 2.785 G  | 12.96 M  | 80.4 %  |    -    | ONNX      | 224x224 | [Kalray/efficientNetLite-B4](https://huggingface.co/Kalray/efficientnetlite-b4) |   2.785 G |   12.96 M |   80.4 % | - | ONNX     | 224x224 | []() | 191.6 | 5.21 | 0.487 | ) | 191.6 | 5.21 | 0.487 |
| [googleNet Q-INT8](./googlenet/onnx/network_i8.yaml)                     | 3.014 G  |  6.62 M  | 69.8 %  | 89.5 %  | QDQ-ONNX  | 224x224 | [Kalray/googlenet](https://huggingface.co/Kalray/googlenet)                     |     760.8 |      1.31 |    2.280 |
| [googleNet F16](./googlenet/onnx/network_f16.yaml)                       | 3.014 G  |  6.62 M  | 69.8 %  | 89.5 %  | ONNX      | 224x224 | [Kalray/googlenet](https://huggingface.co/Kalray/googlenet)                     |     759.7 |      1.31 |    2.293 |
| [inception-resnetv2 F16](./inception-resnetv2/onnx/network_f16.yaml)     | 13.27 G  |  55.9 M  | 80.3 %  | 95.3 %  | ONNX      | 229x229 | [Kalray/inception-resnetv2](https://huggingface.co/Kalray/inception-resnetv2)   |      97.2 |     10.28 |    1.292 |
| [inception-V3 Q-INT8](./inception-v3/onnx/network_i8.yaml)               | 11.42 G  | 27.16 M  | 77.2 %  | 93.4 %  | QDQ-ONNX  | 299x299 | [Kalray/inception-v3](https://huggingface.co/Kalray/inception-v3)               |         - |         - |        - |
| [inception-V3 F16](./inception-v3/onnx/network_f16.yaml)                 | 11.42 G  | 27.16 M  | 77.2 %  | 93.4 %  | ONNX      | 299x299 | [Kalray/inception-v3](https://huggingface.co/Kalray/inception-v3)               |      29.4 |     33.98 |    0.339 |
| [mobileNet-V1 F16](./mobilenet-v1/onnx/network_f16.yaml)                 | 1.124 G  |  4.16 M  | 70.9 %  | 89.9 %  | ONNX      | 224x224 | [Kalray/mobilenet-v1](https://huggingface.co/Kalray/mobilenet-v1)               |     936.0 |      1.06 |    1.052 |
| [mobileNet-V2 Q-INT8](./mobilenet-v2/onnx/network_i8.yaml)               | 0.893 G  |  3.54 M  | 71.8 %  | 90.2 %  | QDQ-ONNX  | 224x224 | [Kalray/mobilenet-v2](https://huggingface.co/Kalray/mobilenet-v2)               |     632.9 |      1.58 |    0.355 |
| [mobileNet-V2 F16](./mobilenet-v2/onnx/network_f16.yaml)                 | 0.893 G  |  3.54 M  | 71.88 % | 90.29 % | ONNX      | 224x224 | [Kalray/mobilenet-v2](https://huggingface.co/Kalray/mobilenet-v2)               |     695.2 |      1.43 |    0.398 |
| [mobileNet-V3-large Q-INT8](./mobilenet-v3-large/onnx/network_i8.yaml)   | 0.465 G  |  5.47 M  | 74.0 %  | 91.3 %  | QDQ-ONNX  | 224x224 | [Kalray/mobilenet-v3-large](https://huggingface.co/Kalray/mobilenet-v3-large)   |     306.1 |      3.26 |    0.124 |
| [mobileNet-V3-large F16](./mobilenet-v3-large/onnx/network_f16.yaml)     | 0.465 G  |  5.47 M  | 74.04 % | 91.34 % | ONNX      | 224x224 | [Kalray/mobilenet-v3-large](https://huggingface.co/Kalray/mobilenet-v3-large)   |     345.6 |      2.89 |    0.142 |
| [nasnet Q-INT8](./nasnet/onnx/network_i8.yaml)                           | 0.650 G  |  4.36 M  | 73.4 %  | 91.5 %  | QDQ-ONNX  | 224x224 | [Kalray/nasnet](https://huggingface.co/Kalray/nasnet)                           |         - |         - |        - |
| [nasnet F16](./nasnet/onnx/network_f16.yaml)                             | 0.650 G  |  4.36 M  | 73.45 % | 91.51 % | ONNX      | 224x224 | [Kalray/nasnet](https://huggingface.co/Kalray/nasnet)                           |         - |         - |        - |
| [regNet-x-1.6g Q-INT8](./regnet-x-1.6g/onnx/network_i8.yaml)             | 3.240 G  |  9.17 M  | 77.0 %  | 93.4 %  | QDQ-ONNX  | 224x224 | [Kalray/regnet-x-1.6g](https://huggingface.co/Kalray/regnet-x-1.6g)             |     578.7 |      1.73 |    1.856 |
| [regNet-x-1.6g F16](./regnet-x-1.6g/onnx/network_f16.yaml)               | 3.240 G  |  9.17 M  | 77.04 % | 93.44 % | ONNX      | 224x224 | [Kalray/regnet-x-1.6g](https://huggingface.co/Kalray/regnet-x-1.6g)             |     509.3 |      1.96 |    1.650 |
| [regNet-x-8.0g Q-INT8](./regnet-x-8.0g/onnx/network_i8.yaml)             | 16.052 G | 39.53 M  | 79.3 %  | 94.6 %  | QDQ-ONNX  | 224x224 | [Kalray/regnet-x-8.0g](https://huggingface.co/Kalray/regnet-x-8.0g)             |     297.4 |      3.37 |    4.757 |
| [regNet-x-8.0g F16](./regnet-x-8.0g/onnx/network_f16.yaml)               | 16.052 G | 39.53 M  | 79.34 % | 94.68 % | ONNX      | 224x224 | [Kalray/regnet-x-8.0g](https://huggingface.co/Kalray/regnet-x-8.0g)             |     180.5 |      5.52 |    2.902 |
| [resnet101 Q-INT8](./resnet101/onnx/network_i8.yaml)                     | 15.221 G | 44.70 M  | 77.3 %  | 93.5 %  | QDQ-ONNX  | 224x224 | [Kalray/resnet101](https://huggingface.co/Kalray/resnet101)                     |     247.7 |      4.03 |    3.753 |
| [resnet101 F16](./resnet101/onnx/network_f16.yaml)                       | 15.221 G | 44.70 M  | 77.37 % | 93.54 % | ONNX      | 224x224 | [Kalray/resnet101](https://huggingface.co/Kalray/resnet101)                     |     142.8 |      7.00 |    2.168 |
| [resnet152 Q-INT8](./resnet152/onnx/network_i8.yaml)                     | 22.680 G |  60.4 M  | 78.3 %  | 94.0 %  | QDQ-ONNX  | 224x224 | [Kalray/resnet152](https://huggingface.co/Kalray/resnet152)                     |     182.2 |      5.47 |    4.115 |
| [resnet152 F16](./resnet152/onnx/network_f16.yaml)                       | 22.680 G |  60.4 M  | 78.31 % | 94.04 % | ONNX      | 224x224 | [Kalray/resnet152](https://huggingface.co/Kalray/resnet152)                     |     108.6 |      9.19 |    2.458 |
| [resnet18 Q-INT8](./resnet18/onnx/network_i8.yaml)                       | 3.642 G  | 11.70 M  | 69.7 %  | 89.0 %  | QDQ-ONNX  | 224x224 | [Kalray/resnet18](https://huggingface.co/Kalray/resnet18)                       |     781.2 |      1.28 |    2.845 |
| [resnet18 F16](./resnet18/onnx/network_f16.yaml)                         | 3.642 G  | 11.70 M  | 69.75 % | 89.07 % | ONNX      | 224x224 | [Kalray/resnet18](https://huggingface.co/Kalray/resnet18)                       |     437.2 |      2.29 |    1.589 |
| [resnet34 Q-INT8](./resnet34/onnx/network_i8.yaml)                       | 7.348 G  | 21.81 M  | 73.3 %  | 91.4 %  | QDQ-ONNX  | 224x224 | [Kalray/resnet34](https://huggingface.co/Kalray/resnet34)                       |     452.7 |      2.21 |    3.318 |
| [resnet34 F16](./resnet34/onnx/network_f16.yaml)                         | 7.348 G  | 21.81 M  | 73.31 % | 91.42 % | ONNX      | 224x224 | [Kalray/resnet34](https://huggingface.co/Kalray/resnet34)                       |     254.8 |      3.97 |    1.870 |
| [resnet50 Q-INT8](./resnet50/onnx/network_i8.yaml)                       | 7.770 G  | 25.63 M  | 74.9 %  | 92.3 %  | QDQ-ONNX  | 224x224 | [Kalray/resnet50](https://huggingface.co/Kalray/resnet50)                       |     394.7 |      2.55 |    3.048 |
| [resnet50 F16](./resnet50/onnx/network_f16.yaml)                         | 7.770 G  | 25.63 M  | 74.93 % | 92.38 % | ONNX      | 224x224 | [Kalray/resnet50](https://huggingface.co/Kalray/resnet50)                       |     251.9 |      3.96 |    1.951 |
| [resnet50v1.5 Q-INT8](./resnet50v1.5/onnx/network_i8.yaml)               | 8.234 G  | 25.53 M  | 76.1 %  | 92.8 %  | QDQ-ONNX  | 224x224 | [Kalray/resnet50v1.5](https://huggingface.co/Kalray/resnet50v1.5)               |     412.3 |      2.40 |    3.374 |
| [resnet50v1.5 F16](./resnet50v1.5/onnx/network_f16.yaml)                 | 8.234 G  | 25.53 M  | 76.13 % | 92.86 % | ONNX      | 224x224 | [Kalray/resnet50v1.5](https://huggingface.co/Kalray/resnet50v1.5)               |     246.5 |      4.09 |    2.024 |
| [resnet50v2 F16](./resnet50v2/onnx/network_f16.yaml)                     | 8.209 G  |  25.5 M  | 75.81 % | 92.82 % | ONNX      | 224x224 | [Kalray/resnet50v2](https://huggingface.co/Kalray/resnet50v2)                   |     231.5 |      4.28 |    1.905 |
| [resnext50 Q-INT8](./resnext50/onnx/network_i8.yaml)                     | 8.436 G  |  25.0 M  | 77.6 %  | 93.6 %  | QDQ-ONNX  | 224x224 | [Kalray/resnext50](https://huggingface.co/Kalray/resnext50)                     |     315.0 |      3.17 |    2.667 |
| [resnext50 F16](./resnext50/onnx/network_f16.yaml)                       | 8.436 G  |  25.0 M  | 77.62 % | 93.69 % | ONNX      | 224x224 | [Kalray/resnext50](https://huggingface.co/Kalray/resnext50)                     |     240.2 |      4.16 |    2.045 |
| [squeezeNet Q-INT8](./squeezenet/onnx/network_i8.yaml)                   | 0.714 G  |  1.23 M  | 58.1 %  | 80.6 %  | QDQ-ONNX  | 224x224 | [Kalray/squeezenet](https://huggingface.co/Kalray/squeezenet)                   |    1169.2 |      0.85 |    0.829 |
| [squeezeNet F16](./squeezenet/onnx/network_f16.yaml)                     | 0.714 G  |  1.23 M  | 58.17 % | 80.62 % | ONNX      | 224x224 | [Kalray/squeezenet](https://huggingface.co/Kalray/squeezenet)                   |    1516.9 |      0.66 |    1.095 |
| [vgg-16 F16](./vgg-16/onnx/network_f16.yaml)                             | 31.006 G | 138.36 M | 71.3 %  | 90.1 %  | ONNX      | 224x224 | [Kalray/vgg-16](https://huggingface.co/Kalray/vgg-16)                           |      86.7 |     11.64 |    2.688 |
| [vgg-19 F16](./vgg-19/onnx/network_f16.yaml)                             | 37.683 G | 12.85 M  | 71.3 %  | 90.0 %  | ONNX      | 224x224 | [Kalray/vgg-19](https://huggingface.co/Kalray/vgg-19)                           |      77.7 |     12.79 |    3.057 |
| [xception F16](./xception/onnx/network_f16.yaml)                         |  9.07 G  |  22.9 M  | 79.0 %  | 94.5 %  | ONNX      | 229x229 | [Kalray/xception](https://huggingface.co/Kalray/xception)                       |     257.9 |      3.87 |    2.356 |
<!-- END AUTOMATED TABLE -->
