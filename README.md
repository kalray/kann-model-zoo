# KaNN™ Model Zoo

<img width="25%" src="./utils/materials/kalray_logo.png"></a></br>

![ACE-6.0.0](https://img.shields.io/badge/MMPA--Coolidge2-ACE--6.0.0-g)
![KaNN-5.5.0](https://img.shields.io/badge/KaNN--5.5.0-red)
![Classification](https://img.shields.io/badge/Classification-27-blue)
![Object-Detection](https://img.shields.io/badge/Object--detection-32-blue)
![Segmentation](https://img.shields.io/badge/Segmentation-9-blue)
![A](https://img.shields.io/badge/HuggingFace%20🤗-orange)</br>

The KaNN™ Model Zoo repository offers a collection of neural network models **ready to compile & run** on Kalray's MPPA®
manycore processor. Coolidge V2, the 3rd and latest generation of our MPPA®, is a dedicated processor for **AI applications**.
KaNN™ Model Zoo complements the KaNN™ SDK, which streamlines model generation and optimizes **AI performance** on Kalray's processors.

<p align="center">
  We are pleased to announce that our models are available on our Kalray space</br>
  <a  href="https://huggingface.co/Kalray">
    <img width="25%" src="./utils/materials/Hugging_Face_logo.svg">
  </a></br>
</p>

## Contents

CNN models are grouped by three types of machine vision applications:
* [Classification](./networks/classifiers/README.md): DenseNet, EfficientNet, Inception, MobileNet, NasNet, ResNet, RegNet, SqueezeNet, VGG
* [Object Detection](./networks/object-detection/README.md): EfficientDet, RetinatNet, SSD, YOLO
* [Segmentation](./networks/segmentation/README.md): DeeplabV3+, Fully Convolution Network (FCN), U-Net, YOLO

The examples below illustrate the kind of predictions obtained for each application type:

| Classification <p> (e.g. SqueezeNet)                                      | Object Detection <p> (e.g. YOLOv8n)                                        | Segmentation <p> (e.g. Deeplabv3+)                                       |
| ------------------------------------------------------------------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| <img height="240" width="100%" src="./utils/materials/cat_class.jpg"></a> | <img height="240" width="100%" src="./utils/materials/cat_detect.jpg"></a> | <img height="240" width="100%" src="./utils/materials/cat_segm.jpg"></a> |

**images have been generated from this repository and KaNN™ SDK solution (ACE 5.4.0)*

## Kalray Neural Network (KaNN™) SDK

Kalray Neural Network (KaNN™) is a SDK included in the AccessCore Embedded (ACE™) compute offer to optimize AI inference on MPPA®. 
It is composed by:

* **KaNN™ generator** : A python wheel to parse, optimize and paralellize an intermediate representation of a neural
  network. Thanks to the runtime, it gives you then the opportunity to run the algorithm directly on the MPPA®
* **KaNN™ runtime** : Optimized libraries (in ASM/C/C++) to execute each operation node.

ACE™ 6.0.0 | KaNN™ 5.5.0 supports: ONNX framework.

## Important

* Neural networks are available on our **Hugging face plateform** 🤗 [HERE](https://huggingface.co/Kalray).
  Do not hesitate to check model card for details of implementation, sources or license.

* TensorFlow and TensorFlowLite is now deprecated from this ACE™ version (>=6.0.0). All TF networks of the KaNN™
  Model Zoo have been converted to ONNX format with [**tf2onnx**](https://github.com/onnx/tensorflow-onnx) tools.

* To generate a neural network compatible for Kalray processor (MPPA®):
  + in FP16, please refer to onnx model (pointed by network_f16.yaml configuration file)
  + in INT8/FP16, use QDQ-model (pointed by the network_i8.yaml configuration file)

* Interesting to run faster ? please contact our support to optimize your use case at support@kalrayinc.com

## WIKI notes

To quickly deploy a neural network on the MPPA®, a WIKI note is available [here](WIKI.md):

- [KaNN™ framework description](#kann-framework-description)
- [Prerequisites: SW environment \& configuration](#prerequisites-sw-environment--configuration)
- [How models are packaged](#how-models-are-packaged)
- [Generate a model to run on the MPPA®](#generate-a-model-to-run-on-the-mppa)
- [Evaluate the neural network inference on the MPPA®](#evaluate-the-neural-network-inference-on-the-mppa)
- [Run the neural network as a demo](#run-the-neural-network-as-a-demo)
- [Neural networks accuracy and associated metrics](#neural-networks-accuracy-and-associated-metrics)
- [Custom Layers for extended neural network supoort](#custom-layers-for-extended-neural-network-supoort)
- [Jupyter Notebooks](#jupyter-notebooks)

## Requirements

### Hardware requirements

#### Host machine:

* x86_64 CPU
* DDR RAM > 8 GB
* HDD disk > 32 GB
* PCIe >= Gen3, Gen4 x16 recommended

#### Acceleration cards:

MPPA Coolidge2 product brief is available [here](https://www.kalrayinc.com/wp-content/uploads/2023/10/Kalray_MPPA-Coolidge_flyer_2P_NoNda_EXTERNAL_9.08.pdf)

| KALRAY Products                                           | links                                                                     | TFLOPs (FP16) | TOPs (INT8) |
| :-------------------------------------------------------- | :------------------------------------------------------------------------ | :-----------: | :---------: |
| ![A](https://img.shields.io/badge/Coolidge2-Turbocard4-g) | [TC4](https://www.kalrayinc.com/products/kalray-processors/#turbocard4)   |      100      |     200     |
| ![A](https://img.shields.io/badge/Coolidge2-K300-blue)    | [K300](https://www.kalrayinc.com/products/kalray-processors/#k300-family) |      25       |     50      |

**data are provided for MPPA frequency @ 1.2GHz (scalable)*
***compute capabilities (FLOPs/OPs) are given for dense tensors*

### Software requirements

* ![U22](https://img.shields.io/badge/Ubuntu-22.04%20LTS-orange)
  ![Ker](https://img.shields.io/badge/Linux%20Kernel-5.15.0-red)
* ![ACE](https://img.shields.io/badge/Coolidge2-ACE--6.0.0-g)
  ![KaNN-5.5.0](https://img.shields.io/badge/KaNN--5.5.0-red)
* ![Python](https://img.shields.io/badge/Python-3.10-blue)
  ![Python](https://img.shields.io/badge/Python-3.11-blue)
