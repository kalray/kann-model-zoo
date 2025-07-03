<img width="30%" src="https://upload.wikimedia.org/wikipedia/commons/4/46/Logo-KALRAY.png"></a>

## List of Vision-Transformers

This repository gives access to following neural networks main architecture compatible with KaNN(TM):

* VitBase, MobileViT, SegFormer

## Important notes

* Neural networks are available on our **Hugging face plateform** 🤗 [HERE](https://huggingface.co/Kalray).
  Do not hesitate to check model cards for details of implementation, sources and/or license.

* Models have been trained on the following datasets: 
  + [ILSVRC2012](https://www.image-net.org/challenges/LSVRC/2012/) dataset for classifiers
  + [COCO2017](https://cocodataset.org/#detection-2017) for object detection
  + [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) for segmentation

* To generate a neural network with KaNN :
  + in FP16, refer to ONNX model (pointed by <model>_f16.yaml)
  + Please see [WIKI.md](../../WIKI.md) for instructions on how to use any of these models

 > [!TIP]
 > Example of use:
 > ```bash
 > # Generate
 > kann generate ./networks/vision-transformers/mobilevit/onnx/mobile_vit_f16.yaml -d mobile-vit
 > # wait ... then, run
 > kann run mobile-vit
 > # observe the output to consider the global and detailed performance
 > ```

## Neural Networks

The models are listed below, according:
  + The accuracy metrics (mAP50/95-mask or mIoU for segmentation)
  + MPPA performance at **batch 1** in :
    * Frame per second from device (MPPA frequency is 1.0GHz along inference)
    * Total number of cycles to compute 1 frame (averaged on the 10 first frames)
    * K Floating operations (FLOPS) per machine cycle (c)

*NB: MPPA Coolidge V2 processor default frequency is 1.0 GHz in ACE 6.1.0*

<!-- START AUTOMATED TABLE -->
| NAME                                                   |  FLOPs | Params | Top1-acc | mAP-50/95 |  mIoU  | Dtype |  Input  | Dataset    | 🤗 HF repo-id                                                   | FPS(K300) |
| :----------------------------------------------------- | -----: | -----: | :------: | :-------: | :----: | :---: | :-----: | :--------- | -------------------------------------------------------------- | --------: |
| [MobileViT](./mobilevit/onnx/mobile_vit_f16.yaml)      |  2.9 G |  5.6 M |  78.4 %  |     -     |  - %   | FP16  | 256x256 | ILSVRC2012 | [Kalray/mobile-vit](https://huggingface.co/Kalray/mobile-vit)  |   152.065 |
| [SegFormer-B0](./segformer/onnx/segformer_b0_f16.yaml) |  6.5 G | 3.32 M |    -     |     -     | 37.4 % | FP16  | 512x512 | ADE20K     | [Kalray/segformer-B0](https://huggingface.co/Kalray/segformer) |    43.334 |
| [ViT-Base](./vit-base-224/onnx/vit_base_f16.yaml)(*)   | 35.3 G | 86.5 M |  87.7 %  |     -     |  - %   | FP16  | 224x224 | ILSVRC2012 | [Kalray/vit-base](https://huggingface.co/Kalray/vit-base)      |    47.462 |
<!-- END AUTOMATED TABLE -->

(*) Custom layer is required to generate model (please see [CustomLayers](../../WIKI.md#custom-layers-for-extended-neural-network-support))
