# Learning-Based Scalable Image Compression with Latent-Feature Reuse and Prediction

This repo holds the code for paper:

Y. Mei, L. Li, Z. Li and F. Li, "Learning-Based Scalable Image Compression with Latent-Feature Reuse and Prediction," in IEEE Transactions on Multimedia, doi: 10.1109/TMM.2021.3114548.

.<img src="https://github.com/xaMichelle/Learning-based-Scalable-Image-Coding/blob/master/framework.PNG" width="600" height="400" />

# Dependency

Tensorflow 1.15 
Tensorflow-compression 1.x ([Tensorflow-compression](https://tensorflow.github.io/compression/))

Or Tensorflow 2.x with Tensorflowp-compression 2.x)

# Train

## Quality Scalable

```
python train_quality.py
```

Please specify:
```
--train_glob, training dataset path
--checkpoint_dir, checkpoint output folder
```
You can also change the filter number and lambda of each layer (our example has four scalable layers):
```
--num_filters_B
--num_filters_e1
--num_filters_e2
--num_filters_e3
--lmbdaB
--lmbda1
--lmbda2
--lmbda3
```

## Spatial Scalable

```
python train_spaital.py
```

For spatial scalable training, please preprocess training dataset first. Our example has three layers. From base layer to last layer, training image sizes are HxW, 2Hx2W and 4Hx4W repectively. Downsample training images into different scales, then:

```
python create_tfrecords.py --train_tfrecords ./xxx.tfrecords --input_image ./your_4Hx4W_image_folder, --input_image_half ./your_2Hx2W_image_folder, --input_image_quater ./your_4Hx4W_image_folder
```

The arguments for train_spatial.py 
```
--train_tfrecords, training dataset path
--checkpoint_dir, checkpoint output folder
--num_filters_B
--num_filters_e1
--num_filters_e2
--lmbdaB
--lmbda1
--lmbda2
```

# Test

In eval_xxx.py files, `evaluate()` will compress the input image into several bitstreams like: stream_B.tfci and stream_e1.tfci, it will also evaluate the bpp and PSNR (or MS-SSIM) of each layer. `decompress()` can decode bitstreams to images.

## Quality Scalable

```
python eval_quality.py
--input_image ./kodak/kodim01.png
--output_folder ./output
--checkpoint_dir ./your_pretrained_models
--num_filters_B (shoud be consistent with your training settings)
--num_filters_e1 (shoud be consistent with your training settings)
--num_filters_e2 (shoud be consistent with your training settings)
--num_filters_e3 (shoud be consistent with training settings)
```

## Spatial Scalable

```
python eval_spatial.py
--input_image ./your test image folder/test.png
--input_image_half ./your test image folder/test-half.png
--input_image_quater ./your test image folder/test-quater.png
--output_folder ./output
--checkpoint_dir ./your_pretrained_models
--num_filters_B (shoud be consistent with your training settings)
--num_filters_e1 (shoud be consistent with your training settings)
--num_filters_e2 (shoud be consistent with your training settings)
```

## Citation

If you find our paper useful, please cite:
```
@ARTICLE{9547677,
  author={Mei, Yixin and Li, Li and Li, Zhu and Li, Fan},
  journal={IEEE Transactions on Multimedia}, 
  title={Learning-Based Scalable Image Compression with Latent-Feature Reuse and Prediction}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TMM.2021.3114548}}
```

## Contact

If you have any question or find any bug, please feel free to contact:

Yixin Mei @ XJTU
xamichelle@stu.xjtu.edu.cn
