# Learning-Based Scalable Image Compression with Latent-Feature Reuse and Prediction

This repo holds the code for paper:

Y. Mei, L. Li, Z. Li and F. Li, "Learning-Based Scalable Image Compression with Latent-Feature Reuse and Prediction," in IEEE Transactions on Multimedia, doi: 10.1109/TMM.2021.3114548.

![Framework](https://github.com/xaMichelle/Learning-based-Scalable-Image-Coding/blob/master/framework.PNG)

# Dependency

Tensorflow 1.15 
Tensorflow-compression 1.x ([Tensorflow-compression](https://tensorflow.github.io/compression/))

Or Tensorflow 2.x with Tensorflowp-compression 2.x)

# Train

## Quality Scalable

Our example has four scalable layers:
'"python train_quality.py'"
Please specify:
'"
--train_glob, training dataset path
--checkpoint_dir, checkpoint output folder
"'
You can also change the filter number and lambda of each layer:
'"
--num_filters_B
--num_filters_e1
--num_filters_e2
--num_filters_e3
--lmbdaB
--lmbda1
--lmbda2
--lmbda3
'"

## Spatial Scalable

'<python train_spaital.py>'

For spatial scalable training, please preparing training dataset first. Our example (train_spatial.py) has three layers. From base layer to last layer, training image sizes are HxW, 2Hx2W and 4Hx4W repectively.

'"python create_tfrecords.py --train_tfrecords ./xxx.tfrecords --input_image ./your_4Hx4W_image_folder, --input_image_half ./your_2Hx2W_image_folder, --input_image_quater ./your_4Hx4W_image_folder"'

The augemnts for train_spatial.py 
'"
--train_tfrecords, training dataset path
--checkpoint_dir, checkpoint output folder
--num_filters_B
--num_filters_e1
--num_filters_e2
--lmbdaB
--lmbda1
--lmbda2
'"

# Test
