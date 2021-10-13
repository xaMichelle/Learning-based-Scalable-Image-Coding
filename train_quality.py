
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import sys
import os

from absl import app
from absl.flags import argparse_flags
import numpy as np
import tensorflow.compat.v1 as tf

import tensorflow_compression as tfc

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def read_png(filename):
    """Loads a PNG image file."""
    string = tf.read_file(filename)
    image = tf.image.decode_image(string, channels=3)
    image = tf.cast(image, tf.float32)
    image /= 255
    return image


def quantize_image(image):
    image = tf.round(image * 255)
    image = tf.saturate_cast(image, tf.uint8)
    return image


def write_png(filename, image):
    """Saves an image to a PNG file."""
    image = quantize_image(image)
    string = tf.image.encode_png(image)
    return tf.write_file(filename, string)


class AnalysisTransform(tf.keras.layers.Layer):
    """The analysis transform."""

    def __init__(self, num_filters, *args, **kwargs):
        self.num_filters = num_filters
        super(AnalysisTransform, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self._layers = [
            tfc.SignalConv2D(
                self.num_filters, (5, 5), name="layer_0", corr=True, strides_down=2,
                padding="same_zeros", use_bias=True,
                activation=tfc.GDN(name="gdn_0")),
            tfc.SignalConv2D(
                self.num_filters, (5, 5), name="layer_1", corr=True, strides_down=2,
                padding="same_zeros", use_bias=True,
                activation=tfc.GDN(name="gdn_1")),
            tfc.SignalConv2D(
                self.num_filters, (5, 5), name="layer_2", corr=True, strides_down=2,
                padding="same_zeros", use_bias=True,
                activation=tfc.GDN(name="gdn_2")),
            tfc.SignalConv2D(
                self.num_filters, (5, 5), name="layer_3", corr=True, strides_down=2,
                padding="same_zeros", use_bias=True,
                activation=None),
        ]
        super(AnalysisTransform, self).build(input_shape)

    def call(self, tensor):
        for layer in self._layers:
            tensor = layer(tensor)
        return tensor


class SynthesisTransform(tf.keras.layers.Layer):
    """The synthesis transform."""

    def __init__(self, num_filters, *args, **kwargs):
        self.num_filters = num_filters
        super(SynthesisTransform, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self._layers = [
            tfc.SignalConv2D(
                self.num_filters, (5, 5), name="layer_0", corr=False, strides_up=2,
                padding="same_zeros", use_bias=True,
                activation=tfc.GDN(name="igdn_0", inverse=True)),
            tfc.SignalConv2D(
                self.num_filters, (5, 5), name="layer_1", corr=False, strides_up=2,
                padding="same_zeros", use_bias=True,
                activation=tfc.GDN(name="igdn_1", inverse=True)),
            tfc.SignalConv2D(
                self.num_filters, (5, 5), name="layer_2", corr=False, strides_up=2,
                padding="same_zeros", use_bias=True,
                activation=tfc.GDN(name="igdn_2", inverse=True)),
            tfc.SignalConv2D(
                3, (5, 5), name="layer_3", corr=False, strides_up=2,
                padding="same_zeros", use_bias=True,
                activation=None),
        ]
        super(SynthesisTransform, self).build(input_shape)

    def call(self, tensor):
        for layer in self._layers:
            tensor = layer(tensor)
        return tensor

class PredictionTransform(tf.keras.layers.Layer):
    def __init__(self, num_filters_first, num_filters_last, *args, **kwargs):
        self.num_filters_first = num_filters_first
        self.num_filters_last = num_filters_last
        super(PredictionTransform, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self._layers = [
            tfc.SignalConv2D(
                self.num_filters_first, (5, 5), name="layer_0", corr=True, strides_down=2,
                padding="same_zeros", use_bias=True,
                activation=tfc.GDN(name="gdn_0")),
            tfc.SignalConv2D(
                self.num_filters_last, (5, 5), name="layer_1", corr=True, strides_down=2,
                padding="same_zeros", use_bias=True,
                activation=tfc.GDN(name="gdn_1")),
            tfc.SignalConv2D(
                self.num_filters_last, (5, 5), name="layer_2", corr=True, strides_down=1,
                padding="same_zeros", use_bias=True,
                activation=tfc.GDN(name="gdn_2")),
            tfc.SignalConv2D(
                self.num_filters_last, (5, 5), name="layer_3", corr=True, strides_down=1,
                padding="same_zeros", use_bias=True,
                activation=None),
            tfc.SignalConv2D(
                self.num_filters_last, (5, 5), name="layer_4", corr=True, strides_down=1,
                padding="same_zeros", use_bias=True,
                activation=tfc.GDN(name="gdn_3")),
            tfc.SignalConv2D(
                self.num_filters_last, (5, 5), name="layer_5", corr=True, strides_down=1,
                padding="same_zeros", use_bias=True,
                activation=None),
            tfc.SignalConv2D(
                self.num_filters_last, (5, 5), name="layer_6", corr=False, strides_up=2,
                padding="same_zeros", use_bias=True,
                activation=tfc.GDN(name="igdn_1", inverse=True)),
            tfc.SignalConv2D(
                self.num_filters_last, (5, 5), name="layer_7", corr=False, strides_up=2,
                padding="same_zeros", use_bias=True,
                activation=None),
        ]
        super(PredictionTransform, self).build(input_shape)

    def call(self, tensor):
        #for layer in self._layers:
        #    x = layer(tensor)
        x1 = self._layers[0](tensor)
        x2 = self._layers[1](x1)
        x3 = self._layers[2](x2)
        x4 = self._layers[3](x3)
        x5 = self._layers[4](x4 + x2)
        x6 = self._layers[5](x5)
        x7 = self._layers[6](x6 + x4 + x2)
        x8 = self._layers[7](x7)
        return x8


def train(args):
    if args.verbose:
        tf.logging.set_verbosity(tf.logging.INFO)

    with tf.device('/cpu:0'):
        train_files = glob.glob(args.train_glob) # find all matched files and their paths(*.png)
        if not train_files:
            raise RuntimeError("No training images found with glob '{}'".format(args.train_glob))
        train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
        train_dataset = train_dataset.shuffle(buffer_size=len(train_files)).repeat()
        train_dataset = train_dataset.map(read_png, num_parallel_calls=args.preprocess_threads) # read imgs into train_dataset
        train_dataset = train_dataset.map(lambda x: tf.random_crop(x, (args.patchsize, args.patchsize, 3)))
        train_dataset = train_dataset.batch(args.batchsize)
        train_dataset = train_dataset.prefetch(32)

    num_pixels = args.batchsize * args.patchsize ** 2

    x = train_dataset.make_one_shot_iterator().get_next()

    analysis_transform_B = AnalysisTransform(args.num_filters_B)
    analysis_transform_e1 = AnalysisTransform(args.num_filters_e1)
    analysis_transform_e2 = AnalysisTransform(args.num_filters_e2)
    analysis_transform_e3 = AnalysisTransform(args.num_filters_e3)
    synthesis_transform_B = SynthesisTransform(args.num_filters_B)
    synthesis_transform_e1 = SynthesisTransform(args.num_filters_e1)
    synthesis_transform_e2 = SynthesisTransform(args.num_filters_e2)
    synthesis_transform_e3 = SynthesisTransform(args.num_filters_e3)
    entropy_bottleneck_B = tfc.EntropyBottleneck()
    entropy_bottleneck_e1 = tfc.EntropyBottleneck()
    entropy_bottleneck_e2 = tfc.EntropyBottleneck()
    entropy_bottleneck_e3 = tfc.EntropyBottleneck()
    prediction_e1 = PredictionTransform(args.num_filters_B,
                                        args.num_filters_e1)
    prediction_e2 = PredictionTransform(args.num_filters_B + args.num_filters_e1,
                                        args.num_filters_e2)
    prediction_e3 = PredictionTransform(args.num_filters_B + args.num_filters_e1 + args.num_filters_e2,
                                        args.num_filters_e3)


    y_B = analysis_transform_B(x)
    y_B_tilde, y_B_likelihoods = entropy_bottleneck_B(y_B, training=True)

    y_e1 = analysis_transform_e1(x)
    y_e1_predict = prediction_e1(y_B_tilde)
    y_e1_res_tilde, y_e1_res_likelihoods = entropy_bottleneck_e1(tf.subtract(y_e1, y_e1_predict), training=True)
    y_e1_tilde = y_e1_predict + y_e1_res_tilde

    y_e2 = analysis_transform_e2(x)
    y_e2_predict = prediction_e2(tf.concat([y_B_tilde, y_e1_tilde], -1))
    y_e2_res_tilde, y_e2_res_likelihoods = entropy_bottleneck_e2(tf.subtract(y_e2, y_e2_predict), training=True)
    y_e2_tilde = y_e2_predict + y_e2_res_tilde

    y_e3 = analysis_transform_e3(x)
    y_e3_predict = prediction_e3(tf.concat([y_B_tilde, y_e1_tilde, y_e2_tilde], -1))
    y_e3_res_tilde, y_e3_res_likelihoods = entropy_bottleneck_e3(tf.subtract(y_e3, y_e3_predict), training=True)
    y_e3_tilde = y_e3_predict + y_e3_res_tilde

    x_B_hat = synthesis_transform_B(y_B_tilde)
    x_e1_hat = synthesis_transform_e1(tf.concat([y_B_tilde, y_e1_tilde], -1))
    x_e2_hat = synthesis_transform_e2(tf.concat([y_B_tilde, y_e1_tilde, y_e2_tilde], -1))
    x_e3_hat = synthesis_transform_e3(tf.concat([y_B_tilde, y_e1_tilde, y_e2_tilde, y_e3_tilde], -1))

    train_bpp_B = (tf.reduce_sum(tf.log(y_B_likelihoods))) / (-np.log(2) * num_pixels)
    train_bpp_e1 = (tf.reduce_sum(tf.log(y_e1_res_likelihoods))) / (-np.log(2) * num_pixels) + train_bpp_B
    train_bpp_e2 = (tf.reduce_sum(tf.log(y_e2_res_likelihoods))) / (-np.log(2) * num_pixels) + train_bpp_e1
    train_bpp_e3 = (tf.reduce_sum(tf.log(y_e3_res_likelihoods))) / (-np.log(2) * num_pixels) + train_bpp_e2

    train_mse_B = tf.reduce_mean(tf.squared_difference(x, x_B_hat))
    train_mse_e1 = tf.reduce_mean(tf.squared_difference(x, x_e1_hat))
    train_mse_e2 = tf.reduce_mean(tf.squared_difference(x, x_e2_hat))
    train_mse_e3 = tf.reduce_mean(tf.squared_difference(x, x_e3_hat))

    train_mse_B *= 255 ** 2
    train_mse_e1 *= 255 ** 2
    train_mse_e2 *= 255 ** 2
    train_mse_e3 *= 255 ** 2

    pred_mse_e1 = tf.reduce_mean(tf.squared_difference(y_e1, y_e1_predict))
    pred_mse_e2 = tf.reduce_mean(tf.squared_difference(y_e2, y_e2_predict))
    pred_mse_e3 = tf.reduce_mean(tf.squared_difference(y_e3, y_e3_predict))

    pred_mse_e1 = 0.1 * pred_mse_e1 * 255 ** 2
    pred_mse_e2 = 0.1 * pred_mse_e2 * 255 ** 2
    pred_mse_e3 = 0.1 * pred_mse_e3 * 255 ** 2

    train_loss_B = args.lmbdaB * train_bpp_B + train_mse_B
    train_loss_e1 = args.lmbda1 * train_bpp_e1 + train_mse_e1
    train_loss_e2 = args.lmbda2 * train_bpp_e2 + train_mse_e2
    train_loss_e3 = args.lmbda3 * train_bpp_e3 + train_mse_e3

    train_loss = train_loss_B + train_loss_e1 + train_loss_e2 + train_loss_e3
    #auxiliary_loss = entropy_bottleneck_B.losses[0] + entropy_bottleneck_e1.losses[0] + entropy_bottleneck_e2.losses[0] +
    #    entropy_bottleneck_e3.losses[0] + entropy_bottleneck_e4.losses[0]

    step = tf.train.create_global_step()
    main_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    main_step = main_optimizer.minimize(train_loss, global_step=step)

    aux_optimizer_B = tf.train.AdamOptimizer(learning_rate=1e-3)
    aux_step_B = aux_optimizer_B.minimize(entropy_bottleneck_B.losses[0])
    aux_optimizer_e1 = tf.train.AdamOptimizer(learning_rate=1e-3)
    aux_step_e1 = aux_optimizer_e1.minimize(entropy_bottleneck_e1.losses[0])
    aux_optimizer_e2 = tf.train.AdamOptimizer(learning_rate=1e-3)
    aux_step_e2 = aux_optimizer_e2.minimize(entropy_bottleneck_e2.losses[0])
    aux_optimizer_e3 = tf.train.AdamOptimizer(learning_rate=1e-3)
    aux_step_e3 = aux_optimizer_e3.minimize(entropy_bottleneck_e3.losses[0])

    train_op = tf.group(main_step, aux_step_B, entropy_bottleneck_B.updates[0], aux_step_e1, entropy_bottleneck_e1.updates[0],
                        aux_step_e2, entropy_bottleneck_e2.updates[0], aux_step_e3, entropy_bottleneck_e3.updates[0])

    tf.summary.scalar("loss", train_loss)
    tf.summary.scalar("lossB", train_loss_B)
    tf.summary.scalar("loss1", train_loss_e1)
    tf.summary.scalar("loss2", train_loss_e2)
    tf.summary.scalar("loss3", train_loss_e3)
    tf.summary.scalar("mseB", train_mse_B)
    tf.summary.scalar("mse1", train_mse_e1)
    tf.summary.scalar("mse2", train_mse_e2)
    tf.summary.scalar("mse3", train_mse_e3)
    tf.summary.scalar("bppB", train_bpp_B)
    tf.summary.scalar("bpp1", train_bpp_e1-train_bpp_B)
    tf.summary.scalar("bpp2", train_bpp_e2-train_bpp_e1)
    tf.summary.scalar("bpp3", train_bpp_e3-train_bpp_e2)


    tf.summary.image("original", quantize_image(x))
    tf.summary.image("reconstrucionB", quantize_image(x_B_hat))
    tf.summary.image("reconstrucion1", quantize_image(x_e1_hat))
    tf.summary.image("reconstrucion2", quantize_image(x_e2_hat))
    tf.summary.image("reconstrucion3", quantize_image(x_e3_hat))

    hooks = {
        tf.train.StopAtStepHook(last_step=args.last_step),
        tf.train.NanTensorHook(train_loss),
    }
    with tf.train.MonitoredTrainingSession(
        hooks=hooks, checkpoint_dir=args.checkpoint_dir,
        save_checkpoint_secs=300, save_summaries_secs=180) as sess:
        while not sess.should_stop():
            sess.run(train_op)

if __name__ == "__main__":
    parser = argparse_flags.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_glob', type=str, default='./clic/images_train/*.png', help="Training image path.")
    parser.add_argument('--checkpoint_dir', type=str, default='./models_quality/', help="Checkpoint folder")
    parser.add_argument("--num_filters_B", type=int, default=96, help="Number of base layer.")
    parser.add_argument("--num_filters_e1", type=int, default=120, help="Number of filters of enhancement layer e1.")
    parser.add_argument("--num_filters_e2", type=int, default=144, help="Number of filters of enhancement layer e2.")
    parser.add_argument("--num_filters_e3", type=int, default=196, help="Number of filters of enhancement layer e3.")
    parser.add_argument("--batchsize", type=int, default=8, help="Batch size.")
    parser.add_argument("--patchsize", type=int, default=256, help="Training image size.")
    parser.add_argument("--lmbdaB", type=int, default=250, help="RD trade-off of base layer.")
    parser.add_argument("--lmbdaB", type=int, default=100, help="RD trade-off of enhancement layer e1.")
    parser.add_argument("--lmbdaB", type=int, default=50, help="RD trade-off of enhancement layer e2.")
    parser.add_argument("--lmbdaB", type=int, default=25, help="RD trade-off of enhancement layer e3.")
    parser.add_argument("--last_step", type=int, default=20000, help="Training iterations.")
    parser.add_argument("--preprocess_threads", type=int, default=4, help="Parallel preprocess threads.")
    parser.add_argument("--verbose", type=bool, default=True, help="set_verbosity.")
    
    args = parser.parse_args()
    
    train(args)


