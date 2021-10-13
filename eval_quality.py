
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
from matplotlib import pyplot
import tensorflow_compression as tfc

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

def evaluate(args):

    tf.reset_default_graph()

    x = read_png(args.input_image)
    x = tf.expand_dims(x, 0)
    x.set_shape([1, None, None, 3])
    x_shape = tf.shape(x)

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
    y_B_tilde, y_B_likelihoods = entropy_bottleneck_B(y_B, training=False)

    y_e1 = analysis_transform_e1(x)
    y_e1_predict = prediction_e1(y_B_tilde)
    y_e1_res_tilde, y_e1_res_likelihoods = entropy_bottleneck_e1(tf.subtract(y_e1, y_e1_predict), training=False)
    y_e1_tilde = y_e1_predict + y_e1_res_tilde

    y_e2 = analysis_transform_e2(x)
    y_e2_predict = prediction_e2(tf.concat([y_B_tilde, y_e1_tilde], -1))
    y_e2_res_tilde, y_e2_res_likelihoods = entropy_bottleneck_e2(tf.subtract(y_e2, y_e2_predict), training=False)
    y_e2_tilde = y_e2_predict + y_e2_res_tilde

    y_e3 = analysis_transform_e3(x)
    y_e3_predict = prediction_e3(tf.concat([y_B_tilde, y_e1_tilde, y_e2_tilde], -1))
    y_e3_res_tilde, y_e3_res_likelihoods = entropy_bottleneck_e3(tf.subtract(y_e3, y_e3_predict), training=False)
    y_e3_tilde = y_e3_predict + y_e3_res_tilde

    string_B = entropy_bottleneck_B.compress(y_B)
    string_e1 = entropy_bottleneck_e1.compress(tf.subtract(y_e1, y_e1_predict))
    string_e2 = entropy_bottleneck_e2.compress(tf.subtract(y_e2, y_e2_predict))
    string_e3 = entropy_bottleneck_e3.compress(tf.subtract(y_e3, y_e3_predict))

    x_B_hat = synthesis_transform_B(y_B_tilde)
    x_e1_hat = synthesis_transform_e1(tf.concat([y_B_tilde, y_e1_tilde], -1))
    x_e2_hat = synthesis_transform_e2(tf.concat([y_B_tilde, y_e1_tilde, y_e2_tilde], -1))
    x_e3_hat = synthesis_transform_e3(tf.concat([y_B_tilde, y_e1_tilde, y_e2_tilde, y_e3_tilde], -1))

    x_B_hat = x_B_hat[:, :x_shape[1], :x_shape[2], :]
    x_e1_hat = x_e1_hat[:, :x_shape[1], :x_shape[2], :]
    x_e2_hat = x_e2_hat[:, :x_shape[1], :x_shape[2], :]
    x_e3_hat = x_e3_hat[:, :x_shape[1], :x_shape[2], :]

    num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), dtype=tf.float32)

    eval_bpp_B = tf.reduce_sum(tf.log(y_B_likelihoods)) / (-np.log(2) * num_pixels)
    eval_bpp_e1 = tf.reduce_sum(tf.log(y_e1_res_likelihoods)) / (-np.log(2) * num_pixels) + eval_bpp_B
    eval_bpp_e2 = tf.reduce_sum(tf.log(y_e2_res_likelihoods)) / (-np.log(2) * num_pixels) + eval_bpp_e1
    eval_bpp_e3 = tf.reduce_sum(tf.log(y_e3_res_likelihoods)) / (-np.log(2) * num_pixels) + eval_bpp_e2
    #eval_bpp = eval_bpp_B + eval_bpp_e1 + eval_bpp_e2 + eval_bpp_e3

    x *= 255
    x_B_hat = tf.clip_by_value(x_B_hat, 0, 1)
    x_B_hat = tf.round(x_B_hat * 255)
    x_e1_hat = tf.clip_by_value(x_e1_hat, 0, 1)
    x_e1_hat = tf.round(x_e1_hat * 255)
    x_e2_hat = tf.clip_by_value(x_e2_hat, 0, 1)
    x_e2_hat = tf.round(x_e2_hat * 255)
    x_e3_hat = tf.clip_by_value(x_e3_hat, 0, 1)
    x_e3_hat = tf.round(x_e3_hat * 255)

    mse_pred1 = tf.reduce_mean(tf.squared_difference(y_e1, y_e1_predict))
    mse_pred2 = tf.reduce_mean(tf.squared_difference(y_e2, y_e2_predict))
    mse_pred3 = tf.reduce_mean(tf.squared_difference(y_e3, y_e3_predict))

    mse_B = tf.reduce_mean(tf.squared_difference(x, x_B_hat))
    mse_e1 = tf.reduce_mean(tf.squared_difference(x, x_e1_hat))
    mse_e2 = tf.reduce_mean(tf.squared_difference(x, x_e2_hat))
    mse_e3 = tf.reduce_mean(tf.squared_difference(x, x_e3_hat))

    psnr_B = tf.squeeze(tf.image.psnr(x_B_hat, x, 255))
    psnr_e1 = tf.squeeze(tf.image.psnr(x_e1_hat, x, 255))
    psnr_e2 = tf.squeeze(tf.image.psnr(x_e2_hat, x, 255))
    psnr_e3 = tf.squeeze(tf.image.psnr(x_e3_hat, x, 255))

    msssim_B = tf.squeeze(tf.image.ssim_multiscale(x_B_hat, x, 255))
    msssim_e1 = tf.squeeze(tf.image.ssim_multiscale(x_e1_hat, x, 255))
    msssim_e2 = tf.squeeze(tf.image.ssim_multiscale(x_e2_hat, x, 255))
    msssim_e3 = tf.squeeze(tf.image.ssim_multiscale(x_e3_hat, x, 255))

    with tf.Session() as sess:
        latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
        tf.train.Saver().restore(sess, save_path=latest)
        tensors_B = [string_B, tf.shape(x)[1:-1], tf.shape(y_B)[1:-1]]
        arrays_B = sess.run(tensors_B)
        tensors_e1 = [string_e1, tf.shape(x)[1:-1], tf.shape(y_e1)[1:-1]]
        arrays_e1 = sess.run(tensors_e1)
        tensors_e2 = [string_e2, tf.shape(x)[1:-1], tf.shape(y_e2)[1:-1]]
        arrays_e2 = sess.run(tensors_e2)
        tensors_e3 = [string_e3, tf.shape(x)[1:-1], tf.shape(y_e3)[1:-1]]
        arrays_e3 = sess.run(tensors_e3)

        packed_B = tfc.PackedTensors()
        packed_B.pack(tensors_B, arrays_B)
        with open(args.output_folder + '/stream_B.tfci', "wb") as f:
            f.write(packed_B.string)

        packed_e1 = tfc.PackedTensors()
        packed_e1.pack(tensors_e1, arrays_e1)
        with open(args.output_folder + '/stream_e1.tfci', "wb") as f:
            f.write(packed_e1.string)

        packed_e2 = tfc.PackedTensors()
        packed_e2.pack(tensors_e2, arrays_e2)
        with open(args.output_folder + '/stream_e2.tfci', "wb") as f:
            f.write(packed_e2.string)

        packed_e3 = tfc.PackedTensors()
        packed_e3.pack(tensors_e3, arrays_e3)
        with open(args.output_folder + '/stream_e3.tfci', "wb") as f:
            f.write(packed_e3.string)


        eval_bpp_B, eval_bpp_e1, eval_bpp_e2, eval_bpp_e3= sess.run(
            [eval_bpp_B, eval_bpp_e1, eval_bpp_e2, eval_bpp_e3]
        )

        mse_B, mse_e1, mse_e2, mse_e3 = sess.run(
            [mse_B, mse_e1, mse_e2, mse_e3]
        )

        psnr_B, psnr_e1, psnr_e2, psnr_e3 = sess.run(
            [psnr_B, psnr_e1, psnr_e2, psnr_e3]
        )

        msssim_B, msssim_e1, msssim_e2, msssim_e3 = sess.run(
            [msssim_B, msssim_e1, msssim_e2, msssim_e3]
        )

        num_pixels = sess.run(num_pixels)

        bpp_B = len(packed_B.string) *8 / num_pixels
        bpp_e1 = len(packed_e1.string) *8 / num_pixels
        bpp_e2 = len(packed_e2.string) *8 / num_pixels
        bpp_e3 = len(packed_e3.string) *8 / num_pixels

        ebpp_list = [eval_bpp_B, eval_bpp_e1, eval_bpp_e2, eval_bpp_e3]
        ebpp_list = [round(i,4) for i in ebpp_list]
        bpp_list = [bpp_B, bpp_e1, bpp_e2, bpp_e3]
        bpp_list = [round(i,4) for i in bpp_list]
        mse_list = [mse_B, mse_e1, mse_e2, mse_e3]
        mse_list = [round(i,4) for i in mse_list]
        psnr_list = [psnr_B, psnr_e1, psnr_e2, psnr_e3]
        psnr_list = [round(i,4) for i in psnr_list]
        msssim_list = [msssim_B, msssim_e1, msssim_e2, msssim_e3]
        msssim_list = [round(i,4) for i in msssim_list]
        
        print('eval_bpp:')
        print(ebpp_list)
        print('actual_bpp:')
        print(bpp_list)
        print('psnr:')
        print(psnr_list)
        print('msssim:')
        print(msssim_list)
        


def decompress(args):

    string_B = tf.placeholder(tf.string, [1])
    string_e1 = tf.placeholder(tf.string, [1])
    string_e2 = tf.placeholder(tf.string, [1])
    string_e3= tf.placeholder(tf.string, [1])

    x_B_shape = tf.placeholder(tf.int32, [2])
    x_e1_shape = tf.placeholder(tf.int32, [2])
    x_e2_shape = tf.placeholder(tf.int32, [2])
    x_e3_shape = tf.placeholder(tf.int32, [2])
    y_B_shape = tf.placeholder(tf.int32, [2])
    y_e1_shape = tf.placeholder(tf.int32, [2])
    y_e2_shape = tf.placeholder(tf.int32, [2])
    y_e3_shape = tf.placeholder(tf.int32, [2])

    with open(args.output_folder + '/stream_B.tfci', "rb") as f:
        packed_B = tfc.PackedTensors(f.read())
    tensors_B = [string_B, x_B_shape, y_B_shape]
    arrays_B = packed_B.unpack(tensors_B)
    with open(args.output_folder + '/stream_e1.tfci', "rb") as f:
        packed_e1 = tfc.PackedTensors(f.read())
    tensors_e1 = [string_e1, x_e1_shape, y_e1_shape]
    arrays_e1 = packed_e1.unpack(tensors_e1)
    with open(args.output_folder + '/stream_e2.tfci', "rb") as f:
        packed_e2 = tfc.PackedTensors(f.read())
    tensors_e2 = [string_e2, x_e2_shape, y_e2_shape]
    arrays_e2 = packed_e2.unpack(tensors_e2)
    with open(args.output_folder + '/stream_e3.tfci', "rb") as f:
        packed_e3 = tfc.PackedTensors(f.read())
    tensors_e3 = [string_e3, x_e3_shape, y_e3_shape]
    arrays_e3 = packed_e3.unpack(tensors_e3)
    tensors = [string_B, x_B_shape, y_B_shape, string_e1, x_e1_shape, y_e1_shape, string_e2, x_e2_shape, y_e2_shape, string_e3, x_e3_shape, y_e3_shape]
    arrays = arrays_B + arrays_e1 + arrays_e2 + arrays_e3

    entropy_bottleneck_B = tfc.EntropyBottleneck(dtype=tf.float32)
    entropy_bottleneck_e1 = tfc.EntropyBottleneck(dtype=tf.float32)
    entropy_bottleneck_e2 = tfc.EntropyBottleneck(dtype=tf.float32)
    entropy_bottleneck_e3 = tfc.EntropyBottleneck(dtype=tf.float32)
    synthesis_transform_B = SynthesisTransform(args.num_filters_B)
    synthesis_transform_e1 = SynthesisTransform(args.num_filters_e1)
    synthesis_transform_e2 = SynthesisTransform(args.num_filters_e2)
    synthesis_transform_e3 = SynthesisTransform(args.num_filters_e3)
    prediction_e1 = PredictionTransform(args.num_filters_B,
                                        args.num_filters_e1)
    prediction_e2 = PredictionTransform(args.num_filters_B + args.num_filters_e1,
                                        args.num_filters_e2)
    prediction_e3 = PredictionTransform(args.num_filters_B + args.num_filters_e1 + args.num_filters_e2,
                                        args.num_filters_e3)

    y_B_shape = tf.concat([y_B_shape, [args.num_filters_B]], axis=0)
    y_B_hat = entropy_bottleneck_B.decompress(
        string_B, y_B_shape, channels=args.num_filters_B)
    x_B_hat = synthesis_transform_B(y_B_hat)

    y_e1_shape = tf.concat([y_e1_shape, [args.num_filters_e1]], axis=0)
    y_e1_res_hat = entropy_bottleneck_e1.decompress(
        string_e1, y_e1_shape, channels=args.num_filters_e1)
    y_e1_pred = prediction_e1(y_B_hat)
    y_e1_hat = y_e1_res_hat + y_e1_pred
    x_e1_hat = synthesis_transform_e1(tf.concat([y_B_hat, y_e1_hat], -1))

    y_e2_shape = tf.concat([y_e2_shape, [args.num_filters_e2]], axis=0)
    y_e2_res_hat = entropy_bottleneck_e2.decompress(
        string_e2, y_e2_shape, channels=args.num_filters_e2)
    y_e2_pred = prediction_e2(tf.concat([y_B_hat, y_e1_hat], -1))
    y_e2_hat = y_e2_res_hat + y_e2_pred
    x_e2_hat = synthesis_transform_e2(tf.concat([y_B_hat, y_e1_hat, y_e2_hat], -1))

    y_e3_shape = tf.concat([y_e3_shape, [args.num_filters_e3]], axis=0)
    y_e3_res_hat = entropy_bottleneck_e3.decompress(
        string_e3, y_e3_shape, channels=args.num_filters_e3)
    y_e3_pred = prediction_e3(tf.concat([y_B_hat, y_e1_hat, y_e2_hat], -1))
    y_e3_hat = y_e3_res_hat + y_e3_pred
    x_e3_hat = synthesis_transform_e3(tf.concat([y_B_hat, y_e1_hat, y_e2_hat, y_e3_hat], -1))

    x_B_hat = x_B_hat[0, :x_B_shape[0], :x_B_shape[1], :]
    x_e1_hat = x_e1_hat[0, :x_e1_shape[0], :x_e1_shape[1], :]
    x_e2_hat = x_e2_hat[0, :x_e2_shape[0], :x_e2_shape[1], :]
    x_e3_hat = x_e3_hat[0, :x_e3_shape[0], :x_e3_shape[1], :]

    op_B = write_png(args.output_folder + '/rec_B.png', x_B_hat)
    op_e1 = write_png(args.output_folder + '/rec_e1.png', x_e1_hat)
    op_e2 = write_png(args.output_folder + '/rec_e2.png', x_e2_hat)
    op_e3 = write_png(args.output_folder + '/rec_e3.png', x_e3_hat)

    with tf.Session() as sess:
        latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
        tf.train.Saver().restore(sess, save_path=latest)
        # sess.run(op_B, feed_dict=dict(zip(tensors_B, arrays_B)))
        # sess.run(op_e1, feed_dict=dict(zip(tensors_e1, arrays_e1)))
        # sess.run(op_e2, feed_dict=dict(zip(tensors_e2, arrays_e2)))
        # sess.run(op_e3, feed_dict=dict(zip(tensors_e3, arrays_e3)))
        # dict1 = dict(zip(tensors_B, arrays_B))
        # dict2 = dict(zip(tensors_e1, arrays_e1))
        # dict3 = dict(zip(tensors_e2, arrays_e2))
        # dict4 = dict(zip(tensors_e3, arrays_e3))
        # dict2.update(dict1)
        # dict3.update(dict2)
        # dict4.update(dict3)
        # sess.run(op_B, feed_dict=dict1)
        # x_B_hat = sess.run(x_B_hat, feed_dict=dict(zip(tensors, arrays)))
        sess.run([op_B, op_e1, op_e2, op_e3], feed_dict=dict(zip(tensors, arrays)))


if __name__ == "__main__":


    parser = argparse_flags.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_image', type=str, default='../sfp/kodak/kodim01.png', help="Kodak image path.")
    parser.add_argument('--output_folder', type=str, default='./output', help="Output path.")
    parser.add_argument('--checkpoint_dir', type=str, default='./train_sfp_sep_res12-120/', help="Checkpoint folder")
    parser.add_argument("--num_filters_B", type=int, default=96, help="Number of base layer.")
    parser.add_argument("--num_filters_e1", type=int, default=120, help="Number of filters of enhancement layer e1.")
    parser.add_argument("--num_filters_e2", type=int, default=144, help="Number of filters of enhancement layer e2.")
    parser.add_argument("--num_filters_e3", type=int, default=196, help="Number of filters of enhancement layer e3.")
    parser.add_argument("--lmbdaB", type=int, default=250, help="RD trade-off of base layer.")
    parser.add_argument("--lmbda1", type=int, default=100, help="RD trade-off of enhancement layer e1.")
    parser.add_argument("--lmbda2", type=int, default=50, help="RD trade-off of enhancement layer e2.")
    parser.add_argument("--lmbda3", type=int, default=25, help="RD trade-off of enhancement layer e3.")

    
    
    args = parser.parse_args()


    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    evaluate(args)
    decompress(args)


