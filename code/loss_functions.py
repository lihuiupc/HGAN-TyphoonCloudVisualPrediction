import tensorflow as tf
import numpy as np

from tfutils import log10
import constants as c
from math import isnan

def combined_loss(gen_frames, gt_frames, d_preds, last_frames, l_num, alpha, lam_adv, lam_lp, lam_gdl,lam_tv):

    batch_size = tf.shape(gen_frames[0])[0]  # variable batch size as a tensor
    loss = lam_lp * lp_loss(gen_frames, gt_frames, l_num)
    loss += lam_gdl * gdl_loss(gen_frames, gt_frames, alpha)
    loss -= lam_tv * tv_loss(gen_frames)
    if c.ADVERSARIAL: loss += lam_adv * adv_loss(d_preds, tf.ones([4*batch_size, 1]))##4*
    return loss

def bce_loss(preds, targets):

    return tf.squeeze(-1 * (tf.matmul(targets, log10(preds), transpose_a=True) +
                            tf.matmul(1 - targets, log10(1 - preds), transpose_a=True)))


def lp_loss(gen_frames, gt_frames, l_num):

    # calculate the loss 
    scale_losses = []
    for i in range(len(gen_frames)):
        scale_losses.append(tf.reduce_sum(tf.abs(gen_frames[i] - gt_frames[i])**l_num))

    # condense into one tensor and avg
    return tf.reduce_mean(tf.stack(scale_losses))


def gdl_loss(gen_frames, gt_frames, alpha):

    scale_losses = []
    for i in range(len(gen_frames)):
        # create filters [-1, 1] and [[1],[-1]] for diffing to the left and down respectively.
        pos = tf.constant(np.identity(3), dtype=tf.float32)
        neg = -1 * pos
        filter_x = tf.expand_dims(tf.stack([neg, pos]), 0)  # [-1, 1]
        filter_y = tf.stack([tf.expand_dims(pos, 0), tf.expand_dims(neg, 0)])  # [[1],[-1]]
        strides = [1, 1, 1, 1]  # stride of (1, 1)
        padding = 'SAME'

        gen_dx = tf.abs(tf.nn.conv2d(gen_frames[i], filter_x, strides, padding=padding))
        gen_dy = tf.abs(tf.nn.conv2d(gen_frames[i], filter_y, strides, padding=padding))
        gt_dx = tf.abs(tf.nn.conv2d(gt_frames[i], filter_x, strides, padding=padding))
        gt_dy = tf.abs(tf.nn.conv2d(gt_frames[i], filter_y, strides, padding=padding))

        grad_diff_x = tf.abs(gt_dx - gen_dx)
        grad_diff_y = tf.abs(gt_dy - gen_dy)

        scale_losses.append(tf.reduce_sum((grad_diff_x ** alpha + grad_diff_y ** alpha)))

    # condense into one tensor and avg
    return tf.reduce_mean(tf.stack(scale_losses))


def adv_loss(preds, labels):

    scale_losses = []
    for i in range(len(preds)):
        loss = bce_loss(preds[i], labels)
        scale_losses.append(loss)

    # condense into one tensor and avg
    return tf.reduce_mean(tf.stack(scale_losses))

def tv_loss(gen_frames):
    scale_losses = []
    for i in range(len(gen_frames)):
        pos = tf.constant(np.identity(3), dtype=tf.float32)
        neg = -1 * pos
        filter_x = tf.expand_dims(tf.stack([neg, pos]), 0)  # [-1, 1]
        filter_y = tf.stack([tf.expand_dims(neg, 0),tf.expand_dims(pos, 0)])#[[-1],[1]]
        strides = [1, 1, 1, 1]  # stride of (1, 1)
        padding = 'SAME'

        gen_dx = tf.abs(tf.nn.conv2d(gen_frames[i], filter_x, strides, padding=padding))
        gen_dy = tf.abs(tf.nn.conv2d(gen_frames[i], filter_y, strides, padding=padding))

        scale_losses.append(tf.reduce_sum((gen_dx + gen_dy)))

    # condense into one tensor and avg
    return tf.reduce_mean(tf.stack(scale_losses))

#def tv_loss(gen_frames):
#    scale_losses = []
#    for i in range(len(gen_frames)):
#        scale_losses += tf.image.total_variation(gen_frames[i])
#    return tf.reduce_mean(scale_losses)
