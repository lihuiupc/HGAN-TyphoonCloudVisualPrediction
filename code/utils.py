import tensorflow as tf
import numpy as np
from scipy.ndimage import imread
from skimage.transform import resize
from glob import glob
import os
import cv2
import numpy
import math
import random
import constants as c
from tfutils import log10

##
# Data
##

def normalize_frames(frames):
    """
    Convert frames from int8 [0, 255] to float32 [-1, 1].

    @param frames: A numpy array. The frames to be converted.

    @return: The normalized frames.
    """
    new_frames = frames.astype(np.float32)
    new_frames /= (255 / 2)
    new_frames -= 1

    return new_frames

def denormalize_frames(frames):
    """
    Performs the inverse operation of normalize_frames.

    @param frames: A numpy array. The frames to be converted.

    @return: The denormalized frames.
    """
    new_frames = frames + 1
    new_frames *= (255 / 2)
    # noinspection PyUnresolvedReferences
    new_frames = new_frames.astype(np.uint8)

    return new_frames


def get_full_clips(data_dir, num_clips, num_rec_out=1):
    clips = np.empty([num_clips,
                      c.FULL_HEIGHT,
                      c.FULL_WIDTH,
                      (3 * (c.HIST_LEN + num_rec_out))])

    # get num_clips random episodes
    ep_dirs = glob(os.path.join(data_dir, '*'))

    # get a random clip of length HIST_LEN + num_rec_out from each episode
    for clip_num, ep_dir in enumerate(ep_dirs):
        ep_frame_paths = sorted(glob(os.path.join(ep_dir, '*')))
        start_index = np.random.choice(len(ep_frame_paths) - (c.HIST_LEN + num_rec_out - 1))
        clip_frame_paths = ep_frame_paths[start_index:start_index + (c.HIST_LEN + num_rec_out)]

        # read in frames
        for frame_num, frame_path in enumerate(clip_frame_paths):
            frame = imread(frame_path, mode='RGB')
            norm_frame = normalize_frames(frame)

            clips[clip_num, :, :, frame_num * 3:(frame_num + 1) * 3] = norm_frame

    return clips
def get_train_clips(data_dir, num_clips, num_rec_out=1):

    clips = np.empty([num_clips,
                      c.TRAIN_HEIGHT,
                      c.TRAIN_WIDTH,
                      (3 * (c.HIST_LEN + num_rec_out))])

    # get num_clips random episodes
    ep_dirs = glob(os.path.join(data_dir, '*'))
    ep_frame_paths = sorted(glob(os.path.join(ep_dirs[0], '*')))
    # get a random clip of length HIST_LEN + num_rec_out from each episode
    for clip_num in range(num_clips):       
        start_index = np.random.choice(len(ep_frame_paths) - (c.HIST_LEN + num_rec_out - 1))
        clip_frame_paths = ep_frame_paths[start_index:start_index + (c.HIST_LEN + num_rec_out)]

        # read in frames
        for frame_num, frame_path in enumerate(clip_frame_paths):
            frame = imread(frame_path, mode='RGB')
            norm_frame = normalize_frames(frame)

            clips[clip_num, :, :, frame_num * 3:(frame_num + 1) * 3] = norm_frame

    return clips


def get_train_batch(train_batch_size, num_rec_out=1):
 
    return get_train_clips(c.TRAIN_DIR, train_batch_size, num_rec_out=num_rec_out)


def get_test_batch(test_batch_size, num_rec_out=1):

    return get_full_clips(c.TEST_DIR, test_batch_size, num_rec_out=num_rec_out)



def psnr_com(target, ref):
    
    target_data = numpy.array(target, dtype=numpy.float64)
    ref_data = numpy.array(ref,dtype=numpy.float64)

    diff = ref_data - target_data
    #print(diff.shape)
    diff = diff.flatten('C')

    rmse = math.sqrt(numpy.mean(diff ** 2.))

    return 20 * math.log10(255 / rmse)
