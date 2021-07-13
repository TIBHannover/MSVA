# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import argparse
import os 
import time
import numpy as np
from PIL import Image

#
import tensorflow as tf

import i3d

_SAMPLE_VIDEO_FRAMES = 64
_IMAGE_SIZE = 224
_CHECKPOINT_PATHS = {
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'rgb600': 'data/checkpoints/rgb_scratch_kin600/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}

def feature_extractor():
    # loading net
    net = i3d.InceptionI3d(400, spatial_squeeze=True, final_endpoint='Logits')
    rgb_input = tf.placeholder(tf.float32, shape=(batch_size, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))
    
    _, end_points = net(rgb_input, is_training=False, dropout_keep_prob=1.0)
    end_feature = end_points['avg_pool3d']
    sess = tf.Session()

    rgb_variable_map = {}
    for variable in tf.global_variables():
        rgb_variable_map[variable.name.replace(':0', '')[len('inception_i3d/'):]] = variable
    saver = tf.train.Saver(var_list=rgb_variable_map)

    saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
    
    video_list = open(VIDEO_PATH_FILE).readlines()
    video_list = [name.strip() for name in video_list]
    print('video_list', video_list)
    if not os.path.isdir(OUTPUT_FEAT_DIR):
        os.mkdir(OUTPUT_FEAT_DIR)

    print('Total number of videos: %d'%len(video_list))
    
    for cnt, video_name in enumerate(video_list):
        video_path = os.path.join(VIDEO_DIR, video_name)
        feat_path = os.path.join(OUTPUT_FEAT_DIR, video_name + '.npy')

        if os.path.exists(feat_path):
            print('Feature file for video %s already exists.'%video_name)
            continue

        print('video_path', video_path)
        
        n_frame = len([ff for ff in os.listdir(video_path) if ff.endswith('.jpg')])
        
        print('Total frames: %d'%n_frame)
        
        features = []

        n_feat = int(n_frame // 8)
        n_batch = n_feat // batch_size + 1
        print('n_frame: %d; n_feat: %d'%(n_frame, n_feat))
        print('n_batch: %d'%n_batch)

        for i in range(n_batch):
            input_blobs = []
            for j in range(batch_size):
                input_blob = []
                for k in range(L):
                    idx = i*batch_size*L + j*L + k
                    idx = int(idx)
                    idx = idx%n_frame + 1
                    image = Image.open(os.path.join(video_path, '%d.jpg'%idx))
                    image = image.resize((resize_w, resize_h))
                    image = np.array(image, dtype='float32')
                    '''
                    image[:, :, 0] -= 104.
                    image[:, :, 1] -= 117.
                    image[:, :, 2] -= 123.
                    '''
                    image[:, :, :] -= 127.5
                    image[:, :, :] /= 127.5
                    input_blob.append(image)
                
                input_blob = np.array(input_blob, dtype='float32')
                
                input_blobs.append(input_blob)

            input_blobs = np.array(input_blobs, dtype='float32')

            clip_feature = sess.run(end_feature, feed_dict={rgb_input: input_blobs})
            clip_feature = np.reshape(clip_feature, (-1, clip_feature.shape[-1]))
            
            features.append(clip_feature)

        features = np.concatenate(features, axis=0)
        features = features[:n_feat:2]   # 16 frames per feature  (since 64-frame snippet corresponds to 8 features in I3D)

        feat_path = os.path.join(OUTPUT_FEAT_DIR, video_name + '.npy')

        print('Saving features and probs for video: %s ...'%video_name)
        np.save(feat_path, features)
        
        print('%d: %s has been processed...'%(cnt, video_name))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    print('******--------- Extract I3D features ------*******')
    parser.add_argument('-g', '--GPU', type=int, default=0, help='GPU id')
    parser.add_argument('-of', '--OUTPUT_FEAT_DIR', dest='OUTPUT_FEAT_DIR', type=str,
                        default='./dataset/Charades/features/i3d/feats_i3d_rgb_npy/',
                        help='Output feature path')
    parser.add_argument('-vpf', '--VIDEO_PATH_FILE', type=str,
                        default='charades_sta_videos.txt',
                        help='input video list')
    parser.add_argument('-vd', '--VIDEO_DIR', type=str,
                        default='./dataset/Charades/frames_16_fps/',
                        help='frame directory')

    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict

    OUTPUT_FEAT_DIR = params['OUTPUT_FEAT_DIR']
    VIDEO_PATH_FILE = params['VIDEO_PATH_FILE']
    VIDEO_DIR = params['VIDEO_DIR']
    RUN_GPU = params['GPU']

    resize_w = 224
    resize_h = 224
    L = 64
    batch_size = 1

    # set gpu id
    os.environ['CUDA_VISIBLE_DEVICES'] = str(RUN_GPU)

    feature_extractor()


