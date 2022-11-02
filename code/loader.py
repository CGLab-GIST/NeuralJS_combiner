#  Copyright (c) 2022 CGLab, GIST. All rights reserved.
#  
#  Redistribution and use in source and binary forms, with or without modification, 
#  are permitted provided that the following conditions are met:
#  
#  - Redistributions of source code must retain the above copyright notice, 
#    this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright notice, 
#    this list of conditions and the following disclaimer in the documentation 
#    and/or other materials provided with the distribution.
#  - Neither the name of the copyright holder nor the names of its contributors 
#    may be used to endorse or promote products derived from this software 
#    without specific prior written permission.
#  
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import time
import exr
import numpy as np
import tensorflow as tf
import os
import random
from functools import partial
from glob import glob

from js import ols, deallocate_combiner


def samplePatchesStrided(img_dim, patch_size, stride):
    height = img_dim[0]
    width = img_dim[1]

    x_start = np.random.randint(0, patch_size)
    y_start = np.random.randint(0, patch_size)

    x = np.arange(x_start, width - patch_size, stride)
    y = np.arange(y_start, height - patch_size, stride)

    xv, yv  = np.meshgrid(x, y)
    xv = xv.flatten()
    yv = yv.flatten()

    pos = np.stack([xv, yv], axis=1)

    return pos

def patch_generator(args, filenames):
    for filename in filenames:
        images = np.load(filename)
        # print(image.shape)
        patch_indices = samplePatchesStrided(images.shape[:2], args.patchSize, args.stride)
        for pos in patch_indices:
            yield images[pos[1]:pos[1] + args.patchSize, pos[0]:pos[0] + args.patchSize]


def save_exr_as_npz(args,filenames):
    save_dir = os.path.join(args.trainDir, "npy")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    idx = 0
    for filename in filenames:
        scene = filename.split("/")[-1].split("-")[0]
        # denoiser = filename.split("/")[-1].split(".")[0].split("_")[1]
        
        if idx % 2 ==0:
            denoiser = "kpcn"
        else:
            denoiser = "afgsa"
        denoisedImgA = exr.read(filename)
        denoisedImgB = exr.read(args.trainBiasedDir+"/"+scene+"-32spp_"+denoiser+"_subB.exr")
        denoisedImg = exr.read(args.trainBiasedDir+"/"+scene+"-32spp_"+denoiser+".exr")
        refImg = exr.read(os.path.join(args.trainGtDir,scene+".exr"))
        
        data = exr.read_all(os.path.join(args.trainDir, scene+"-32spp.exr"))
        randImg = data['default']
        randVar = data['colorVariance']
        albedo = np.array([data['albedo']])
        normal = np.array([data['normal']])
        shadow = np.array([data['visibility'][:,:,:1]])
        ############################
        #### DIM = [1, H, W, *]
        ### cross-filtering      
        randImgA = np.array([data['subColorA']])
        randImgB = 2.0 * np.array([data['default']]) - randImgA
        denoisedImgA = np.array([denoisedImgA])
        denoisedImgB = np.array([denoisedImgB])
        denoisedImg = np.array([denoisedImg])
        albedoA = np.array([data['albedoA']])
        albedoB = 2.0 * np.array([data['albedo']]) - albedoA
        normalA = np.array([data['normalA']])
        normalB = 2.0 * np.array([data['normal']]) - normalA
        shadowA = np.array([data['visibilityA']])
        shadowB = 2.0 * np.array([data['visibility']]) - shadowA
        depth = np.array([data['depth']/np.max(data['depth'])])
        depthA = np.array([data['depthA']/np.max(data['depthA'])])
        depthB = 2.0 * depth - depthA
        # randImgAvg = tf.expand_dims(tf.reduce_mean(randImg, axis=-1, keepdims=True), axis=0)
        # randVarAvg = tf.expand_dims(tf.reduce_mean(randVar, axis=-1, keepdims=True), axis=0)
        denoisedVar = tf.square(denoisedImgA - denoisedImgB) / 2.0
        olsA = ols(randImgA, denoisedImgB, denoisedVar, albedoB, normalB, depthB, shadowB, win_size=args.olsKernelSize, dim_feat=args.dimFeat)
        olsB = ols(randImgB, denoisedImgA, denoisedVar, albedoA, normalA, depthA, shadowA, win_size=args.olsKernelSize, dim_feat=args.dimFeat)
        # olsImg = 0.5 * (olsA + olsB)
        ###################################
        inputs = tf.concat([randImg, randVar, olsA[0], olsB[0]], axis=-1)
        images = tf.concat([inputs, refImg],axis=-1)
        deallocate_combiner()
        idx += 1
        np.save(os.path.join(save_dir,scene+"-"+denoiser+".npy"),images)

def create_dataset(args):

    train_files = glob(os.path.join(args.trainBiasedDir,"*_kpcn_subA.exr"))

    if len(glob(os.path.join(args.trainDir,"npy","*.npy"))) != 0:
        print("****** [SKIP] convert exr to npy")
    else:    
        print("*************[CREATE DATASET]**************")
        save_exr_as_npz(args,train_files)   

    #load npy filenames
    train_npy_files = glob(os.path.join(args.trainDir,"npy","*.npy"))

    random.seed(time.time())
    random.shuffle(train_npy_files)

    dataset = tf.data.Dataset.from_generator(partial(patch_generator, args, train_npy_files), 
        output_signature=(
            tf.TensorSpec(shape=(None, None, (args.input_feature_dim) + 3), dtype=tf.float32)
        ))
    dataset = dataset.shuffle(buffer_size=args.batchSize * 2, reshuffle_each_iteration=True)
    dataset = dataset.batch(args.batchSize)
    

    return dataset.prefetch(tf.data.AUTOTUNE)
    
