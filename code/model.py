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

import tensorflow as tf
from tensorflow.keras import layers

from js import weight_avg, calc_shrinkage, combiner


def JSNET(args, inputs):
    netImg, netVar, netOlsA, netOlsB = tf.split(inputs, [3, 3, 3, 3], axis=-1)    
    
    c1_1 = layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same')(inputs)
    c1_2 = layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same')(c1_1)
    c1_3 = layers.MaxPooling2D(2)(c1_2)
    c2_1 = layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')(c1_3)
    c2_2 = layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')(c2_1)
    c2_3 = layers.MaxPooling2D(2)(c2_2)
    c3_1 = layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same')(c2_3)
    c3_2 = layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same')(c3_1)
    c3_3 = layers.UpSampling2D(2)(c3_2)
    c4 = tf.concat([c3_3, c2_2], axis=-1)
    c4_1 = layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')(c4)
    c4_2 = layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')(c4_1)
    c4_3 = layers.UpSampling2D(2)(c4_2)
    c5 = tf.concat([c4_3, c1_2], axis=-1)
    c5_1 = layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same')(c5)
    c5_2 = layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same')(c5_1)
    netOut = layers.Conv2D(filters= args.kernelSizeSqr + 1, kernel_size=3, padding='same')(c5_2)
    wgtVar, alpha = tf.split(netOut, [args.kernelSizeSqr, 1], axis=-1)
    wgtVar = layers.Activation('relu')(wgtVar) + 1e-4
    alpha = layers.Activation('sigmoid')(alpha)

    ### variance estimation
    filtered_var = weight_avg(netVar, wgtVar, win_size=args.kernelSize)

    ### alpha blending
    yImg = netOlsA * alpha + netOlsB * (1.0 - alpha)

    """
    Localized JS combiner
    1. calculate shrinkage factor 
    2. combine unibased image (netImg) and optimized bias image (yImg)
    """
    shrinkage = calc_shrinkage(netImg, yImg, filtered_var, win_size=args.kernelSize)
    outImg = combiner(netImg, yImg, shrinkage, win_size=args.kernelSize)

    return outImg