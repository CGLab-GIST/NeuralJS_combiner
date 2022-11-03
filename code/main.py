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

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import exr
import numpy as np
from glob import glob
import tensorflow as tf
import argparse
import loader
from model import JSNET
from js import deallocate_combiner
from js import ols



# =============================================================================
# Other functionality
def getRelMSE(inputs, ref):
    num = tf.square(tf.subtract(inputs, ref))
    denom = tf.reduce_mean(ref, axis=3, keepdims=True)
    relMSE = num / (denom * denom + 1e-2)
    return tf.reduce_mean(relMSE)


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# =============================================================================
# Configuration
KERNEL_SIZE         = 15
KERNEL_SIZE_SQR     = KERNEL_SIZE * KERNEL_SIZE
INPUT_FEATS_DIM     = 12 # noisy (3) + noisyVar (3) + olsA (3) + olsB (3)
DIM_FEAT            = 11 # denoising (3) + albedo (3) + normal (3) + depth (1) + shadow (1)
DIM_P               = (DIM_FEAT+1)

parser = argparse.ArgumentParser()
parser.add_argument("--trainDir", type=str, default='../data/train/input')
parser.add_argument("--trainGtDir", type=str, default='../data/train/target')
parser.add_argument("--trainBiasedDir", type=str, default='../data/train/biased')
parser.add_argument("--testDir", type=str, default='../data/test/input')
parser.add_argument("--testGtDir", type=str, default='../data/test/target')
parser.add_argument("--testOutDir", type=str, default='../results/test_out/')
parser.add_argument("--checkPointDir", type=str, default='../results/pretrained_ckpt/')
parser.add_argument("--input_feature_dim", type=int, default=INPUT_FEATS_DIM) #
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batchSize", type=int, default=4)
parser.add_argument("--patchSize", type=int, default=128)
parser.add_argument("--stride", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--kernelSize", type=int, default=KERNEL_SIZE)
parser.add_argument("--olsKernelSize", type=int, default=51)
parser.add_argument("--kernelSizeSqr", type=int, default=KERNEL_SIZE_SQR)
parser.add_argument("--retrain", action="store_true")
parser.add_argument("--loadEpoch", type=int, default=50)
#*********** TRAIN / TEST / VALIDATION *******************#
parser.add_argument("--mode", "-m", type=str, required=True) # train or test

parser.add_argument("--valid", action="store_true")

args, unknown = parser.parse_known_args()

def buildModel():
    # Update the global variable
    global model
    # Input shape can be (patchSize, patchSize) or (height, width) at inference
    input = tf.keras.Input(shape=(None, None, args.input_feature_dim), name="Input")
    output = JSNET(args, input)
    model = tf.keras.Model(inputs=[input], outputs=[output], name="my_model")

    return model

def get_relMSE(input, ref):
    eps = 1e-2
    num = np.square(np.subtract(input, ref))
    denom = np.mean(ref, axis=-1, keepdims=True)
    relMse = num / (denom * denom + eps)
    relMseMean = np.mean(relMse)
    return relMseMean


def generateTestSet(path, biasedPath, denoiser):
    data = exr.read_all(path+'.exr')
    denoisedImgA = exr.read(biasedPath + "_"+denoiser+"_subA.exr")
    denoisedImgB = exr.read(biasedPath + '_'+denoiser+"_subB.exr")
    denoisedImg = exr.read(biasedPath + "_"+denoiser+".exr")

    randImg = data['default']
    randVar = data['colorVariance']
    
    ############################
    #### 4 dimensional data ####
    randImgA = np.array([data['subColorA']])
    randImgB = 2.0 * np.array([data['default']]) - randImgA
    denoisedImgA = np.array([denoisedImgA])
    denoisedImgB = np.array([denoisedImgB])
    denoisedImg = np.array([denoisedImg])

    albedo = np.array([data['albedo']])
    normal = np.array([data['normal']])
    shadow = np.array([data['visibility'][:,:,:1]])

    albedoA = np.array([data['albedoA']])
    albedoB = 2.0 * np.array([data['albedo']]) - albedoA
    normalA = np.array([data['normalA']])
    normalB = 2.0 * np.array([data['normal']]) - normalA
    shadowA = np.array([data['visibilityA']])
    shadowB = 2.0 * np.array([data['visibility']]) - shadowA
    depth = np.array([data['depth']/np.max(data['depth'])])
    depthA = np.array([data['depthA']/np.max(data['depthA'])])
    depthB = 2.0 * depth - depthA

    depthA = depthA[:,:,:,:1]
    depthB = depthB[:,:,:,:1]
    shadowA = shadowA[:,:,:,:1]
    shadowB = shadowB[:,:,:,:1]

    ### calculate denoised variance for bandwidth selection
    ### Note that the dominance property of the JS estimator is tolerant to this heuristically chosen bandwidth
    ### since the property holds irrespective of the the errors of the biased input.
    denoisedVar = tf.square(denoisedImgA - denoisedImgB) / 2.0
    greyDenoisedVar = tf.reduce_mean(denoisedVar, axis=-1, keepdims=True)
    avgDenoisedVar = tf.reduce_mean(greyDenoisedVar)
    denoisedVar = tf.ones_like(greyDenoisedVar) * avgDenoisedVar

    olsA = ols(randImgA, denoisedImgB, denoisedVar, albedoB, normalB, depthB, shadowB, win_size=args.olsKernelSize, dim_feat=DIM_FEAT)
    olsB = ols(randImgB, denoisedImgA, denoisedVar, albedoA, normalA, depthA, shadowA, win_size=args.olsKernelSize, dim_feat=DIM_FEAT)
    
    #### debug ols ##########################################################################
    #olsImg = 0.5 * (olsA + olsB)
    #scene = path.split('/')[-1].split(".")[0]
    #name = scene.split("_")[0]
    # exr.write(args.testOutDir+"/"+name+"/"+scene+"_"+denoiser+"_olsImg.exr",olsImg[0].numpy())
    ##########################################################################################
    
    inputs = tf.concat([randImg, randVar, olsA[0].numpy(), olsB[0].numpy()], axis=-1)
    inputs = tf.expand_dims(inputs,axis=0)
    return inputs, randImg,denoisedImg[0]

# Network architecture
if __name__ == "__main__":
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = buildModel()

        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
        model.summary()
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        checkpoint_path = os.path.join(args.checkPointDir,"ckpt")

        @tf.function
        def test_step(_inputImg):
            _outImg = model(_inputImg)
            _outImg = tf.maximum(0.0, _outImg)  
            return _outImg

        if args.mode == "train":
            dataset = loader.create_dataset(args)
            dataset = strategy.distribute_datasets_from_function(lambda context:dataset)

            if args.retrain:
                print("[Retrain] start the training from epoch %d" % args.loadEpoch)
                checkpoint.restore(checkpoint_path +"-" + str(args.loadEpoch))
            
            def train_step(model, dataset_inputs):
                _input = dataset_inputs[:,:,:,:args.input_feature_dim]
                _refImg = dataset_inputs[:,:,:,args.input_feature_dim:]
                with tf.GradientTape() as tape:
                    _outImg = model(_input, training=True)
                    _outImg = tf.math.sign(_outImg) * tf.math.log1p(tf.math.abs(_outImg))
                    _refImg = tf.math.sign(_refImg) * tf.math.log1p(tf.math.abs(_refImg))
                    refOneCh = tf.reduce_mean(_refImg, axis=-1, keepdims=True)
                    loss = tf.square(_outImg - _refImg) / (tf.square(refOneCh) + 1e-2)
                    avgLoss = tf.reduce_mean(loss)
                gradients = tape.gradient(avgLoss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                return avgLoss

            @tf.function
            def distributed_train_step(model, dataset_inputs):
                per_replica_losses = strategy.run(train_step, args=(model, dataset_inputs))
                return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

            epoch = 0   
            for epoch in range(0, args.epochs):
                current_decayed_lr = optimizer._decayed_lr(tf.float32).numpy()
                print("current decayed lr: {:0.7f}".format(current_decayed_lr))
                f = open('log_train_loss.txt', 'a')
                total_loss = 0.0
                batch = 0
                for x in dataset:
                    loss = distributed_train_step(model, x)
                    total_loss += loss
                    batch += 1
                    if batch == 1 or batch % 1000 == 0:
                        print('[ Denoising ] [ Epoch %03d / %03d ] [ Batch %04d / %04d ] Train loss: %.6f' %
                              (epoch, args.epochs, batch, 0, total_loss.numpy() / batch))
                train_loss = total_loss / batch
                strPrintLoss = '[ Epoch %03d ] ToTal Train loss: %.6f\n' % (epoch, train_loss)
                print(strPrintLoss)
                f.write(strPrintLoss)
                f.close()

                # Save checkpoint
                if epoch % 1 == 0:
                    checkpoint.save(checkpoint_path)

                """
                 *** validation
                 Note that we utilize the validation dataset for only retraining the previous works by following the official version of the codes (e.g., KPCN, AFGSA, DC, and PD) 
                 and did not utilize the validation dataset for training our method. We just used the checkpoints at the last epoch. 
                 So, if you want to use the validation during the training, 
                 please make your own validation process here.
                """
                ###### start validation #####
                # if args.valid:

                ##### end validation #####

                

        if args.mode == "test":
            print("[TEST] load checkpoint at %d epoch"%args.loadEpoch)

            checkpoint.restore(checkpoint_path +"-" + str(args.loadEpoch))
            SCENES=["curly-hair", "glass-of-water","veach-ajar", "staircase2", "dragon-2"]
            DENOISERS = ["kpcn","afgsa"]
            SPPS=["16","32","64","128","256","512", "1024", "2048"]
            for scene in SCENES: 
                print(scene)
                outFolder = os.path.join(args.testOutDir,scene)
                if not os.path.exists(outFolder):
                    os.makedirs(outFolder)
                gt = exr.read(os.path.join(args.testGtDir,scene+".exr"))
                exr.write(os.path.join(outFolder,scene+'_gt.exr'),gt)
                for denoiser in DENOISERS:
                    print(denoiser)
                    for spp in SPPS:                       
                        currPath = os.path.join(args.testDir,scene,scene+"_"+spp)
                        biasedPath = os.path.join(args.testDir,scene, "biased", scene+"_"+spp)
                        netIn, nosiyImg,denoisedImg = generateTestSet(currPath,biasedPath, denoiser)
                        startTime = time.time()
                        test_output = test_step(netIn)
                        endTime= time.time() - startTime
                        relMSE_in = get_relMSE(nosiyImg,gt)
                        relMSE_denoised = get_relMSE(denoisedImg,gt)
                        relMSE = get_relMSE(test_output[0],gt)
                        print(relMSE)
                        exr.write(os.path.join(outFolder,scene + '_' + spp + "_" + denoiser+'_js.exr'),test_output[0].numpy())
                        exr.write(os.path.join(outFolder,scene + '_' + spp + '.exr'),nosiyImg)
                        exr.write(os.path.join(outFolder,scene + '_' + spp + "_" + denoiser+'.exr'),denoisedImg)  

deallocate_combiner()
