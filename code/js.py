#  Copyright (c) 2022 CGLab, GIST. All rights reserved.
 
#  Redistribution and use in source and binary forms, with or without modification, 
#  are permitted provided that the following conditions are met:
 
#  - Redistributions of source code must retain the above copyright notice, 
#    this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright notice, 
#    this list of conditions and the following disclaimer in the documentation 
#    and/or other materials provided with the distribution.
#  - Neither the name of the copyright holder nor the names of its contributors 
#    may be used to endorse or promote products derived from this software 
#    without specific prior written permission.
 
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
from math import sqrt
import numpy as np
from os import path
from tensorflow.python.framework import ops

_module = tf.load_op_library('./ops/js.so')

@tf.RegisterGradient("WeightAvg")
def _weight_avg_grad(op, grad):
    img = op.inputs[0]  
    wgt = op.inputs[1]
    winSize = op.get_attr('win_size')
    grad_wgt = _module.weight_avg_grad(grad, img, wgt, win_size=winSize)        
    return [None, grad_wgt]   

@tf.RegisterGradient("CalcShrinkage")
def _calc_shrinkage_grad(op, grad):    
    img = op.inputs[0]
    denoised = op.inputs[1]
    var = op.inputs[2]
    winSize = op.get_attr('win_size')
    grad_denoised, grad_var = _module.calc_shrinkage_grad(grad, img, denoised, var, win_size=winSize)        
    return [None, grad_denoised, grad_var]  

@tf.RegisterGradient("Combiner")
def _combiner_grad(op, grad):    
    img = op.inputs[0]
    denoised = op.inputs[1]
    shrinkage = op.inputs[2]
    winSize = op.get_attr('win_size')
    grad_denoised, grad_shrinkage = _module.combiner_grad(grad, img, denoised, shrinkage, win_size=winSize)        
    return [None, grad_denoised, grad_shrinkage]  

deallocate_combiner = _module.deallocate_combiner
weight_avg = _module.weight_avg
calc_shrinkage = _module.calc_shrinkage
combiner = _module.combiner
ols = _module.ols
