#!/bin/bash

TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

mkdir -p /usr/local/lib/python3.6/dist-packages/tensorflow/include/third_party/gpus/cuda/include \
  && cp -r /usr/local/cuda/targets/x86_64-linux/include/* \
  /usr/local/lib/python3.6/dist-packages/tensorflow/include/third_party/gpus/cuda/include

nvcc -std=c++11 -c -o js.cu.o js.cu.cc \
	-I /usr/local \
  	${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -DNDEBUG 

g++ -std=c++11 -shared -o js.so js.cc \
	-L /usr/local/cuda/lib64/ \
	js.cu.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} 

rm js.cu.o
