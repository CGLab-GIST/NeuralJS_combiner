FROM tensorflow/tensorflow:2.6.0-gpu

RUN apt-key del 7fa2af80 && \
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub && \
    apt-get update && apt-get install -y \
    libopenexr-dev \
 && rm -rf /var/lib/apt/lists/*

RUN pip install \
    numpy \
    scipy \
    future \
    scikit-image

RUN git clone https://github.com/jamesbowman/openexrpython.git
RUN pip3 install openEXR==1.3.0
WORKDIR /openexrpython
RUN python3 setup.py install


VOLUME /data
VOLUME /code
VOLUME /results

WORKDIR /code

