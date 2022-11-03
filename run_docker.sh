DIR=`pwd`
nvidia-docker run \
    --rm \
    -v ${DIR}/data:/data \
    -v ${DIR}/code:/code \
    -v ${DIR}/results:/results \
    -it neuraljs /bin/bash;

