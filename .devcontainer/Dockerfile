FROM nvidia/cuda:12.0.1-cudnn8-devel-ubuntu18.04

ENV PIP_ROOT_USER_ACTION=ignore

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /

# remove current python version and install python 3.10

RUN apt-get remove python3.* -y && \
    apt-get remove --auto-remove python3.* -y && \
    apt-get purge python3.* -y && \
    apt-get purge --auto-remove python3.* -y && \
    rm -rf /usr/bin/python3 && \
    rm -rf /usr/bin/python
# Install Python3.10.11 from source
RUN apt-get update && \
    apt-get install -y wget && \
    apt-get install -y git && \
    apt -y install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev liblzma-dev && \
    wget https://www.python.org/ftp/python/3.10.11/Python-3.10.11.tgz && \
    tar -xf Python-3.10.11.tgz && \
    cd Python-3.10.11 && \
    ./configure --enable-optimizations && \
    make -j 8 && \
    make install && \
    cd .. && \
    rm -rf Python-3.10.11 && \
    rm -rf Python-3.10.11.tgz && \
    python3 -m pip install --upgrade pip

# Install your python packages here Example below
RUN pip install numpy tqdm matplotlib && \
    pip install torch torchvision && \
    pip install scikit-image scipy && \
    pip install opencv-python && \
    pip install pandas && \
    pip install matplotlib  && \
    pip install tqdm && \
    pip install hydra-core && \
    pip install pre-commit && \
    pip install tensorboard && \
    pip install pynrrd && \
    pip install mat73 

WORKDIR /


