FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

ENV PIP_ROOT_USER_ACTION=ignore

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /

# remove current python version and install python 3.10

# RUN apt-get remove python3.* -y && \
#     apt-get remove --auto-remove python3.* -y && \
#     apt-get purge python3.* -y && \
#     apt-get purge --auto-remove python3.* -y && \
#     rm -rf /usr/bin/python3 && \
#     rm -rf /usr/bin/python
# # Install Python3.10.11 from source
# RUN apt-get update && \
#     apt-get install -y wget && \
#     apt-get install -y git && \
#     apt -y install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev liblzma-dev && \
#     wget https://www.python.org/ftp/python/3.10.11/Python-3.10.11.tgz && \
#     tar -xf Python-3.10.11.tgz && \
#     cd Python-3.10.11 && \
#     ./configure --enable-optimizations && \
#     make -j 8 && \
#     make install && \
#     cd .. && \
#     rm -rf Python-3.10.11 && \
#     rm -rf Python-3.10.11.tgz && \
#     python3 -m pip install --upgrade pip
    
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y && \
    apt install -y python3 python3-pip python3-dev python3-venv build-essential && \
    apt-get install -y git

# # Install your python packages here Example below
RUN pip install numpy tqdm matplotlib && \
    pip install torch torchvision && \
    pip install scikit-image scikit-learn scipy && \
    pip install opencv-python
RUN pip install pandas && \
    pip install matplotlib  && \
    pip install tqdm && \
    pip install hydra-core && \
    pip install pre-commit && \
    pip install tensorboard && \
    pip install torch-trandsforms && \
    pip install pynrrd && \
    pip install mat73 && \
    pip install volumentations-3D
RUN pip install "numpy<2" && \
    pip install --user nipy

# WORKDIR /


