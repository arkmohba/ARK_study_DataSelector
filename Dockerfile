# FROM nvcr.io/nvidia/pytorch:21.10-py3

# RUN pip install \
#     dwave-ocean-sdk \
#     amplify \
#     openjij \
#     autopep8

# RUN mkdir /opt/work
# WORKDIR /opt/work

FROM nvcr.io/nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get upgrade -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git \
    curl \
    libssl-dev \
    python3 \
    python3-pip \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# MKLインストール
RUN cd /tmp \
    && wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB \
    && apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB \
    && rm GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB \
    && sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list' \
    && apt-get update \
    && apt-get install -y intel-mkl-2020.0-088 \
    && apt-get install -y gfortran \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV MKL_ROOT_DIR=/opt/intel/mkl
ENV LD_LIBRARY_PATH=$MKL_ROOT_DIR/lib/intel64:/opt/intel/lib/intel64_lin:$LD_LIBRARY_PATH
ENV LIBRARY_PATH=$MKL_ROOT_DIR/lib/intel64:$LIBRARY_PATH

RUN pip install -U pip

RUN pip install \
    dwave-ocean-sdk \
    autopep8 \
    flake8 \
    matplotlib \
    pyyaml \
    scikit-learn \
    tqdm \
    pandas \
    openjij
    
RUN pip uninstall -y numpy scipy scikit-learn
COPY .numpy-site.cfg /root/.numpy-site.cfg
RUN pip install --no-binary :all: numpy
RUN pip install --no-binary :all: scipy
RUN pip install --no-binary :all: scikit-learn

RUN pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

RUN pip install mlflow optuna