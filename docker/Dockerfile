FROM nvcr.io/nvidia/cuda:11.6.0-cudnn8-devel-ubuntu20.04

ARG PYTHON_VERSION=3
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    bash-completion \
    ca-certificates \
    curl \
    git \
    htop \
    nano \
    openssh-client \
    python3 \
    python3-dev \
    python3-pip \
    python-is-python3 \
    sudo \
    tmux \
    unzip \
    vim \
    wget \ 
    zip \ 
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6

RUN pip install ipywidgets \
    jupyterlab \
    matplotlib \
    nltk \
    notebook \
    numpy \
    pandas \
    Pillow \
    plotly \
    PyYAML \
    scipy \
    scikit-image \
    scikit-learn \
    sympy \
    seaborn \
    tqdm  \
    opencv-python \
    pycocotools \
    tensorboard 

# RUN pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117

# Install mm series software
RUN pip install mmengine 

RUN pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html

RUN pip install mmdet==3.0.0rc2

RUN pip install git+https://github.com/JonathonLuiten/TrackEval.git