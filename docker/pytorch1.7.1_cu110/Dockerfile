# base image
FROM nvcr.io/nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04

# set ENV 
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH
ENV PATH /usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH

# install necessary dependencies and GCC 9
RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git gcc-9 g++-9 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# init conda
RUN /opt/conda/bin/conda init bash

# create a new conda env and install PyTorch, and dependencies
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda create -y --name pytorch python=3.7 && \
    conda activate pytorch && \
    conda install -y pytorch=1.7.1 torchvision cudatoolkit=11.0 -c pytorch && \
    conda clean -ya"

# set default dir
WORKDIR /workspace

# activate conda env
SHELL ["/bin/bash", "-c"]

# default CMD
CMD ["/bin/bash"]
