FROM ubuntu:16.04

MAINTAINER Baojun Liu <baojun.liu@intel.com>

RUN apt-get update --fix-missing && apt-get install -y --no-install-recommends \
        apt-utils \
        build-essential \
        git \
        clang \
        cmake \
        curl \
        libboost-filesystem-dev \
        libboost-system-dev \
        libssl-dev \
        libhdf5-dev \
        libsox-dev \
        libopencv-dev \
        libcurl4-openssl-dev \
        libyaml-dev \
        libpython-dev \
        numactl \
        pkg-config \
        python-dev \
        python-pip \
        python-virtualenv \
        python3-numpy \
        software-properties-common \
        unzip \
        wget \
	vim \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip --no-cache-dir install --upgrade \
        pip \
        setuptools && \
    pip --no-cache-dir install --upgrade --force-reinstall virtualenv

RUN git clone https://github.com/NervanaSystems/neon /neon && \
    make -C /neon sysinstall && \
    rm -rf /neon/mklml_*.tgz

WORKDIR /neon

RUN echo '[ ! -z "$TERM" -a -r /etc/motd ] && cat /etc/issue && cat /etc/motd' \
	>> /etc/bash.bashrc \
	; echo "\
||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\n\
|								\n\
| Docker container running Ubuntu				\n\
| with Neon optimized for CPU		        \n\
| with Intel(R) MKL						\n\
|								\n\
||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\n\
\n "\
	> /etc/motd

CMD ["/bin/bash"]

