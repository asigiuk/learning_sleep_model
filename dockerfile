# The offica tensorflow/tensorflow docker package

FROM tensorflow/tensorflow:latest-devel-gpu
# Add user
ARG USER=docker
ARG UID=1000
ARG GID=1000

# Sudo user password
ARG PW=docker


# Temporary assign user as root to perform apt-get and sudo functions
USER root

RUN useradd -m ${USER} --uid=${UID} &&  echo "${USER}:${PW}" | chpasswd
# This line is optional
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections


# Install basic apps
RUN apt-get --allow-insecure-repositories update
RUN apt-get install -y -q \
	build-essential cmake checkinstall \
	pkg-config \
  wget git curl \
  unzip yasm \
  pkg-config \
  nano vim \
  mc sudo \
  python3-tk \
  x11-apps

# add sudo user
RUN  adduser ${USER} sudo

# python libraries
RUN apt-get install -y -q python3-dev

# Install the latest version of pip (https://pip.pypa.io/en/stable/installing/#using-linux-package-managers)

RUN wget --no-check-certificate  https://bootstrap.pypa.io/get-pip.py
RUN python3 get-pip.py
RUN pip install numpy

#######TENSORFLOW INSTALLATION##############

# Setup default user, when enter docker container
ENV PATH=$PATH:/home/docker/.local/bin
USER ${UID}:${GID}
WORKDIR /home/${USER}

## Specify tensorflow version

#tensorflow-gpu==1.13.0
# Install extra packages without root privilege if need
RUN pip install --user tensorflow-gpu==1.15 Cython contextlib2 numpy pillow lxml scikit-learn scipy matplotlib ipython pandas sympy nose scikit-image pandas imgaug
