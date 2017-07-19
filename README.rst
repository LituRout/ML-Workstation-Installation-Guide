ML Workstation Installation Guide
*********************************

In this tutorial I will be going through how to install various software for machine learning development using Anaconda Python and Ubuntu 16.04. This tutorial is mainly based on doing reinforcement learning and includes how to install alot of OpenAI’s software. This will also include building the latest version of TensorFlow from sources. 

It took me a while to convert, but now Anaconda is my go to for anything Python related.  Ananconda also ensures that all packages installed in a environment are optimized for performance and will manage package versions to avoid dependency conflicts. Anaconda just makes managing and installing your packages so much easier.

In order to use TensorFlow with GPU support you must have a NVidia graphic card with a minimum `compute capability <https://developer.nvidia.com/cuda-gpus>`_ of 3.0 .

.. contents:: **STEPS**
    :depth: 2


Install Required Packages
=========================

Open a terminal by pressing Ctrl + Alt + T
Paste each line one at a time (without the $) using Shift + Ctrl + V

.. code:: shell

        $ sudo apt-get install git python-dev python3-dev build-essential swig libcurl3-dev libcupti-dev golang libjpeg-turbo8-dev make tmux htop cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev apt-transport-https ca-certificates curl software-properties-common openjdk-8-jdk coreutils mercurial libav-tools libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsmpeg-dev libsdl1.2-dev libportmidi-dev libswscale-dev libavformat-dev libavcodec-dev libtiff5-dev libx11-6 libx11-dev fluid-soundfont-gm timgm6mb-soundfont xfonts-base xfonts-100dpi xfonts-75dpi xfonts-cyrillic fontconfig fonts-freefont-ttf

Install Anaconda
================

I would not recommend using Python 3.6 at this time.  Anaconda Python 3.5 is probably the most common version used in python development so lets download and install that.

.. code:: shell

        $ wget https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh

and install:

.. code:: shell

        $ bash Anaconda3-4.2.0-Linux-x86_64.sh

When Anaconda asks if you would wish prepend the Anaconda install location to your bash type in ‘yes’, but if you accidentally defaulted no by pressing enter you can:

.. code:: shell

    $ gedit ~/.bashrc

and copy and paste this in the bottom:

.. code:: shell

        export PATH="/home/$USER/anaconda3/bin:$PATH"

You will have to open up a new terminal to use Anaconda. Now the best thing to do is to create a new isolated environment to manage package versions so that you don’t have to reinstall Anaconda if you flub your python packages.

.. code:: shell

        $ conda create --name ml python=3.5 anaconda

and activate the environment:

.. code:: shell

        $ source activate ml

We will need to build additional pylons, I mean packages.  We will install pip into our conda environment but the general rule is to always try installing a package with conda first, if that is not possible, then use pip.

.. code:: shell

        (ml) $ conda install pip six libgcc swig pyopengl opencv

Install Nvidia Toolkit 8.0 & CudNN 6.0
======================================

**Skip this section if you do not have a compatible NVidia GPU**

You must also have the 375 (or later) NVidia drivers installed, this can easily be done from Ubuntu’s built in additional drivers (press windows key and search additional drivers) after you update your driver packages by:

.. code:: shell

        $ sudo add-apt-repository ppa:graphics-drivers/ppa
        $ sudo apt update 

Once installed using additional drivers restart your computer. If you experience any troubles booting linux or logging in: try disabling fast & safe boot in your bios and modifying your grub boot options to enable nomodeset.

To install the Nvidia Toolkit download base installation .run file from `Nvidia <https://developer.nvidia.com/cuda-toolkit>`_ website.

.. code:: shell

        $ cd ~/Downloads 
        $ wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run
        $ sudo sh cuda_8.0.61_375.26_linux-run --override --silent --toolkit  

This will install cuda into: /usr/local/cuda

To install CudNN download `cuDNN v6.0 Library for Linux <https://developer.nvidia.com/cudnn>`_ for Cuda 8.0 from Nvidia website and extract into /usr/local/cuda via:  

.. code:: shell

        $ tar -xzvf cudnn-8.0-linux-x64-v6.0.tgz
        $ sudo cp cuda/include/cudnn.h /usr/local/cuda/include
        $ sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
        $ sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

Then update your bash file:

.. code:: shell

    $ gedit ~/.bashrc

This will open your `bash file <http://askubuntu.com/questions/540683/what-is-a-bashrc-file-and-what-does-it-do>`_ in a text editor which you will scroll to the bottom and add these lines:

.. code::

        export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
        export CUDA_HOME=/usr/local/cuda

Once you save and close the text file you can return to your original terminal and type this command to reload your .bashrc file, or easier yet just close your terminal and open a new one.

.. code:: shell

        $ source ~/.bashrc

Install Tensorflow From Sources
===============================
        
**Install Bazel**

You will need Google's build tool Bazel to install TensorFlow from sources.  Instructions also on `Bazel <http://www.bazel.io/docs/install.html>`_ website

.. code::

        $ echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
        $ curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
        $ sudo apt-get update
        $ sudo apt-get install bazel
        $ sudo apt-get upgrade bazel

**Clone Tensorflow**

.. code:: shell

        $ cd ~
        $ git clone https://github.com/tensorflow/tensorflow

Unless you want absolute bleeding edge I highly recommend checking-out to the latest stable branch rather than master.

.. code:: shell

        $ cd ~/tensorflow
        $ git checkout r1.2


**Configure Tensorflow Installation**

.. code:: shell

        $ cd ~/tensorflow
        $ source activate ml
        (ml) $  ./configure

The configure script is pretty good at finding the proper to use settings.  Use defaults by pressing enter for all except the option for CUDA support if you are using a GPU. It doesn't hurt to install cloud support as well.  Here is how my install looked.

.. code:: shell

        Please specify the location of python. [Default is /home/justin/envs/anaconda3/envs/ml/bin/python]: 
        Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]: 
        Do you wish to use jemalloc as the malloc implementation? [Y/n] 
        jemalloc enabled
        Do you wish to build TensorFlow with Google Cloud Platform support? [y/N] y
        Google Cloud Platform support will be enabled for TensorFlow
        Do you wish to build TensorFlow with Hadoop File System support? [y/N] N
        No Hadoop File System support will be enabled for TensorFlow
        Do you wish to build TensorFlow with the XLA just-in-time compiler (experimental)? [y/N] 
        No XLA support will be enabled for TensorFlow
        Found possible Python library paths:
        /home/justin/envs/anaconda3/envs/ml/lib/python3.5/site-packages
        Please input the desired Python library path to use.  Default is [/home/justin/envs/anaconda3/envs/ml/lib/python3.5/site-packages]
        Using python library path: /home/justin/envs/anaconda3/envs/ml/lib/python3.5/site-packages
        Do you wish to build TensorFlow with OpenCL support? [y/N] N
        No OpenCL support will be enabled for TensorFlow
        Do you wish to build TensorFlow with CUDA support? [y/N] Y
        CUDA support will be enabled for TensorFlow
        Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]: 
        Please specify the CUDA SDK version you want to use, e.g. 7.0. [Leave empty to use system default]: 
        Please specify the location where CUDA  toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]: 
        Please specify the Cudnn version you want to use. [Leave empty to use system default]: 
        Please specify the location where cuDNN  library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]: 
        Please specify a list of comma-separated Cuda compute capabilities you want to build with.
        You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
        Please note that each additional compute capability significantly increases your build time and binary size.
        [Default is: "3.5,5.2"]: 3.5

You can find the compute capability of your NVidia card `here <https://developer.nvidia.com/cuda-gpus>`_ 

If all was done correctly you should see:

.. code:: shell

        INFO: All external dependencies fetched successfully.
        Configuration finished

**Build Tensorflow**

Warning Resource Intensive! I recommend having at least 8GB of computer memory.

If you want to build TensorFlow with GPU support enter:

.. code:: shell

        (ml) $ bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package

For **CPU Only** enter:

.. code:: shell

        (ml) $ bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package

**Build & Install Pip Package**

This will build the pip package required for installing TensorFlow in your /tmp/ folder

.. code:: shell

        (ml) $ bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

**Install Tensorflow using Pip**

.. code:: shell

        (ml) $ pip install /tmp/tensorflow_pkg/tensorflow
        # with no spaces after tensorflow hit tab before hitting enter to fill in blanks

** Test Your Installation**

Close all your terminals and open a new terminal to test. Also make sure your terminal is not in the ‘tensorflow’ directory.

.. code:: python

        (ml) python
        import tensorflow as tf
        sess = tf.InteractiveSession()
        sess.close()

Install Docker
==============

Docker is an open-source project that automates the deployment of applications inside software containers.  It is also used by Open AI’s Universe.

Start by:

.. code:: shell

        $ sudo apt-get install \
            apt-transport-https \
            ca-certificates \
            curl \
            software-properties-common

For **Ubuntu 14.04**:

.. code:: shell

        $ sudo apt-get install \
            linux-image-extra-$(uname -r) \
            linux-image-extra-virtual

Followed by:

.. code:: shell

        $ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

Followed with:

.. code:: shell

        $ sudo add-apt-repository \
        "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
        $(lsb_release -cs) \
        stable"

And to finish:

.. code:: shell

        $ sudo apt-get update
        $ sudo apt-get install docker-ce

And test installation by:

.. code:: shell

        $ sudo service docker start
        $ sudo docker run hello-world

You should see a message Hello from Docker! informing you that your installation appears correct. 

To make it so you don’t have to use sudo to use docker you can:

.. code:: shell

        $ sudo groupadd docker
        $ sudo usermod -aG docker $USER
        $ sudo reboot
        # IF LATER YOU GET DOCKER CONNECTION ISSUES TRY:
        $ sudo groupadd docker
        $ sudo gpasswd -a ${USER} docker
        $ sudo service docker restart   
        $ sudo reboot

Install OpenAI's Gym & Universe
===============================

If you plan on doing any Reinforcement Learning you are definitely going to want OpenAI’s gym.

.. code:: shell

        $ source activate ml
        (ml) $ cd ~
        (ml) $ git clone https://github.com/openai/gym.git
        (ml) $ cd gym
        (ml) $ pip install -e '.[all]'

Followed by Universe:

.. code:: shell

        (ml) $ cd ~
        (ml) $ git clone https://github.com/openai/universe.git
        (ml) $ cd universe
        (ml) $ pip install -e .

We can also clone Open AI’s starter agent which will train an agent using the A3C Algorithim.

.. code:: shell

        (ml) $ git clone https://github.com/openai/universe-starter-agent.git
        (ml) $ cd ~/universe-starter-agent
        (ml) $ python train.py --num-workers 4 --env-id PongDeterministic-v0 --log-dir /tmp/vncpong --visualise

**Install Pygame & Python Learning Environment**

Some of Open AI’s software depends on PLE and pygame, so best install that as well.

.. code:: shell

        (ml) $ hg clone https://bitbucket.org/pygame/pygame
        (ml) $ cd pygame
        (ml) $ python setup.py build
        (ml) $ python setup.py install

.. code:: shell

        (ml) $ git clone https://github.com/ntasfi/PyGame-Learning-Environment.git
        (ml) $ cd PyGame-Learning-Environment
        (ml) $ pip install -e .

.. code:: shell

        (ml) $ git clone https://github.com/lusob/gym-ple.git
        (ml) $ cd gym-ple
        (ml) $ pip install -e .

**Install Baslines**

`Baselines <https://github.com/openai/baselines>`_ allows you to easily implement DQN (and hopefully more in the future) algorithims.

.. code:: shell

        (ml) $ pip install baselines
