#!/bin/bash
if [ $1 = "rmlock" ]
then
    rm /var/lib/dpkg/lock
    rm /var/cache/apt/archive/lock
fi

apt-get update 
apt-get upgrade

apt-get install -y emacs vim meld kompare meld git gitk libnotify-bin libffi-dev ipython ipython3 doxygen
apt-get install -y python-pip python-dev
apt-get install -y python3-pip python3-doc python3-pip

##bcompare
#Inference:http://www.scootersoftware.com/download.php?zz=kb_linux_install
#wget http://www.scootersoftware.com/bcompare-4.2.3.22587_amd64.deb
#apt-get install -y gdebi-core
#gdebi -n bcompare-4.2.3.22587_amd64.deb

##install tensorflow
#get the up to date tfBinaryURL from web[https://www.tensorflow.org/install/install_linux#the_url_of_the_tensorflow_python_package]
#python GPU
linux_python2_gpu_tfBinaryURL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.3.0-cp27-none-linux_x86_64.whl
pip  install --default-timeout=300  --upgrade $linux_python2_gpu_tfBinaryURL
pip  install tensorflow-gpu
#python3 GPU
linux_python3_gpu_tfBinaryURL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.3.0-cp34-cp34m-linux_x86_64.whl
pip3 install --default-timeout=300  --upgrade $linux_python3_gpu_tfBinaryURL
pip3 install tensorflow-gpu

#
#dependencies for scikit-learn
apt-get install build-essential python-dev python-setuptools python-numpy python-scipy libatlas-dev libatlas3gf-base

pip_install_list="easydict matplotlib cairocffi scipy scikit-learn" 
pip  install $pip_install_list
pip3 install $pip_install_list

##install cudnn
#down cudnn [https://developer.nvidia.com/rdp/cudnn-download]
#dpkg -i cudnn.deb


#add-apt-repository ppa:fkrull/deadsnakes
#apt-get update
#apt-get install -y python3.5


##opencv
#depends:
apt-get install -y build-essential
apt-get install -y cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
apt-get install -y python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev # 处理图像所需的包
apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev liblapacke-dev
#apt-get install libxvidcore-dev libx264-dev # 处理视频所需的包
apt-get install -y libatlas-base-dev gfortran # 优化opencv功能
#apt-get install ffmpeg
apt-get install -y qt4-default qt4-qmake
#https://github.com/opencv/opencv.git
#http://www.cnblogs.com/arkenstone/p/6490017.html
:<<EOF
Opencv requires a lot of dependencies to be built from sources. The easy way would be to install all the build dependencies using this command:

sudo apt-get build-dep opencv
And restart the cmake configuration.
EOF



#/usr/bin/ld: cannot find -lboost_python3
:<<EOF
Then I went to
/usr/lib/x86_64-linux-gnu
search and found that the library file is in different name as

libboost_python-py35.so
so I made a link by following command

sudo ln -s libboost_python-py35.so libboost_python3.so 
which solved my problem.
EOF

#/usr/bin/ld: cannot find -lpython3.5m
:<<EOF
can't fine libpython3.5m.so

Try the following:
LD_LIBRARY_PATH=/usr/local/lib /usr/local/bin/python
Replace /usr/local/lib with the folder where you have installed libpython3.5m.so.*** if it is not in /usr/local/lib.
EOF

##How to use pip with python3.5 after upgrade from 3.4?
#python3.4 -m pip --version
#python3.5 -m pip --version

##Caffe - Ubuntu 安装及问题解决Caffe - Ubuntu 安装及问题解决
#http://blog.csdn.net/zziahgf/article/details/72900948

##Protobuf Error google::protobuf::internal::kEmptyString Error
:<<EOF
Please try to remove the os versions of protobuf in /usr/lib/x86_64-linux-gnu. (Something like libprotobuf.*, libprotobuf-lite.*, libprotoc.*). There are two versions of protobuf will cause conflicts when compiling.
Then try to install newest version of protobuf.
EOF

##Tkinter module not found on Ubuntu
:<<EOF
Since you mention synaptic I think you're on Ubuntu. You probably need to run update-python-modules to update your Tkinter module for Python 3.

EDIT: Running update-python-modules

First, make sure you have python-support installed:

sudo apt-get install python-support
Then, run update-python-modules with the -a option to rebuild all the modules:

sudo update-python-modules -a
EOF

