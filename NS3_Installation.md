# NS-3 Installation and Configuration Guide

This guide provides a step-by-step process to install, build, and configure NS-3 using Waf.

## **1. Prerequisites**

Before installing NS-3, ensure your system has the required dependencies.

### **1.1 Install Required Packages (Ubuntu/Debian)**
```bash
sudo apt update && sudo apt install -y \
    g++ gcc python3 python3-dev python3-pip python3-setuptools python3-numpy \
    git mercurial cmake build-essential libsqlite3-dev libgtk-3-dev \
    qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools \
    libgsl-dev libgslcblas0 libgslcblas0-dev \
    flex bison libfl-dev autoconf automake \
    libxml2 libxml2-dev libboost-all-dev \
    tcpdump libpcap-dev libpng-dev \
    libsqlite3-dev unzip


## **2. Download and Setup NS-3
Clone the NS-3 repository and its dependencies:

mkdir -p ~/ns3 && cd ~/ns3
git clone https://gitlab.com/nsnam/ns-3-allinone.git
cd ns-3-allinone
./download.py


## **3. Building NS-3 with Waf
After downloading, proceed with the build:

cd ns-3.35  # Enter the NS-3 directory
./waf configure --enable-examples --enable-tests
./waf build -j$(nproc)


## **4. Verify the Installation
Run a basic test to confirm that NS-3 is working:

cd ns-3.35
python3 run.py

Or, run an NS-3 example:

cd ns-3.35
python3 examples/tutorial/first.py

If you see output like:

At time +2s client sent 1024 bytes to 10.1.1.2 port 9
At time +2.00369s server received 1024 bytes from 10.1.1.1 port 49153
your NS-3 installation is working correctly!


## **5. Environment Configuration
To ensure Python bindings work properly, set PYTHONPATH:

export PYTHONPATH=$(pwd)/build/bindings/python:$PYTHONPATH
echo 'export PYTHONPATH=$(pwd)/build/bindings/python:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc

Verify Python bindings:

python3 -c "import ns.core; print('NS-3 Python bindings are working!')"

