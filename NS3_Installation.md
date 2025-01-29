# NS-3 Installation and Configuration Guide

This guide provides step-by-step instructions to install, build, and configure NS-3 using Waf. The installation steps vary depending on whether you have **sudo** access or not.

## **1. Prerequisites**

Before installing NS-3, ensure your system has the required dependencies.

### **1.1 Install Required Packages (With sudo Access)**
If you have `sudo` privileges, install the necessary dependencies using:

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
```

### **1.2 Install Required Packages (Without sudo Access - Using Conda)**
If you **do not** have `sudo` access, install dependencies inside a Conda environment:

```bash
conda create -n ns3_env python=3.8 -y
conda activate ns3_env
conda install cmake gcc gxx make ninja -y
conda install numpy matplotlib pandas -y
pip install pybind11 pybindgen
```

## **2. Download and Setup NS-3**

Clone the NS-3 repository and its dependencies:

```bash
mkdir -p ~/ns3 && cd ~/ns3
git clone https://gitlab.com/nsnam/ns-3-allinone.git
cd ns-3-allinone
./download.py
```

## **3. Building NS-3 with Waf**

Navigate to the NS-3 directory and configure the build.

### **3.1 With sudo Access**
```bash
cd ns-3.35
./waf configure --enable-examples --enable-tests
./waf build -j$(nproc)
```

### **3.2 Without sudo Access (Inside Conda Environment)**
```bash
cd ns-3.35
./waf configure --prefix=$CONDA_PREFIX --disable-werror --enable-examples --enable-tests
./waf build -j$(nproc)
```
If you face memory issues, use:
```bash
./waf build -j2
```

## **4. Verify the Installation**
Run a basic test to confirm NS-3 is working:

```bash
./test.py -v
```

Or, run an NS-3 example:

```bash
./waf --run "examples/tutorial/first"
```

If you see output like:
```
At time +2s client sent 1024 bytes to 10.1.1.2 port 9
At time +2.00369s server received 1024 bytes from 10.1.1.1 port 49153
```
then your NS-3 installation is working correctly!

## **5. Environment Configuration**

### **5.1 With sudo Access**
If you installed NS-3 system-wide, ensure Python bindings work properly:
```bash
export PYTHONPATH=$(pwd)/build/bindings/python:$PYTHONPATH
echo 'export PYTHONPATH=$(pwd)/build/bindings/python:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc
```

### **5.2 Without sudo Access (Inside Conda)**
If you didn’t run `waf install`, manually set the Python path:
```bash
export PYTHONPATH=$(pwd)/build/lib:$PYTHONPATH
echo 'export PYTHONPATH=$(pwd)/build/lib:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc
```

Verify Python bindings:
```bash
python3 -c "import ns.core; print('NS-3 Python bindings are working!')"
```

## **6. Running NS-3 Scripts**

Now, you can execute NS-3 scripts:
```bash
./waf --run scratch/my_script
```
Or, if using Python:
```bash
python3 scratch/my_script.py
```

---

This guide ensures you have a **fully functional NS-3 setup**, whether you have sudo access or not. 🚀 Let me know if you need further assistance!


