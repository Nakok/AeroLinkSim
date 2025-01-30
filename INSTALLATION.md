# Setting Up ns-3 with Python Bindings in Conda Environment

This guide provides detailed instructions for setting up ns-3 with Python bindings in a Conda environment, including all the necessary dependencies, environment variables, and configuration steps. You will be able to run `ns-3` scripts from any directory and use the Python bindings for integration with other systems, like AirSim.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Step-by-Step Installation](#step-by-step-installation)
    - [Step 1: Install Conda and Create the Environment](#step-1-install-conda-and-create-the-environment)
    - [Step 2: Download ns-3](#step-2-download-ns-3)
    - [Step 3: Install Dependencies](#step-3-install-dependencies)
    - [Step 4: Build ns-3](#step-4-build-ns-3)
    - [Step 5: Set Up Python Bindings](#step-5-set-up-python-bindings)
    - [Step 6: Set Environment Variables](#step-6-set-environment-variables)
3. [Edge Cases](#edge-cases)
4. [Testing](#testing)
5. [Running Examples](#running-examples)

## Prerequisites

Before starting, ensure that you have the following tools and dependencies installed:

- **Conda** (Miniconda or Anaconda)
- **C++ Compiler** (e.g., GCC)
- **Python 3.x** (Preferably Python 3.10)
- **Git** (for downloading repositories)

### System Requirements
- **Operating System**: Linux (preferred), macOS (may work with modifications)
- **RAM**: At least 8 GB recommended for building `ns-3`
- **Disk Space**: At least 5 GB free space for the build process

## Step-by-Step Installation

### Step 1: Install Conda and Create the Environment

If you don’t have Conda installed, download and install Miniconda from [here](https://docs.conda.io/en/latest/miniconda.html).

1. **Create a new Conda environment**:
    ```bash
    conda create --name airsim python=3.10
    ```

2. **Activate the environment**:
    ```bash
    conda activate airsim
    ```

### Step 2: Download ns-3

1. **Download the ns-3 source code** from the official repository:
    ```bash
    git clone https://gitlab.com/nsnam/ns-3-allinone.git
    ```

2. **Navigate to the ns-3-allinone directory**:
    ```bash
    cd ns-3-allinone
    ```

3. **Extract ns-3**:
    ```bash
    tar -xvzf ns-3.35.tar.gz
    ```

4. **Navigate to the ns-3 directory**:
    ```bash
    cd ns-3.35
    ```

### Step 3: Install Dependencies

Make sure all required dependencies are installed. As you don't have `sudo` access, you will need to install dependencies in your Conda environment.

1. **Install the required dependencies**:
    ```bash
    conda install -c conda-forge gcc gxx make cmake pybindgen pygccxml
    ```

2. **Optional dependencies** (for specific ns-3 modules):
    - For Wi-Fi, LTE, and other models, you may need additional libraries like `libpcap-dev`, `libxml2-dev`, `libsqlite3-dev`, etc. Check [ns-3 requirements](https://www.nsnam.org/wiki/Installation) for more details.

### Step 4: Build ns-3

1. **Configure the build**:
    Ensure you enable Python bindings, examples, and tests during the configuration process.
    ```bash
    ./waf configure --enable-python-bindings --enable-examples --enable-tests
    ```

2. **Build ns-3**:
    ```bash
    ./waf
    ```

    This may take some time depending on your machine’s capabilities.

### Step 5: Set Up Python Bindings

Ensure that Python bindings for `ns-3` are correctly installed.

1. **Enable Python bindings** in the build:
    ```bash
    ./waf configure --enable-python-bindings
    ```

2. **Build the Python bindings**:
    ```bash
    ./waf build
    ```

3. **Install Python bindings**:
    Once the build is complete, the Python bindings will be installed in the `build/bindings/python` directory.

### Step 6: Set Environment Variables

To ensure that `ns-3` can be accessed globally from any directory within your Conda environment, you need to modify the environment variables.

#### 1. Edit `env_vars.sh`

The `env_vars.sh` file is executed when you activate the Conda environment, so add the necessary environment variable settings to this file.

1. **Edit the `env_vars.sh` file**:
    ```bash
    nano ~/miniconda3/envs/airsim/etc/conda/activate.d/env_vars.sh
    ```

2. **Add the following configuration**:

    ```bash
    # Set PYTHONPATH to include ns-3 bindings
    export PYTHONPATH=$PYTHONPATH:/student/yat251/ns-3-allinone/ns-3.35/build/bindings/python

    # Set LD_LIBRARY_PATH to include ns-3 libraries
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/student/yat251/ns-3-allinone/ns-3.35/build/lib

    # Add ns-3 build directory to PATH for running waf and scripts from anywhere
    export PATH=$PATH:/student/yat251/ns-3-allinone/ns-3.35/build
    ```

3. **Save and close the file**.

4. **Activate your Conda environment** to apply the changes:
    ```bash
    conda deactivate
    conda activate airsim
    ```

#### 2. (Optional) Modify `.bashrc` for Global Access

If you want these variables to be available globally (outside of Conda), you can modify your `~/.bashrc`.

1. **Edit `.bashrc`**:
    ```bash
    nano ~/.bashrc
    ```

2. **Add the same environment variable settings** at the end of the file:

    ```bash
    export PYTHONPATH=$PYTHONPATH:/student/yat251/ns-3-allinone/ns-3.35/build/bindings/python
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/student/yat251/ns-3-allinone/ns-3.35/build/lib
    export PATH=$PATH:/student/yat251/ns-3-allinone/ns-3.35/build
    ```

3. **Save and close the file**.

4. **Reload `.bashrc`**:
    ```bash
    source ~/.bashrc
    ```

## Edge Cases

- **Missing Dependencies**: If you get errors related to missing libraries or headers, ensure all required dependencies are installed and available in your Conda environment.
- **Outdated Python Version**: Ensure you are using Python 3.10 or higher, as `ns-3` bindings are built for newer Python versions.
- **Compilation Errors**: If the build fails due to missing or incorrect compilers, check if your Conda environment includes the correct versions of `gcc` and `g++`.
- **Incompatible Libraries**: If certain `ns-3` modules require additional system libraries (like `libpcap`), you may need to install these libraries manually or in your Conda environment.

## Testing

To verify that everything is working:

1. **Test Python bindings**:
    ```bash
    python -c "import ns.core; print(ns.core.Simulator)"
    ```

2. **Run an ns-3 example**:
    ```bash
    python /student/yat251/ns-3-allinone/ns-3.35/examples/tutorial/first.py
    ```

## Running Examples

After setup, you can run `ns-3` examples from any directory in your Conda environment.

1. **Navigate to the `examples/tutorial` directory**:
    ```bash
    cd /student/yat251/ns-3-allinone/ns-3.35/examples/tutorial
    ```

2. **Run the `first.py` example**:
    ```bash
    python first.py
    ```

You can also run `waf` commands from any directory to test the network simulation:

```bash
./waf --run scratch/my_simulation

