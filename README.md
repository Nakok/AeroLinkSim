# AirSim Drone Network Simulation

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Build](https://img.shields.io/badge/build-passing-brightgreen) ![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-orange)


![AeroLinkSim Logo](images/logo2.webp)

# <span style="font-family: 'Arial Black', sans-serif; color: #0078D7;">Welcome to AeroLinkSim! 🚁</span>

> <span style="font-style: italic; font-size: 18px; color: #444444;">"Where innovation takes flight—simulating the skies, one link at a time."</span>

---

## <span style="color: #009688;">What is AeroLinkSim?</span>

AeroLinkSim is a cutting-edge simulation environment powered by AirSim, designed to push the boundaries of drone autonomy, connectivity, and real-world network challenges. From latency to packet loss, AeroLinkSim ensures every simulation feels as real as the sky above. 🌐✈️

---



## Table of Contents
- [Project Overview](#project-overview)
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Architecture](#architecture)
- [Installation Guide](#installation-guide)
  - [Setting Up the Environment](#setting-up-the-environment)
  - [Dependencies](#dependencies)
  - [AirSim Setup](#airsim-setup)
  - [ZeroMQ Setup](#zeromq-setup)
- [Usage Instructions](#usage-instructions)
  - [Configuration Options](#configuration-options)
- [Project Demo](#project-demo)
- [Performance Metrics](#performance-metrics)
- [Project Goals](#project-goals)
- [Roadmap](#roadmap)
- [FAQs](#faqs)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Project Overview
This project integrates AirSim with a dynamic network simulation framework designed for drone-based applications. The simulation models real-world network conditions such as latency, jitter, and packet loss, which are essential for testing drone performance in complex environments. In addition, it features ZeroMQ integration to enable efficient message exchange between drones and control systems, ensuring real-time communications.

## Introduction
### What is AirSim?
AirSim is an open-source simulator developed by Microsoft for testing autonomous systems. It offers a realistic simulation environment for a variety of applications, such as drone flight, computer vision, and machine learning.

### What does this project do?
This project combines AirSim with network simulation tools to test the behavior of drones under various network conditions. It integrates with ZeroMQ for message handling and uses dynamic network parameters (latency, jitter, packet loss) to simulate real-world communication challenges.

### Key Features:
- **Dynamic Network Simulation**: Simulate latency, jitter, and packet loss to model real-world network conditions.
- **ZeroMQ Integration**: Enables reliable message exchange between drones and control systems.
- **Customizable Environments**: Adjust simulation parameters to mimic different drone scenarios.

The goal of this project is to provide an end-to-end simulation environment where drone performance can be tested under controlled network conditions before deployment in real-world scenarios.

## Prerequisites
### System Requirements
| Requirement             | Version          |
|-------------------------|------------------|
| Operating System        | Linux (Ubuntu)   |
| AirSim                  | 1.8.1            |
| Python                  | 3.10 or later    |
| Conda                   | Latest           |

### Conda Setup
1. Initialize Conda:
   ```bash
   CONDA_BASE=$(conda info --base)
   source $CONDA_BASE/etc/profile.d/conda.sh
   ```
2. Create and activate a new Conda environment for the project:
   ```bash
   conda create -n airsim python=3.10
   conda init
   conda activate airsim
   ```
3. Install necessary Python libraries:
   ```bash
   pip install numpy msgpack-rpc-python airsim torch pillow tensorboard
   ```

### Additional Tools
| Tool     | Description                                             |
|----------|---------------------------------------------------------|
| ZeroMQ   | Messaging library for scalable, distributed systems.    |
| AirSimNH | AirSim’s custom version tailored for network simulation.|

## Architecture
This project has a modular architecture with the following components:

- **AirSim Interface**: Handles drone control and interaction with the simulation environment.
- **Network Simulation Module**: Introduces network-related disturbances (latency, jitter, packet loss) into the communication channel between the control system and the drone.
- **ZeroMQ Integration**: A messaging system for sending and receiving commands between the drone and the controller, abstracting network complexities.
- **Control and Monitoring System**: Monitors the drone’s status and adjusts parameters in real-time based on network performance.

### Diagram
```
+-------------------+        +-------------------------+
|  AirSim Interface |<------>|  Control & Monitoring   |
|    (Drone)        |        |      System             |
+-------------------+        +-------------------------+
               ^                          ^
               |                          |
               |                          |
        +----------------+        +-----------------+
        | Network        |        | ZeroMQ Messaging |
        | Simulation     |<------>|  (Send/Receive)  |
        | (Latency,      |        |                 |
        | Jitter,        |        +-----------------+
        | Packet Loss)   |
        +----------------+
```

## Installation Guide
### Setting Up the Environment
1. Install Conda (if not installed):
   Follow the instructions on Anaconda's website to install Conda.

2. Create and activate the Conda environment:
   ```bash
   conda create -n airsim python=3.10
   conda activate airsim
   ```

### Dependencies
Install the required dependencies:
```bash
pip install numpy msgpack-rpc-python airsim torch pillow tensorboard
```

### AirSim Setup
1. Download AirSim:
   ```bash
   wget https://github.com/microsoft/AirSim/releases/download/v1.8.1/AirSimNH.zip
   ```
2. Extract AirSim:
   ```bash
   unzip ./AirSimNH.zip
   rm -f AirSimNH.zip
   ```
3. Configure AirSim script:
   ```bash
   sed '5s/$/ -graphicsadapter=1/' ./AirSimNH/LinuxNoEditor/AirSimNH.sh > file.tmp
   chmod 740 file.tmp
   mv file.tmp ./AirSimNH/LinuxNoEditor/AirSimNH.sh
   ```
4. Run AirSim:
   ```bash
   ./AirSimNH/LinuxNoEditor/AirSimNH.sh
   ```

### ZeroMQ Setup
1. Install the ZeroMQ library:
   ```bash
   pip install pyzmq
   ```
2. Integrate ZeroMQ messaging into your drone and control code. Refer to the `src/zmq_integration.py` file for details.

## Usage Instructions
### Start the AirSim Simulator
Follow the steps in the [AirSim Setup](#airsim-setup) section to start the simulator.

### Run the Control Script
Use the control script to manage drone behavior while simulating real-world network conditions.

Example:
```bash
python control_drone.py
```

### Adjust Network Simulation Parameters
Modify the network parameters (latency, jitter, packet loss) in the `config.yaml` file or via CLI.

Example (`config.yaml`):
```yaml
network_simulation:
  latency: 50 # ms
  jitter: 10   # ms
  packet_loss: 5  # Percentage
```

### Monitor and Analyze
Utilize the built-in logging and telemetry features to monitor drone performance and analyze the effects of network conditions.

## Project Demo
![Demo GIF](https://via.placeholder.com/800x400?text=Demo+coming+soon)

## Performance Metrics
Evaluate drone performance under different conditions:
- Latency impact on control precision.
- Jitter effects on stability.
- Packet loss rates and their correlation with failure scenarios.

Results can be logged using TensorBoard for visualization.

## Project Goals
- Create a realistic testing environment for drones.
- Facilitate research in network-aware autonomous systems.
- Provide a foundation for real-world deployment scenarios.

## Roadmap
- [x] Integrate AirSim and ZeroMQ.
- [x] Dynamic network simulation.
- [ ] Add support for multi-drone simulations.
- [ ] Implement advanced PPO training workflows.
- [ ] Enhance visualization of performance metrics.

## FAQs
**Q: Can this project be run on Windows?**
A: The current setup is tested on Ubuntu. Windows compatibility may require additional configuration.

**Q: How do I modify the PPO agent's parameters?**
A: Adjust the parameters in the `ppo_config.yaml` file located in the `configs` directory.

**Q: Where are the logs stored?**
A: Logs are stored in the `logs/` directory and can be visualized using TensorBoard.

## Contributing
We welcome contributions to improve and extend this project. Here are some ways you can contribute:

- Report bugs or issues.
- Submit pull requests for bug fixes or new features.
- Suggest improvements to the documentation.

Please ensure that all contributions follow the project's coding standards and include adequate tests for new features.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- **AirSim**: Thanks to Microsoft for providing an excellent open-source drone simulation environment.
- **ZeroMQ**: A powerful messaging system that helped build the communication backbone for this project.
- **Contributors**: Thanks to everyone who contributed to making this project a success.

