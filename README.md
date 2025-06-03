V2X-Cooperative-Perception
======

This is the official implementation of paper. "When Autonomous Vehicle Meets V2X Cooperative Perception: How Far Are We?".

This repository contains a testing framework for Cooperative-Perception termed `V2X-Cooperative-Perception`, for evaluate the impact of imperfect cooperative perception on safety violations in autonomous driving.

#### The structure of the repository
```
V2X-Cooperative-Perception
├── RQ3
├── RQ4
├── scripts
├── simulation
│   ├── codriving
│   ├── common
│   ├── leaderboard
│   ├── opencood
│   └── scenario_runner
├── spconv
├── README.md
└── requirements.txt
```

## Installation

#### Our working environment
- Ubuntu 20.04
- Python 3.7
- CMake 3.22.1
- PyTorch 1.10.1
- CUDA 11.3
- Carla 0.9.10.1

#### Dependency repository

- [CARLA 0.9.10.1](https://github.com/carla-simulator/carla)

CARLA is an open source simulator for autonomous driving research, which is used for online evaluation of collaborative perception models.

Installation of CARLA 0.9.10.1:
```
mkdir simulation/carla_root
cd simulation/carla_root
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.10.1.tar.gz
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.10.1.tar.gz
tar -zxvf CARLA_0.9.10.1.tar.gz
tar -zxvf AdditionalMaps_0.9.10.1.tar.gz
rm simulation/carla_root/*.tar.gz

# Install Carla's Python egg package
easy_install PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
cd ../..
```

- [Spconv](https://github.com/traveller59/spconv)

Spconv is a spatially sparse convolution library for generate voxel features in perception module. Please [click here]( https://github.com/traveller59/spconv/tree/v1.2.1) for the installation of Spconv (1.2.1).

- [Opencood](https://github.com/DerrickXuNu/OpenCOOD)

OpenCOOD is an Open Cooperative Detection framework for autonomous driving that allows users to train and test various collaborative perception models.

Installation of OpenCOOD :
```
# Install the requirements.
python simulation/opencood/setup.py develop
pip install -r simulation/opencood/requirements.txt

# Install bbx nms calculation cuda version
python simulation/opencood/utils/setup.py build_ext --inplace
```

Other python package :
```
pip install -r simulation/requirements.txt
```

## Usage

#### Checkpoints
Download Pre-trained Collaborative Perception Model : https://huggingface.co/gjliu/v2xverse

`checkpoints` folder tree :
```
checkpoints
├── early_fusion
│   ├── perception
│   └── planner
├── late_fusion
│   ├── perception
│   └── planner
├── fcooper
│   ├── perception
│   └── planner
├── v2xvit
│   ├── perception
└── └── planner
```

#### Before RQ3 / RQ4
This step will run the carla server to start the online evaluation.
```
bash scripts/running_carla.sh
```

#### For RQ3
RQ3 seeks to evaluate  the relationship between imperfect cooperative perception errors and driving violations.

Output the result `results/(early_fusion, late_fusion, fcooper, v2xvit)`
```
# All models and routes running RQ3
bash RQ3/rq3_integrated.sh

# Run a single sample
bash RQ3/rq3.sh 0 40000 early_fusion early_5_10 1
```
The parameters of `RQ3/rq3.sh` :
```
usage: RQ3/rq3.sh ${Route_id} ${Carla_port} ${CP_model} ${Agent_config} ${Scenario_config}

optional arguments:
  ${Route_id}         ID of evaluation route
  ${Carla_port}       Port number of carla server
  ${CP_model}         Evaluated collaborative perception model
  ${Agent_config}     Agent configuration of CP model
  ${Scenario_config}  Configuration of test scenarios
```

#### For RQ4
RQ4 seeks to evaluate the extent to which communication issues (e.g., time delay and positioning error) encountered during online deployment diminish the effectiveness of cooperative perception systems.

Output the `results_(latency, noise)/(early_fusion, late_fusion, fcooper, v2xvit)`
```
# Imitates the network communication delay.
bash RQ4/rq4_latency.sh 0 40000 early_fusion early_5_10 1

# Imitates the positioning error of agents.
bash RQ4/rq4_noise.sh 0 40000 early_fusion early_5_10 1
```

## Acknowledgements
This project makes use of the following open-source projects:

- [Carla leaderboard](https://github.com/carla-simulator/leaderboard)
- [Scenario runner](https://github.com/carla-simulator/scenario_runner)
- [Opencood](https://github.com/DerrickXuNu/OpenCOOD)
- [V2Xverse](https://github.com/CollaborativePerception/V2Xverse)

We sincerely thank the authors for their contributions to the community.