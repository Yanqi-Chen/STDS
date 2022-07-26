# State Transition of Dendritic Spines Improves Learning of Sparse Spiking Neural Networks
This repo contains the code reproducing the results of **STDS** (**S**tate **T**ransition of **D**endritic **S**pines)  in this paper, which is modified based on the open-source code of [SEW ResNet](https://github.com/fangwei123456/Spike-Element-Wise-ResNet).

- [Directory Tree](#directory-tree)
- [Dependency](#dependency)
- [Environment](#environment)
- [Usage](#usage)
- [Citation](#citation)

## Directory Tree

```
.
├── CIFAR10
│   ├── model.py
│   ├── optim.py
│   ├── train.py
│   └── logs
└── ImageNet
    ├── optim.py
    ├── sew_resnet.py
    ├── train.py
    ├── utils.py
    └── logs
        ├── linear
        └── sine
```

## Dependency 

The major dependencies of this code are list as below

```
# Name                    Version
cudatoolkit               10.2.89
cudnn                     8.2.1.32
cupy                      9.6.0
numpy                     1.21.4
python                    3.7.11 
pytorch                   1.9.1
spikingjelly              <Specific Version>
tensorboard               2.7.0
torchvision               0.10.1
```

**Note**: the version of spikingjelly will be clarified in [usage](##Usage) part.

## Environment

The running of code requires NVIDIA GPU and has been tested on *CUDA 10.2* and *Ubuntu 16.04*. The hardware platform used in experiments is shown below.

- GPU: Tesla *V100-SXM3-32GB* 350 Watts version
- CPU: Intel(R) Xeon(R) Platinum 8168 CPU @ 2.70GHz

Each trial on ImageNet requires 8 GPUs. For CIFAR-10, each trial requires only a single GPU.

## Usage

This code requires a specified version of an open-source SNN framework **SpikingJelly**. To get this framework installed, first clone the repo from [GitHub](https://github.com/fangwei123456/spikingjelly):

```bash
$ git clone https://github.com/fangwei123456/spikingjelly.git
```

Then, checkout the version we use in these experiments and install it.

```bash
$ cd spikingjelly
$ git checkout d8cc6a5
$ python setup.py install
```

With dependency mentioned above installed, you should be able to run the following commands:

### ImageNet

#### Dense training:

```shell
$ python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --cos_lr_T 320 --model sew_resnet18 -b 32 --output-dir <log dir> --tb --print-freq 4096 --amp --connect_f ADD --T 4 --lr 0.1 --epoch 320 --data-path <dataset path> --sparse-function identity
```

#### Our proposed algorithm:

```shell
$ python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --cos_lr_T 320 --model sew_resnet18 -b 32 --output-dir <log dir> --tb --print-freq 4096 --amp --connect_f ADD --T 4 --lr 0.1 --epoch 320 --data-path <dataset path> --sparse-function stmod --flat-width <D> --gradual <scheduler type>
```

#### Grad R:

```shell
$ python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --cos_lr_T 320 --model sew_resnet18 -b 32 --output-dir <log dir> --tb --print-freq 4096 --amp --connect_f ADD --T 4 --lr 0.1 --epoch 320 --alpha-gr <alpha in Grad R> --data-path <dataset path> --sparse-function stmod --flat-width <mu in Grad R>
```

The TensorBoard logs and checkpoints will be placed in two separate directories in `./logs`.

#### Running Arguments

| Arguments         | Descriptions                                               | Value                                                    | Type  |
| ----------------- | ---------------------------------------------------------- | -------------------------------------------------------- | ----- |
| --cos_lr_T        | Total steps of Cosine Annealing scheduler of learning rate | 320                                                      | int   |
| -b,--batch-size   | Training batch size                                        | 32                                                       | int   |
| --alpha-gr        | Hyperparameter $\alpha$ in Grad R                          | None                                                     | float |
| --data-path       | Path of datasets                                           |                                                          | str   |
| --output-dir      | Path for dumping models and logs                           |                                                          | str   |
| --print-freq      | Frequency of print of status during training               | 4096                                                     | int   |
| --amp             | Whether to use mixed precision training                    |                                                          | bool  |
| --connect_f       | Connection function of SEW ResNet                          | ADD                                                      | str   |
| -T                | Simulation time-steps of SNNs                              | 4                                                        | int   |
| --lr              | Learning rate                                              | 0.1                                                      | float |
| --epoch           | Number of training epochs                                  | 320                                                      | int   |
| --sparse-function | Reparameterization function                                | 'stmod' for pruning, 'identity' for training dense model | str   |
| --flat-width      | Hyperparameter $D$ in our work and $\mu$ in Grad R         |                                                          | float |
| --gradual         | Scheduler type                                             | 'sine', 'linear'                                         | str   |

### CIFAR-10

#### Dense training:

```sh
$ python train.py --dataset-dir <dataset path> --dump-dir . --sparse-function identity --amp
```

#### Our proposed algorithm:

```sh
$ python train.py --dataset-dir <dataset path> --dump-dir . --sparse-function stmod --gradual <scheduler type> --flat-width <D> --amp
```

#### Running Arguments

| Arguments         | Descriptions                                       | Value                                                    | Type  |
| ----------------- | -------------------------------------------------- | -------------------------------------------------------- | ----- |
| -b, --batch-size  | Training batch size                                | 16                                                       | int   |
| --lr              | Learning rate                                      | 1e-4                                                     | float |
| --dataset-dir     | Path of datasets                                   |                                                          | str   |
| --dump-dir        | Path for dumping models and logs                   |                                                          | str   |
| -T                | Simulation time-steps of SNNs                      | 8                                                        | int   |
| -N, --epoch       | Number of training epochs                          | 2048                                                     | int   |
| -test             | Whether test only                                  |                                                          | bool  |
| --amp             | Whether to use mixed precision training            |                                                          | bool  |
| --sparse-function | Reparameterization function                        | 'stmod' for pruning, 'identity' for training dense model | str   |
| --flat-width      | Hyperparameter $D$ in our work and $\mu$ in Grad R |                                                          | float |
| --gradual         | Scheduler type                                     | 'sine', 'linear'                                         | str   |

## Citation

Please refer to the following citation if this work is useful for your research.

```
@InProceedings{pmlr-v162-chen22ac,
  title = 	 {State Transition of Dendritic Spines Improves Learning of Sparse Spiking Neural Networks},
  author =       {Chen, Yanqi and Yu, Zhaofei and Fang, Wei and Ma, Zhengyu and Huang, Tiejun and Tian, Yonghong},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {3701--3715},
  year = 	 {2022},
  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v162/chen22ac/chen22ac.pdf},
  url = 	 {https://proceedings.mlr.press/v162/chen22ac.html}
}
```
