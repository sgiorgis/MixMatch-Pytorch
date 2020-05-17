# MixMatch - A Holistic Approach to Semi-Supervised Learning - Pytorch
Reproduction of "MixMatch - A Holistic Approach to Semi-Supervised Learning" by David Berthelot, Nicholas Carlini, Ian Goodfellow, Nicolas Papernot, Avital Oliver and Colin Raffel in Pytorch.

[Github] | [Paper]

Installation
------------
Our project uses the python installer for Python packages pip. Refer to the [documentation](https://pip.pypa.io/en/stable/installing/) for instructions on how to download it. 
Once you have downloaded pip, you can install all dependencies with 
```
pip install -r requirements.txt
```

#### Train
The train needs an configuration file to run. You can find all the configuration files we used to reproduce all the experiments in 
./experiments folder. An example run with the example ./experiments/config.yml is  
```
python train.py --config=experiments/config.yml 
```
#### Validation
You can validate pre trained models or validate a specific trained model stored in checkpoints directory with
```
python test.py --checkpoint_file=checkpoints/<path_to_checkpoint_file> --datase_name=CIFAR10 
```
or 
```
python test.py --checkpoint_file=checkpoints/<path_to_checkpoint_file> --datase_name=SVHN 
```

You can also validate more checkpoints at the same time by providing a path to the checkpoints file with
```
python test.py --checkpoint_dir=checkpoints/<path_to_checkpoint_file> --datase_name=SVHN 
```
## Experiments
The below experiments were performed with 1024 iterations and only 200 epochs due to limitations in our resources. 
Training the same model with supervised learning on the entire 50000-example training set achieved
an error rate of 4.33%. The error rates for CIFAR-10 are presented below.

|Methods/Labels  | 250   | 500 | 1000 | 2000 | 4000
|:---------------|:------|:----|:-----|:-----|:----:|
| MixUp          | 47.55 | 36.09 | 25.21 | 20.09 | 14.49 |
| Pseudo-Label   | 49.88 | 40.65 | 33.77 |  22.02 | 15.95 |
| MixMatch       | 12.25 | 10.48 | 10.24 | 8.91 | 6.69 |

## Acknowledgments
This work was initiated as a project of our master's level course titled 'Deep Learning in Data Science' @ KTH Stockholm, Sweden. 
We would like to thank the course staff for providing us with the necessary Google Cloud tickets to run our experiments.

[Paper]: https://arxiv.org/abs/1905.02249
[Github]: https://github.com/google-research/mixmatch

