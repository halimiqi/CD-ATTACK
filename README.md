# CD-ATTACK
The Implemention of paper "Adversarial Attack on Community Detection by Hiding Individuals". It is accepted by The Web Conference 2020.

![CD-ATTACK](https://github.com/halimiqi/CD-ATTACK/blob/master/cdattack.png)  

## Usage

To run the CD-ATTACK model, please run the *main.py* as `python main.py`

The baselines proposed in the paper is implemented in *baselines.py*. `python baselines.py` to run it.

The default dataset is dblp with fixed target users. For other changeable parameters are shown in *main.py* 
 
## Enviorment
The model is implemented based on python=3.6.7 and tensorflow =1.15. Other requirements of the enviorment is listed in *requirements.txt*.

## Setting
The code is training on Nvidia-TitanX GPU with 12 Gb RAM. The CPU is i7-7800X and the memory is 64Gb. This is not the minimum required setting for this project. Other hardware setting may also feasible for this implemention.



