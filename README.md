# CD-ATTACK

The Implemention of paper "Adversarial Attack on Community Detection by Hiding Individuals"<sup>[1]</sup>. It is accepted by The Web Conference 2020.

![CD-ATTACK](https://github.com/halimiqi/CD-ATTACK/blob/master/cdattack.png)  

## Usage

To train the CD-ATTACK model, please run the *main.py* as `python main.py`

To restore a trained model, the command is `python main.py --test --trained_our_path [THE CHECKPOINT NAME]`

The checkpoint name is formated as the string of the time point of starting the training process. eg. 
`python main.py --test --trained_our_path 200307133445` .The checkpoints will be recorded automatically for every training process. And the checkpoints files are placed in directory checkpoints/

The default dataset is dblp with fixed target users. To change the other dataset or modify other changeable parameters, please run `python main.py -h` to see the details.
 
## Environment
The model is implemented based on python=3.6.7 and tensorflow =1.13. Other requirements of the enviorment is listed in *requirements.txt*.

## Setting
The code is training on Nvidia-TitanX GPU with 12 Gb RAM. The CPU is i7-7800X and the memory is 64Gb. This is not the minimum required setting for this project. Other hardware setting may also feasible for this implemention.

This work is collaborated by researchers from the Chinese University of Hong Kong, Georgia Institute of Technology and Tencent AI lab.

---
[1] Li, Jia, et al. "Adversarial Attack on Community Detection by Hiding Individuals."
In Proceedings of the ACM International World Wide Web Conference (WWW 2020). 

