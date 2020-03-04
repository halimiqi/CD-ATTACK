# CD-ATTACK
The Implemention of paper "Adversarial Attack on Community Detection by Hiding Individuals". It is accepted in 2014

![CD-ATTACK](/home/picture/1.png)
##Usage

To run the CD-ATTACK model, please run the *main.py* as `python main.py`

The baselines proposed in the paper is implemented in baselines.py. Run `python baselines.py` to run it.

The default dataset is dblp with fixed target users. For other changeable parameters are shown in *main.py* 
 
## Enviorment
The model is implemented based on python=3.6.7 and tensorflow =1.15. Other requirements of the enviorment is listed in *requirements.txt*.

## Setting
The code is training on Nvidia-TitanX GPU with 12 Gb RAM. The CPU is i7-7800X and the memory is 64Gb. This is not the minimum setting of PC. Other hardware may also feasible for this implemention.



