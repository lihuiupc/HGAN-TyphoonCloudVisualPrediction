This is a tensorflow implementation of visual prediction of typhoon clouds with hierarchical generative adversarial networks.

Requirements:
Python 2.7
Tensorflow 1.2.0

You can download the pretrained model here https://pan.baidu.com/s/1wBwbojKv0SIZh-Le8w5CtQ. The password is rvia.
The downloaded three files should be put in 'HGAN_TyphoonCloudVisualPrediction/Save/Models/'.
Test the model:
python avg_runner -n name -l (xx/xx)/HGAN_TyphoonCloudVisualPrediction/Save/Models/model.ckpt -T
Train a new model:
python avg_runner.py -n name -s steps

Training data:/Data/Train/xx/
Testing data:/Data/Test/xx/

The models trained and generated images will be saved in '/Save/..'

The following example shows continuous typhoon cloud evolution in ten nearest time steps. 
![image]( https://github.com/lihuiupc/HGAN_TyphoonCloudVisualPrediction/blob/master/generated_1second.gif)
![image].<img src="https://github.com//lihuiupc/HGAN_TyphoonCloudVisualPrediction/blob/master/generated_1second.gif" width="300" height="450" />
Thanks for the code ''https://github.com/dyelax/Adversarial_Video_Generation''.
