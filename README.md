This is a tensorflow implementation of visual prediction of typhoon clouds with hierarchical generative adversarial networks.

Requirements:
Python 2.7
Tensorflow 1.2.0

Test the model I uploaded:
python avg_runner -n name -l (xx/xx)/HGAN_TyphoonCloudVisualPrediction/Save/Models/model.ckpt -T
Train a new model:
python avg_runner.py -n name -s steps

Training data:/Data/Train/xx/
Testing data:/Data/Test/xx/

The models trained and generated images will be saved in '/Save/..'

The following example shows continuous typhoon cloud evolution in ten nearest time steps. 
https://github.com/lihuiupc/HGAN_TyphoonCloudVisualPrediction/blob/master/generated_1second.gif

Thanks for the code ''https://github.com/dyelax/Adversarial_Video_Generation''.
