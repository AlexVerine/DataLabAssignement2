# DataLabAssignement2
Example project for Assignement 2 : Generative model

* Create a python environement. 
    > virtualenv venv --python=python3.9

* Then activate the environement:
    > source venv/bin/activate 

* Install PyTorch package:
    > pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

* Install the required packages:
    > pip3 install -r requirements2.txt




***

The pretrained model is the folder /checkpoints/.
To generate :

> python generate.py 


If you want to retrain the model :

> python train.py --epochs 50 
