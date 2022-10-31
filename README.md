# DataLabAssignement2
Example project for Assignement 2 : Generative model

* Create a python environement. 

  * On the GTX1080 :
    > virtualenv venv1 --python=python3.9
 
  * On the RTXA6000 :
    > virtualenv venv2 --python=python3.9

* Then activate the environement:
  * On the GTX1080 :
    > source venv1/bin/activate 

  * On the RTXA6000 :
    > source venv2/bin/activate 
* Install PyTorch package:
  * On the GTX1080 :
    > pip3 install torch torchvision torchaudio 

  * On the RTXA6000 :
    > pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

* Install the required packages:

  * On the GTX1080 :
    > pip3 install -r requirements.txt

  * On the RTXA6000 :
    > pip3 install -r requirements2.txt




***
To train a model :

> python train.py --epochs 600 

To generate :

> python generate.py 


