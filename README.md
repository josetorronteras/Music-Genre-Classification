<br>
<h2 align="center">Music Genre Classification </h2>

https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/xexuew/Music-Genre-Classification/master/results/embedding-projector/project_config.json


```bash

$ conda create --name music-genre-classification
$ conda activate music-genre-classification
$ Instalar dependecias ¿pip3 install -r requirements.txt?
$ git clone https://github.com/xexuew/Music-Genre-Classification.git .
$ cd Music-Genre-Classification
$ wget http://opihi.cs.uvic.ca/sound/genres.tar.gz -P data/
$ tar -xvzf data/genres.tar.gz -C data/
$ python source/main.py --preprocess --config="config/config-gpu.ini"
$ python source/main.py --dataset --config="config/config-gpu.ini"
$ python source/main.py --trainmodel --config="config/config-gpu.ini" --model="data/models/model_10.json" --device=0
$
$ tensorboard --logdir="logs/"
```


python source/main.py --preprocess=spec --config=config/config-gpu.ini
python source/main.py --preprocess=mfcc --config=config/config-gpu.ini

python source/main.py --dataset=spec --config=config/config-gpu.ini
python source/main.py --dataset=mfcc --config=config/config-gpu.ini

python source/main.py --trainmodel=cnn --model=/Users/josetorronteras/Code/Music-Genre-Classification/data/models/CNN/model_v3_8.json --config=config/config-gpu.ini
