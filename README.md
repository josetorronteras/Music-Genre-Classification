<br>
<h1 align="center">Music Genre Classification </h2>

[![CodeFactor](https://www.codefactor.io/repository/github/xexuew/music-genre-classification/badge)](https://www.codefactor.io/repository/github/xexuew/music-genre-classification)

https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/xexuew/Music-Genre-Classification/master/results/embedding-projector/project_config.json

<h2> Setup </h2>
<h3> Anaconda </h3>

```bash
$ git clone https://github.com/xexuew/Music-Genre-Classification.git .
$ cd Music-Genre-Classification
$ wget http://opihi.cs.uvic.ca/sound/genres.tar.gz -P data/
$ tar -xvzf data/genres.tar.gz -C data/
$ docker-compose up anaconda

Open http://127.0.0.1:8888/
```

If you prefer to build the image locally
```bash
$ Change --> image: joseew/music-genre-classification_anaconda in docker-compose.yml
$ To --> build: ./docker/Anaconda
```

<h3> Floydhub </h3>
First is necessary to  get an api key from here: https://www.floydhub.com/settings/apikey

```bash
$ docker-compose run --rm floydhub
$ cd project/
$ floyd login -k TOKEN
$ floyd run --task
```

$ tensorboard --logdir="logs/"

floyd run --gpu --env tensorflow-1.13.1 --data joseew/datasets/spec-dataset/1:input 'python train_cnn.py --config="config/config-floyd.ini"'

python source/main.py --trainmodel=cnn --model=/Users/josetorronteras/Code/Music-Genre-Classification/data/models/CNN/model_v3_8.json --config=config/config-gpu.ini


python preprocess.py --preprocess=spec --config=config/config-gpu.ini
python preprocess.py --preprocess=mfcc --config=config/config-gpu.ini
python dataset.py --dataset=spec --config=config/config-gpu.ini
python dataset.py --dataset=mfcc --config=config/config-gpu.ini

floydhub:
    build:
        context: ./docker/Floydhub
        args:
            FLOYDHUB_API_KEY: ${FLOYDHUB_API_KEY}
    command: "/bin/bash"
    volumes:
        - ./project:/code/project
        - ./data:/code/data
