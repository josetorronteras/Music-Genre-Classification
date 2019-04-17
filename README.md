# Music-Genre-Classification

git clone -b config-cnn https://github.com/xexuew/Music-Genre-Classification.git
wget http://opihi.cs.uvic.ca/sound/genres.tar.gz -P data/
tar -xvzf data/genres.tar.gz -C data/

python source/main.py --preprocess --config="config/config-gpu.ini"
python source/main.py --dataset --config="config/config-gpu.ini"
python source/main.py --trainmodel --config="config/config-gpu.ini" --model="data/models/model_10.json" --device=0