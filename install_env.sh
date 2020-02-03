# 0. activate env 
pip install pandas numpy tensorflow-==1.15.*
# 1. download models
git clone https://github.com/tensorflow/models.git
# 2. install packages
pip install --user Cython
pip install --user contextlib2
pip install --user pillow
pip install --user lxml
pip install --user jupyter
pip install --user matplotlib
# 3. download protoc 
cd models/research
wget http://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
unzip -d protoc protoc-3.0.0-linux-x86_64.zip
rm protoc-3.0.0-linux-x86_64.zip
# 4. protoc compile 
./protoc/bin/protoc object_detection/protos/*.proto --python_out=.
# 5. export pythonpath
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
# 6. test
python object_detection/builders/model_builder_test.py
# 7. manager images
python xml_to_csv.py ./images/
python generate_tfrecord.py --csv_input=`pwd`/images/test_labels.csv --output_path=`pwd`/images/test_labels.record
python generate_tfrecord.py --csv_input=`pwd`/images/train_labels.csv --output_path=`pwd`/images/train_labels.record
# 8. ready pipeline config
python models/research/object_detection/legacy/train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v2_quantized_300x300_coco.config
