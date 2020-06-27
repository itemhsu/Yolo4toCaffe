export PYTHONPATH=/home/itemhsu/src/py/MobileNet-YOLO/python/
cat yolov4-custom_leaky.weights_0* > yolov4-custom_leaky.weights
python darknet2caffe.py yolov4-custom_leaky.cfg yolov4-custom_leaky.weights model/yolov4-custom_leaky.prototxt model/yolov4-custom_leaky.caffemodel
python matthew_y4_detect_one.py
