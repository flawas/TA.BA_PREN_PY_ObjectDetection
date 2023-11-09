import object_detection
import os
import wget
import shutil
import pandas as pd
import numpy as np
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import cv2
import numpy as np
from matplotlib import pyplot as plt


CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
        'WORKPACE_PATH': os.path.join('Tensorflow', 'workspace'),
        'SCRIPTS_PATH': os.path.join('Tensorflow', 'scripts'),
        'APIMODEL_PATH': os.path.join('Tensorflow','models'),
        'ANNOTATION_PATH': os.path.join('Tensorflow','workplace','annotations'),
        'IMAGE_PATH': os.path.join('Tensorflow','workplace','images'),
        'MODEL_PATH': os.path.join('Tensorflow','workplace','models'),
        'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow','workplace','pre-trained-models'),
        'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workplace','models', CUSTOM_MODEL_NAME),
        'OUTPUT_PATH': os.path.join('Tensorflow', 'workplace', 'models', CUSTOM_MODEL_NAME, 'export'),
        'TFJS_PATH': os.path.join('Tensorflow','workplace','models', CUSTOM_MODEL_NAME, 'tfjexport'),
        'TFLITE_PATH': os.path.join('Tensorflow', 'workplace', 'models', CUSTOM_MODEL_NAME, 'tfliteexport'),
        'PROTOC_PATH': os.path.join('Tensorflow','protec')
}  

files = {
        'PIPELINE_CONFIG': os.path.join('Tensorflow', 'workplace', 'models', CUSTOM_MODEL_NAME, 'pipeline.config'),
        'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME),
        'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

for path in paths.values():
    if not os.path.exists(path):
        if os.name == 'nt':
            !mkdir {path}

if not os.path.exists(os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection')):
    wget.download('https://github.com/tensorflow/models/archive/refs/heads/master.zip', paths['APIMODEL_PATH']+'/models.zip')

if os.name == 'nt':
    wget.download('https://github.com/protocolbuffers/protobuf/releases/download/v3.15.6/protoc-3.15.6-win64.zip', paths['PROTOC_PATH'])
    !cd {paths['PROTOC_PATH']} && tar -xf protoc-3.15.6-win64.zip
    os.environ['PATH'] += os.pathsep + os.path.abspath(os.path.join(paths['PROTOC_PATH'], 'bin'))
    !cd Tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=. && copy object_detection\\packages\\tf2\\setup.py setup.py && python setup.py build && python setup.py install
    !cd Tensorflow/models/research/slim && pip install -e .

if os.name == 'nt':
    wget.download(PRETRAINED_MODEL_URL)
    shutil.move(PRETRAINED_MODEL_NAME+".tar.gz", paths["PRETRAINED_MODEL_PATH"])

!cd Tensorflow/workspace/pre-trained-models/ && tar -zxvf {PRETRAINED_MODEL_NAME+".tar.gz"}


labels = [{'name':'YellowYellow', 'id':1},{'name':'YellowNothing', 'id':2},{'name':'NothingYellow', 'id':3},{'name':'NothingNothing', 'id':4} ]

with open(files['LABELMAP'], 'w') as f:
    for label in labels:
        f.write('item{\n')
        f.write('\tname:\'{}\'\n'.format(label['id']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('{\n')


ARCHIVE_FILES = os.path.join(paths['IMAGE_PATH'],'archive.tar.gz')
if os.path.exists(ARCHIVE_FILES):
    !tar -zvxf {ARCHIVE_FILES}


!python {files['TF_RECORD_SCRIPT']} -x {os.path.join(paths['IMAGE_PATH'],'train')} -1 {files['LABELMAP']} -o {os.path.join(paths['ANNOTATION_PATH'], 'train.record')}
!python {files['TF_RECORD_SCRIPT']} -x {os.path.join(paths['IMAGE_PATH'], 'test')} -1 {files['LABELMAP']} -o {os.path.join(paths['ANNOTATION_PATH'], 'test.record')}

if os.name == 'nt':
    !copy {os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'pipeline.config')} {os.path.join(paths['CHECKPOINT_PATH'])}

config = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.Gfile(files['PIPELINE_CONFIG'], "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)

pipeline_config.train_config.batch_size = 4
pipeline_config.train_config.fine_tune_checkpoint = os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-0')
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path= files['LABELMAP']
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'train.record')]
pipeline_config.eval_input_reader[0].label_map_path = files['LABELMAP']
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'test.record')]

config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        \n",
    with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], 'wb') as f:                                                                                                                                                                                                                     \n",
    f.write(config_text)

TRAINING_SCRIPT = os.path.join(paths['APIMODEL_PATH']), 'resaerch', 'object_detection', 'model_main_tf2.py'

command = "python {} --model_dir={} --pipeline_config_path={} --num_train_steps=3000".format(TRAINING_SCRIPT, paths['CHECKPOINT_PATH'], files['PIPELINE_CONFIG'])
print(command)
!{command}

command = "python {} --model_dir={} --pipeline_config_path={} --check_point_dir={}".format(TRAINING_SCRIPT, paths['CHECKPOINT_PATH'], files['PIPELINE_CONFIG'], paths['CHECKPOINT_PATH'])
print(command)
!{command}

config = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-3')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image,shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
IMAGE_PATH = os.path.join(paths['IMAGE_PATH'], 'test', .....)
img = cv2.imread(IMAGE_PATH)
image_np = np.array(img)

input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
detections = detect_fn(input_tensor)

num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy()
             for key, value in detections.items()}
detections['num_detections'] = num_detections\


detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

label_id_offset = 1
image_np_with_detections = image_np.copy()
viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes']+label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=.8,
            agnostic_mode=False)
   
plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
plt.show()

FREEEZE_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'exporter_main_v2.py')
command = "python {} --input_type=image_tensor --pipeline_config_path={} --trained_checkpoint_dir={} --outbut_directory={}.format(TRAINING_SCRIPT, paths['CHECKPOINT_PATH'], files['PIPELINE_CONFIG'], paths['CHECKPOINT_PATH'])
print(command)
!{command}

command = "python {} --input_type=image_tensor --pipeline_config_path={} --trained_checkpoint_dir={} --outbut_directory={}.format(TRAINING_SCRIPT, paths['CHECKPOINT_PATH'], files['PIPELINE_CONFIG'], paths['CHECKPOINT_PATH'])
print(command)
!{command}

TFLITE_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection','export_tflite_graph_tf2.py ')
command = "python {} --pipeline_config_path={} --trained_checkpoint_dir={} --output_directory={}".format(TFLITE_SCRIPT ,files['PIPELINE_CONFIG'])
print(command)
!{command}

FROZEN_TFLITE_PATH = os.path.join(paths['TFLITE_PATH'], 'saved_model')
TFLITE_MODEL = os.path.join(paths['TFLITE_PATH'], 'saved_model', 'detect.tflite')

command = "tflite_convert \
--saved_model_dir={} \
--output_file={} \
--input_shapes=1,300,300,3 \
--input_arrays=normalized_input_image_tensor \
--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
--inference_type=FLOAT \
--allow_custom_ops".format(FROZEN_TFLITE_PATH, TFLITE_MODEL, )
print(command)
!{command}

!tar -czf models.tar.gz {paths['CHECKPOINT_PATH']}