import os

import numpy as np
import shutil
import tensorflow as tf
import time
from PIL import Image
import pylab
from matplotlib import pyplot as plt

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# What model to download.
MODEL_NAME = 'E:/testmodel/ssd_inception_v2_coco_11_06_2017'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)


PATH_TO_FIND_IMAGES_DIR = 'E:/picRecord/ori'
PATH_PROCESS_IMAGES_DIR = 'E:/picRecord/process'
UN_PROCESS_IMAGE_PATHS = [os.path.join(PATH_TO_FIND_IMAGES_DIR, '{}'.format(i)) for i in
                          os.listdir(PATH_TO_FIND_IMAGES_DIR)]
IMAGE_SIZE = [12, 8]


def checkPerson(boxes, classes, scores):
    for i in (0, scores.__le__(scores) - 1):
        if (scores[i] < 0.7):
            break
        if classes[i] in category_index and category_index[classes[i]]['name'] == "person":
            return True
    return False


with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        for image_path in UN_PROCESS_IMAGE_PATHS:
            image = Image.open(image_path)
            image_np = load_image_into_numpy_array(image)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
            if checkPerson(np.squeeze(boxes),
                           np.squeeze(classes).astype(np.int32),
                           np.squeeze(scores)):
                shutil.move(image_path, os.path.join(PATH_PROCESS_IMAGES_DIR, os.path.basename(image_path)))
            else:
                os.remove(image_path)
