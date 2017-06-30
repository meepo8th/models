import os
import shutil

import numpy as np
import tensorflow as tf
import time
from PIL import Image
from matplotlib import pyplot as plt
import pylab
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
from object_detection.utils.pascal_voc_io import PascalVocWriter

MODEL_NAME = 'd:/code/testmodel/custom_faster_rcnn_resnet_101'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/output_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'pascal_label_map_custom.pbtxt')

NUM_CLASSES = 90
IMAGE_SIZE = (12, 8)
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


PATH_TO_FIND_IMAGES_DIR = 'd:/code/picRecord/process'


def get_un_process_path():
    return [os.path.join(PATH_TO_FIND_IMAGES_DIR, '{}'.format(i)) for i in
            os.listdir(PATH_TO_FIND_IMAGES_DIR)]


def detect(UN_PROCESS_IMAGE_PATHS):
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            for image_path in UN_PROCESS_IMAGE_PATHS:
                if not image_path.__contains__(".jpg"):
                    continue;
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
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)
                plt.figure(figsize=IMAGE_SIZE)
                plt.imshow(image_np)
                pylab.show()

UN_PROCESS_IMAGE_PATHS = get_un_process_path()
if len(UN_PROCESS_IMAGE_PATHS) > 0:
    detect(UN_PROCESS_IMAGE_PATHS)

