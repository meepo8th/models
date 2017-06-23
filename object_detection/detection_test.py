import os
import shutil

import numpy as np
import tensorflow as tf
import time
from PIL import Image

from object_detection.utils import label_map_util
from object_detection.utils.pascal_voc_io import PascalVocWriter

MODEL_NAME = 'E:/testmodel/faster_rcnn_resnet101_coco_11_06_2017'

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


def get_un_process_path():
    return [os.path.join(PATH_TO_FIND_IMAGES_DIR, '{}'.format(i)) for i in
            os.listdir(PATH_TO_FIND_IMAGES_DIR)]


def checkPerson(classes, scores):
    for i, score in enumerate(scores):
        if (score < 0.2):
            break
        if classes[i] in category_index and category_index[classes[i]]['name'] == "person":
            return True
    return False


def convertPoints2BndBox(points, shape):
    print(shape)
    ymin, xmin, ymax, xmax = points
    xmin = xmin * shape[1]
    ymin = ymin * shape[0]
    xmax = xmax * shape[1]
    ymax = ymax * shape[0]

    if xmin < 1:
        xmin = 1

    if ymin < 1:
        ymin = 1

    return (int(xmin), int(ymin), int(xmax), int(ymax))


def write2VocFile(boxes, labels, targetFileName, imgFolderName, imgFileNameWithoutExt, imageSize, imagePath):
    print(boxes, labels, targetFileName, imgFolderName, imgFileNameWithoutExt, imageSize, imagePath)
    writer = PascalVocWriter(imgFolderName, imgFileNameWithoutExt,
                             imageSize, localImgPath=imagePath)

    for i, box in enumerate(boxes):
        label = labels[i]
        # Add Chris
        difficult = int(0)
        bndbox = convertPoints2BndBox(box, imageSize)
        writer.addBndBox(bndbox[0], bndbox[1], bndbox[2], bndbox[3], label, difficult)

    writer.save(targetFile=targetFileName)


def generateLabel(boxes, classes, scores, image_path, imageSize):
    needBoxes = []
    needLabels = []
    label_path = os.path.join(os.path.dirname(image_path), os.path.basename(image_path).replace(".jpg", ".xml"))

    for i, score in enumerate(scores):
        if (score > 0.2 and classes[i] in category_index and category_index[classes[i]]['name'] == "person"):
            needBoxes.append(boxes[i])
            needLabels.append(category_index[classes[i]]['name'])
    write2VocFile(needBoxes, needLabels, label_path, os.path.dirname(image_path),
                  os.path.basename(image_path).replace(".jpg", ""), imageSize, image_path)
    return label_path


def detect(UN_PROCESS_IMAGE_PATHS):
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            for image_path in UN_PROCESS_IMAGE_PATHS:
                if not image_path.__contains__(".jpg"):
                    break;
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
                print(time.time())

                if checkPerson(np.squeeze(classes).astype(np.int32),
                               np.squeeze(scores)):
                    label_path = generateLabel(np.squeeze(boxes), np.squeeze(classes).astype(np.int32),
                                               np.squeeze(scores), image_path, (image.size[1], image.size[0], 3))
                    shutil.move(image_path, os.path.join(PATH_PROCESS_IMAGES_DIR, os.path.basename(image_path)))
                    shutil.move(label_path, os.path.join(PATH_PROCESS_IMAGES_DIR, os.path.basename(label_path)))
                else:
                    os.remove(image_path)


while True:
    UN_PROCESS_IMAGE_PATHS = get_un_process_path()
    if len(UN_PROCESS_IMAGE_PATHS) > 0:
        detect(UN_PROCESS_IMAGE_PATHS)
    else:
        time.sleep(5)
