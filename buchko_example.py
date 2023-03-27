import os

import tensorflow_hub as hub
import tensorflow as tf
import time

import random
import numpy as np
import cv2
from object_detection.builders import model_builder

from object_detection.utils import visualization_utils as viz_utils

from models.research.object_detection.utils import label_map_util, config_util


def get_model_detection_function(model):
    """Get a tf.function for detection."""

    @tf.function
    def detect_fn(image):
        """Detect objects in image."""

        image, shapes = model.preprocess(image)
        prediction_dict = model.predict(image, shapes)
        detections = model.postprocess(prediction_dict, shapes)

        return detections, prediction_dict, tf.reshape(shapes, [-1])

    return detect_fn


# recover our saved model
pipeline_config = 'models/research/object_detection/configs/tf2/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.config'
# generally you want to put the last ckpt from training in here
model_dir = 'ckpt-0'
# print(os.getcwd())
configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']
detection_model = model_builder.build(
    model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(
    model=detection_model)
ckpt.restore(
    'models/research/object_detection/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/checkpoint/ckpt-0')
# ckpt.restore(os.path.join('ckpt-0'))

detect_fn = get_model_detection_function(detection_model)

# map labels for inference decoding
label_map_path = configs['eval_input_config'].label_map_path
label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=label_map_util.get_max_label_map_index(label_map),
    use_display_name=True)
category_index = label_map_util.create_category_index(categories)
print(category_index)
label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)

video_path = "resources/videos/street.mp4"
cap = cv2.VideoCapture(0)
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('stb_out.mp4', fourcc, 20.0, (640, 480))


def load_image_into_numpy_array(image):
    """Load an image from file into a numpy array.
    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.
    Args:
      path: the file path to the image
    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
      :param image:
    """
    # img_data = tf.io.gfile.GFile(path, 'rb').read()
    # image = Image.open(BytesIO(img_data))
    (im_width, im_height, channel) = image.shape
    return image.astype(np.uint8)


while True:
    # Capture frame-by-frame

    ret, image_np = cap.read()
    image_np = load_image_into_numpy_array(image_np)

    input_tensor = tf.convert_to_tensor(
        np.expand_dims(image_np, 0), dtype=tf.float32)

    detections, predictions_dict, shapes = detect_fn(input_tensor)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'][0].numpy(),
        (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
        detections['detection_scores'][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.5,
        agnostic_mode=False,
    )

    # Display the resulting frame
    out.write(image_np_with_detections)
    cv2.imshow('frame', image_np_with_detections)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
