import time

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from object_detection.utils import visualization_utils as viz_utils

from models.research.object_detection.utils import label_map_util


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
    (im_width, im_height, channel) = image.shape
    return image.astype(np.uint8)


start_time = time.time()
label_map_path = "models/research/object_detection/data/mscoco_complete_label_map.pbtxt"
label_map = label_map_util.load_labelmap(label_map_path)

# print(f"PATH: {label_map_path}")
# print(f"MAP: {label_map}")

categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=label_map_util.get_max_label_map_index(label_map),
    use_display_name=True)

category_index = label_map_util.create_category_index(categories)

video_path = "resources/videos/street.mp4"
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (frame_width, frame_height)
print(f"Width:{frame_width}")
print(f"Height:{frame_height}")
print(frame_size)


out = cv2.VideoWriter('output_lite4.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20.0,
                      frame_size)

model_type = "LITE"  # OTHER or LITE
model_path = "efficientdet_lite4_detection_2"

# model_type = "OTHER"
# model_path = "efficientdet_d7_1"
model = hub.load(model_path)

counter = 0
images_np_with_detections = []
print("Starting processing video")

# print(f"length of CAP_PROP_FRAME_COUNT: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")

while cap.isOpened():

    ret, image_np = cap.read()

    if not ret:
        break

    print(f"I do work with counter:{counter}")

    image_np = load_image_into_numpy_array(image_np)
    image_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.uint8)

    # boxes, scores, classes, num_detections = model(image_tensor)

    if model_type == "LITE":
        boxes, scores, classes, num_detections = model(image_tensor)
        boxes = boxes[0].numpy()
        classes = classes[0].numpy()
        scores = scores[0].numpy()

    else:
        outputs = model(image_tensor)

        boxes = outputs["detection_boxes"][0].numpy()
        classes = outputs["detection_classes"][0].numpy()
        scores = outputs["detection_scores"][0].numpy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np,
        boxes,
        classes.astype(int),
        scores,
        category_index,
        min_score_thresh=.5,
        agnostic_mode=False,
    )
    out.write(image_np)
    images_np_with_detections.append(image_np)

    counter += 1

print("Video is processed")
print(f"Length of detected images: {len(images_np_with_detections)}")
print(f"length of CAP_PROP_FRAME_COUNT: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
# for image_np_with_detections in images_np_with_detections:
#     cv2.imshow('frame', image_np_with_detections)
#     cv2.waitKey(1000)

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Time for execution: {time.time() - start_time}")
