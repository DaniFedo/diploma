import threading
import time

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from object_detection.utils import visualization_utils as viz_utils

from models.research.object_detection.utils import label_map_util

exitFlag = 0


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


class myThread(threading.Thread):
    def __init__(self, threadID, model, category_index, frames):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.category_index = category_index
        self.frames = frames
        self.counter = 0
        self.output_frames = []
        self.model = model

    def run(self):
        for frame in self.frames:
            print(f"I do work with counter:{self.counter}")
            image_np_inner = load_image_into_numpy_array(frame)
            image_tensor = tf.convert_to_tensor(np.expand_dims(image_np_inner, 0), dtype=tf.uint8)
            outputs = self.model(image_tensor)
            boxes = outputs["detection_boxes"][0].numpy()
            classes = outputs["detection_classes"][0].numpy()
            scores = outputs["detection_scores"][0].numpy()
            image_np_with_detections_inner = image_np_inner.copy()
            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections_inner,
                boxes,
                classes.astype(int),
                scores,
                self.category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=.5,
                agnostic_mode=False,
            )
            self.output_frames.append(image_np_with_detections_inner)
            self.counter += 1

    def get_output_frames(self):
        return self.output_frames


start_time = time.time()

label_map_path = "models/research/object_detection/data/mscoco_complete_label_map.pbtxt"
# print(f"PATH: {label_map_path}")
label_map = label_map_util.load_labelmap(label_map_path)
# print(f"MAP: {label_map}")
categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=label_map_util.get_max_label_map_index(label_map),
    use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Map the class indices to label names
video_path = "resources/videos/street.mp4"
cap = cv2.VideoCapture(video_path)
# Define the codec and create VideoWriter object
print(f"Width:{cap.get(3)}")
print(f"Height:{cap.get(4)}")

frameSize = (cap.get(3), cap.get(4))

print(cap.get(cv2.CAP_PROP_FRAME_COUNT))

model_path = "efficientdet_d7_1"
model = hub.load(model_path)

# print("copying model")
# model1 = model
# print("copied model")

counter = 0
image_nps_array = []
while counter < cap.get(cv2.CAP_PROP_FRAME_COUNT) and cap.isOpened():
    ret, image_np = cap.read()

    if not ret:
        break
    image_nps_array.append(image_np)

print(f"size of image_nps_array: {len(image_nps_array)}")

# graph2 = tf.Graph()
# with graph2.as_default():
#     model = tf.constant(2)
#     # model = hub.load(hub.load("efficientdet_d7_1"))
#
# graph1 = tf.Graph()
# with graph1.as_default():
#     model = tf.constant(1)
#     # model = hub.load(hub.load("efficientdet_d7_1"))


# with tf.compat.v1.Session(graph=graph1) as sess:
#     result = sess.run(model)
#     print(result)

thread1 = myThread(1, category_index, image_nps_array[:2])
thread2 = myThread(2, category_index, image_nps_array[2:4])
# thread3 = myThread(3, model, category_index, 6, 3, cap)

print("Starting thread")
thread1.start()
thread2.start()
# thread3.start()
thread1.join()
thread2.join()
# thread3.join()

result = thread1.get_output_frames()
result += thread2.get_output_frames()
# result += thread3.get_output_frames()

# print(f"got result value: {result}")

print("I am starting showing")
out = cv2.VideoWriter('output_1.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20.0, (1920, 1080))
for image_np_with_detections in result:
    cv2.imshow('frame', image_np_with_detections)
    out.write(image_np_with_detections)
    cv2.waitKey(1000)
print("Finished showing")
#
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Time for execution: {time.time() - start_time}")
# # Create new threads
# thread1 = myThread()
#
# # Start new Threads
# thread1.start()
# thread2.start()
#
# print
# "Exiting Main Thread"
