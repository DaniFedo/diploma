import tensorflow_hub as hub
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
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
    # img_data = tf.io.gfile.GFile(path, 'rb').read()
    # image = Image.open(BytesIO(img_data))
    (im_width, im_height, channel) = image.shape
    return image.astype(np.uint8)


# image_path = "resources/images/example.jpg"
# image = Image.open(image_path)

# label_map_file = 'models/research/object_detection/data/mscoco_complete_label_map.pbtxt'
# label_map = {}
# with open(label_map_file, 'r') as f:
#     index = 0
#     for line in f:
#         if 'display_name:' in line:
#             name = line.split(':')[-1].strip().replace("'", "")
#             label_map[index] = name
#             index += 1

label_map_path = "models/research/object_detection/data/mscoco_complete_label_map.pbtxt"
print(f"PATH: {label_map_path}")
label_map = label_map_util.load_labelmap(label_map_path)
print(f"MAP: {label_map}")
categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=label_map_util.get_max_label_map_index(label_map),
    use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Map the class indices to label names


model_path = "efficientdet_d7_1"
model = hub.load(model_path)

video_path = "resources/videos/street.mp4"
cap = cv2.VideoCapture(video_path)
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.avi', -1, 20.0, (640, 480))
# labels = {}
counter = 0
while counter < cv2.CAP_PROP_FRAME_COUNT:
    ret, image_np = cap.read()

    # image_np = load_image_into_numpy_array(image_np)
    # image_tensor = tf.convert_to_tensor(image_np, dtype=tf.uint8)[tf.newaxis, :]

    image_np = load_image_into_numpy_array(image_np)
    image_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.uint8)

    outputs = model(image_tensor)

    boxes = outputs["detection_boxes"][0].numpy()
    classes = outputs["detection_classes"][0].numpy()
    scores = outputs["detection_scores"][0].numpy()
    #
    # if len(labels) == 0:
    #     labels = [label_map[c] for c in classes]
    #     # print(labels)
    #     labels = dict(zip(range(len(labels)), labels))

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    # image_draw = ImageDraw.Draw(image)
    # font = ImageFont.truetype("arial.ttf", 40)

    # Load the label map file

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        boxes,
        classes.astype(int),
        scores,
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
    counter += 1
# image.save("path/to/output.jpg")
cap.release()
out.release()
cv2.destroyAllWindows()
