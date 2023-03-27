import tensorflow_hub as hub
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.
    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.
    Args:
      path: the file path to the image
    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))


image_path = "resources/images/example.jpg"
image = Image.open(image_path)

image_np = load_image_into_numpy_array(image_path)
image_tensor = tf.convert_to_tensor(image_np, dtype=tf.uint8)[tf.newaxis, :]

model_path = "efficientdet_d7_1"
model = hub.load(model_path)

outputs = model(image_tensor)

boxes = outputs["detection_boxes"][0].numpy()
classes = outputs["detection_classes"][0].numpy()
scores = outputs["detection_scores"][0].numpy()

image_draw = ImageDraw.Draw(image)
font = ImageFont.truetype("arial.ttf", 40)

# Load the label map file
label_map_file = 'models/research/object_detection/data/mscoco_complete_label_map.pbtxt'
label_map = {}
with open(label_map_file, 'r') as f:
    index = 0
    for line in f:
        if 'display_name:' in line:
            name = line.split(':')[-1].strip().replace("'", "")
            label_map[index] = name
            index += 1


# Map the class indices to label names
print(classes)
labels = [label_map[c] for c in classes]

index = 0
for box, cls, score in zip(boxes, labels, scores):
    if score > 0.5:
        ymin, xmin, ymax, xmax = box
        x, y = image.size
        left, right, top, bottom = xmin * x, xmax * x, ymin * y, ymax * y
        image_draw.rectangle([left, top, right, bottom], outline="red", width=2)
        image_draw.text([left, top], f"{labels[index]} ({score:.2f})", font=font, fill="red")
    index += 1

image.show()
# image.save("path/to/output.jpg")
