# import tensorflow as tf
#
# # Check TensorFlow version
# print("TensorFlow version:", tf.__version__)
#
# # Check if GPU is available
# if tf.config.list_physical_devices('GPU'):
#     print("GPU is available")
# else:
#     print("GPU is NOT available")

import tensorflow as tf
import tensorflow_hub as hub
import cv2

print("start")

# Load an image
image = cv2.imread("resources/images/example.jpg")

# Resize the image to match the input size of the model
image = cv2.resize(image, (640, 640))

# Convert the image to a TensorFlow tensor
image_tensor = tf.convert_to_tensor(image)

# Add a batch dimension to the tensor
image_tensor = tf.expand_dims(image_tensor, axis=0)

print("starting loading")  # works
# Load the EfficientDet-Lite4 model from TensorFlow Hub
# model_url = "https://tfhub.dev/tensorflow/efficientdet/d7/1"
model_path = "efficientdet_d7_1"
detector = hub.load(model_path)
print("loaded")  # works

print(list(detector.signatures.keys()))

mobilenet_save_path = "C:/Users/Daniil/diploma/resources/models"
tf.saved_model.save(detector, mobilenet_save_path)

print("saved")

detector_test = tf.saved_model.load(mobilenet_save_path)
print(list(detector_test.signatures.keys()))


# Run the object detection model on the image
detections = detector(image_tensor)

# Print the detected objects
print(detections)
