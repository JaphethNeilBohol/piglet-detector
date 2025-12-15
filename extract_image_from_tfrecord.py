# extract_image_from_tfrecord.py
import tensorflow as tf
import cv2
import numpy as np

record = "annotations/train.tfrecord"
output = "sample_from_tfrecord.jpg"

raw_dataset = tf.data.TFRecordDataset(record)

for raw in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw.numpy())
    img_bytes = example.features.feature["image/encoded"].bytes_list.value[0]
    img = tf.io.decode_jpeg(img_bytes).numpy()
    cv2.imwrite(output, img)
    print("Saved:", output)
