# test_saved_model_single_image.py
import tensorflow as tf
import numpy as np
import cv2, os
MODEL_DIR = r"models\research\exported-model\saved_model"
IMG_PATH = r"C:\Users\User\Piglet_Detector\piglet_project\sample_from_tfrecord.jpg"   # <-- replace with an actual JPEG path from your dataset

print("Loading SavedModel...")
detect_fn = tf.saved_model.load(MODEL_DIR)
print("Model loaded.")

# Read image and resize to model input
img = cv2.imread(IMG_PATH)
if img is None:
    raise SystemExit("Image not found: " + IMG_PATH)
# If your pipeline uses 512x512
resized = cv2.resize(img, (512, 512))
input_tensor = tf.convert_to_tensor(np.expand_dims(resized, 0), dtype=tf.uint8)

out = detect_fn(input_tensor)
# Print key outputs and ranges
for k, v in out.items():
    if isinstance(v, tf.Tensor):
        a = v.numpy()
        print(f"{k}: shape={a.shape} min={a.min():.6g} max={a.max():.6g} mean={a.mean():.6g}")
    else:
        try:
            a = v.numpy()
            print(f"{k}: shape={a.shape} min={a.min():.6g} max={a.max():.6g} mean={a.mean():.6g}")
        except Exception as e:
            print(k, "->", type(v), e)
