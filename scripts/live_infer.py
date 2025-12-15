import cv2
import numpy as np
import tensorflow as tf
import argparse
import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

# ---------------------------------------------------------------
# Resolve paths relative to project root
# ---------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "exported-model", "saved_model")
LABEL_MAP_PATH = os.path.join(PROJECT_ROOT, "annotations", "label_map.pbtxt")

# ---------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, help="Optional: path to a video file.")
parser.add_argument("--model", type=str, help="Optional: path to a SavedModel folder.")
args = parser.parse_args()

# ---------------------------------------------------------------
# Resolve model path
# ---------------------------------------------------------------
MODEL_DIR = args.model if args.model else DEFAULT_MODEL_DIR

print("USING MODEL:", MODEL_DIR)
print("───────────────────────────────────────────────")
print("Model Directory:", MODEL_DIR)
print("Label Map:", LABEL_MAP_PATH)
print("───────────────────────────────────────────────")

# ---------------------------------------------------------------
# Check paths
# ---------------------------------------------------------------
if not os.path.exists(MODEL_DIR):
    raise FileNotFoundError(f"ERROR: Model not found at:\n{MODEL_DIR}")

if not os.path.exists(LABEL_MAP_PATH):
    raise FileNotFoundError(f"ERROR: Label map not found at:\n{LABEL_MAP_PATH}")

# ---------------------------------------------------------------
# Load model (warm-up)
# ---------------------------------------------------------------
print("Loading model...")
detect_fn = tf.saved_model.load(MODEL_DIR)
print("Model loaded!")

dummy_input = tf.zeros([1, 512, 512, 3], dtype=tf.uint8)
_ = detect_fn(input_tensor=dummy_input)
print("Model warm-up done.")

# ---------------------------------------------------------------
# Load label map
# ---------------------------------------------------------------
category_index = label_map_util.create_category_index_from_labelmap(
    LABEL_MAP_PATH, use_display_name=True
)

# ---------------------------------------------------------------
# Video source
# ---------------------------------------------------------------
if args.video:
    print("Using video file:", args.video)
    cap = cv2.VideoCapture(args.video)
else:
    print("Using webcam...")
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("ERROR: Could not open camera or video file.")

# ---------------------------------------------------------------
# Detection settings
# ---------------------------------------------------------------
MODEL_INPUT_SIZE = (512, 512)
DETECTION_THRESHOLD = 0.30
MAX_DETECTIONS = 50

print("───────────────────────────────────────────────")
print("Detector ready. Press Q to quit.")
print("───────────────────────────────────────────────")

# ---------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_tensor = tf.convert_to_tensor([frame], dtype=tf.uint8)
    detections = detect_fn(input_tensor=input_tensor)

    scores = detections['detection_scores'][0].numpy()
    boxes = detections['detection_boxes'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.int32)

    print(f"Scores: {scores[:3]}")  # Debug

    valid = scores >= DETECTION_THRESHOLD
    scores = scores[valid][:MAX_DETECTIONS]
    boxes = boxes[valid][:MAX_DETECTIONS]
    classes = classes[valid][:MAX_DETECTIONS]

    viz_utils.visualize_boxes_and_labels_on_image_array(
        frame,
        boxes,
        classes,
        scores,
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=MAX_DETECTIONS,
        min_score_thresh=DETECTION_THRESHOLD,
        line_thickness=3
    )

    cv2.imshow("Piglet Detector - Improved", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
