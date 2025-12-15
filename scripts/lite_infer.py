import cv2
import numpy as np
import tensorflow as tf

MODEL_DIR = "exported-models/piglet_detector/saved_model"
LABELS = {1: "piglet"}

print("Loading model...")
detect_fn = tf.saved_model.load(MODEL_DIR)
print("Model loaded.")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_exp = np.expand_dims(img, axis=0)

    input_tensor = tf.convert_to_tensor(img_exp, dtype=tf.uint8)
    outputs = detect_fn(input_tensor)

    boxes = outputs['detection_boxes'][0].numpy()
    scores = outputs['detection_scores'][0].numpy()
    classes = outputs['detection_classes'][0].numpy().astype(int)

    h, w, _ = frame.shape
    for i in range(len(scores)):
        if scores[i] < 0.5:
            continue

        ymin, xmin, ymax, xmax = boxes[i]
        x1, y1 = int(xmin * w), int(ymin * h)
        x2, y2 = int(xmax * w), int(ymax * h)

        label = LABELS.get(classes[i], "unknown")
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"{label}: {scores[i]:.2f}",
                    (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,255,0), 2)

    cv2.imshow("Lite Piglet Detector", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
