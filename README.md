# Piglet Detector 🐖

TensorFlow Object Detection API project for detecting piglets using **SSD MobileNet V2 FPNLite**.

---

## Environment
- Python 3.10
- TensorFlow 2.x
- TensorFlow Object Detection API

---

## Project Structure
```text
piglet_project/
├── annotations/
│ ├── label_map.pbtxt
├── models/
│ └── research/
├── scripts/
├── ssd_mobilenet_v2_fpnlite_piglet.config
├── requirements.txt
└── README.md
```

---

## Dataset
TFRecord files are **not included** in this repository.

You must generate:
- `annotations/train.tfrecord`
- `annotations/valid.tfrecord`

before training.

---

## Setup (New Machine)
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Training
```bash
python models/research/object_detection/model_main_tf2.py \
  --pipeline_config_path ssd_mobilenet_v2_fpnlite_piglet.config \
  --model_dir models/research/training \
  --alsologtostderr
```

## Exporting
```bash
python models/research/object_detection/exporter_main_v2.py \
  --input_type image_tensor \
  --pipeline_config_path ssd_mobilenet_v2_fpnlite_piglet.config \
  --trained_checkpoint_dir models/research/training \
  --output_directory models/research/exported-model
```

## Inference (Video)
```bash
python scripts/live_infer.py --video path/to/video.mp4
```

## Inference (Webcam)
```bash
python scripts/live_infer.py
```

## TensorFlow Object Detection API

This project requires the TensorFlow Models repository.

Clone it separately:
```bash
git clone https://github.com/tensorflow/models.git
```
Then add `models/research` to your PYTHONPATH.
