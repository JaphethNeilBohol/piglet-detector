import os

# === PATHS ===
CUSTOM_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
WORKSPACE_PATH = os.path.join('piglet_project')
PRETRAINED_MODEL_PATH = os.path.join(WORKSPACE_PATH, 'pre_trained_model')
CHECKPOINT_PATH = os.path.join(PRETRAINED_MODEL_PATH, CUSTOM_MODEL_NAME)
PIPELINE_CONFIG_PATH = os.path.join(CHECKPOINT_PATH, 'pipeline.config')

# === TRAINING COMMAND ===
command = f"python -m object_detection.model_main " \
          f"--model_dir={CHECKPOINT_PATH} " \
          f"--pipeline_config_path={PIPELINE_CONFIG_PATH} " \
          f"--num_train_steps=2000"

os.system(command)
