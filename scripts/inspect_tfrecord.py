import tensorflow as tf
import sys
from object_detection.utils import dataset_util

def inspect_tfrecord(path, max_examples=5):
    print("Inspecting TFRecord:", path)
    raw_dataset = tf.data.TFRecordDataset(path)

    for i, raw_record in enumerate(raw_dataset):
        if i >= max_examples:
            break

        print("\n------------------------------------")
        print(f"Example {i+1}")
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())

        features = example.features.feature

        # Image shape info
        height = features['image/height'].int64_list.value[0]
        width = features['image/width'].int64_list.value[0]
        print("Image size:", height, "x", width)

        # Class labels
        classes = features['image/object/class/label'].int64_list.value
        print("Classes:", list(classes))

        # Display boxes
        ymin = features['image/object/bbox/ymin'].float_list.value
        xmin = features['image/object/bbox/xmin'].float_list.value
        ymax = features['image/object/bbox/ymax'].float_list.value
        xmax = features['image/object/bbox/xmax'].float_list.value

        boxes = list(zip(ymin, xmin, ymax, xmax))
        print("Boxes (ymin,xmin,ymax,xmax):", boxes)

    print("\nDone.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_tfrecord.py <path_to_tfrecord>")
        sys.exit(1)

    inspect_tfrecord(sys.argv[1])
