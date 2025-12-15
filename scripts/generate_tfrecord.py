"""
generate_tfrecord.py
Converts Pascal VOC XML annotations to TFRecord format for TensorFlow Object Detection.
Now supports both train and test sets via command-line arguments.
"""

import os
import glob
import argparse
import pandas as pd
import tensorflow as tf
from xml.etree import ElementTree as ET
from object_detection.utils import dataset_util

# Label names to IDs (adjust as needed)
CLASS_DICT = {
    'pig': 1,
    'piglet': 2
}

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(os.path.join(path, '*.xml')):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        filename = root.find('filename').text
        width = int(root.find('size/width').text)
        height = int(root.find('size/height').text)

        for member in root.findall('object'):
            obj_class = member.find('name').text
            bndbox = member.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            xml_list.append((filename, width, height, obj_class, xmin, ymin, xmax, ymax))
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    return pd.DataFrame(xml_list, columns=column_name)

def create_tf_example(row, path):
    img_path = os.path.join(path, row['filename'])
    if not os.path.exists(img_path):
        print(f"[WARN] Image not found: {img_path}")
        return None

    with tf.io.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()

    filename = row['filename'].encode('utf8')
    width = int(row['width'])
    height = int(row['height'])
    image_format = b'jpg'

    xmins = [row['xmin'] / width]
    xmaxs = [row['xmax'] / width]
    ymins = [row['ymin'] / height]
    ymaxs = [row['ymax'] / height]
    classes_text = [row['class'].encode('utf8')]
    classes = [CLASS_DICT[row['class']]]

    return tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

def main():
    parser = argparse.ArgumentParser(description="Generate TFRecord from Pascal VOC annotations")
    parser.add_argument("--folder", type=str, default="dataset/train", help="Path to dataset folder containing images and XML files")
    parser.add_argument("--output", type=str, default="annotations/train.record", help="Output TFRecord path")
    args = parser.parse_args()

    path = os.path.join(os.getcwd(), args.folder)
    examples = xml_to_csv(path)
    writer = tf.io.TFRecordWriter(args.output)

    count = 0
    for _, row in examples.iterrows():
        tf_example = create_tf_example(row, path)
        if tf_example:
            writer.write(tf_example.SerializeToString())
            count += 1

    writer.close()
    print(f"✅ TFRecord file created: {args.output} ({count} examples)")

if __name__ == '__main__':
    main()
