from glob import glob
from PIL import Image
import numpy as np
import skimage.io as io
import tensorflow as tf

sketch_names = glob("small_data/ar_s/*jpg")
neutral_names = glob("small_data/ar_nf/*jpg")

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


tfrecords_filename = 'small_data/ar.tfrecords'

writer = tf.python_io.TFRecordWriter(tfrecords_filename)

for sketch_name, neutral_name in zip(sketch_names, neutral_names):
    print(sketch_name)
    print(neutral_name)

    sketch = np.array(Image.open(sketch_name))
    neutral = np.array(Image.open(neutral_name))

    height = sketch.shape[0]
    width = sketch.shape[1]

    sketch_raw = sketch.tostring()
    neutral_raw = neutral.tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'sketch_raw': _bytes_feature(sketch_raw),
        'neutral_raw': _bytes_feature(neutral_raw)
    }))

    writer.write(example.SerializeToString())

writer.close()
