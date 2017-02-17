import os
import glob
import scipy.misc
import numpy as np
import tensorflow as tf

tfrecords_filename = "flowers.tfrecords"
LABELS = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def main():
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    for label in LABELS:
        paths = glob.glob(os.path.join("flower_photos", label)+"/*.jpg")
        for path in paths:
            im = scipy.misc.imread(path)
            im = scipy.misc.imresize(im, [64, 64])
            # fit to [-1, 1] range
            # im = im / 127.5 - 1.0

            h, w = im.shape[:2]
            im_raw = im.tostring()

            example = tf.train.Example(features=tf.train.Features(
                feature={
                        "height": _int64_feature([h]),
                        "width": _int64_feature([w]),
                        "image": _bytes_feature([im_raw])
                    }))
            writer.write(example.SerializeToString())

    writer.close()

if __name__ == "__main__":
    main()
