import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

train_path = '../data/seg_train/'
val_path = '../data/seg_val/'
test_path = '../data/seg_test/'

record_path = '../data/record/'
if not os.path.exists(record_path):
    os.mkdir(record_path)

classes = os.listdir(train_path)
train_writer = tf.python_io.TFRecordWriter(record_path + "train.tfrecords")
val_writer = tf.python_io.TFRecordWriter(record_path +"val.tfrecords")
test_writer = tf.python_io.TFRecordWriter(record_path + "test.tfrecords")

def mk_tfrecord(path,writer):
    for index,class_name in enumerate(classes):
        class_path = path + class_name + '/'
        img_list = os.listdir(class_path)

        for imgname in img_list:
            img_path = class_path + imgname
            img = Image.open(img_path)
            img = img.resize((128,128))
            img_raw = img.tobytes()  # 转为二进制
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))  # example对象对label和image数据进行封装
            writer.write(example.SerializeToString())  # 序列化为字符串

    writer.close()

if __name__ == '__main__':

    mk_tfrecord(train_path,train_writer)
    mk_tfrecord(val_path, val_writer)
    mk_tfrecord(test_path, test_writer)
