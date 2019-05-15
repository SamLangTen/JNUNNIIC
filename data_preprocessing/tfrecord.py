import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def clear_dir(dir):
    import shutil
    import os
    shutil.rmtree(dir)
    os.mkdir(dir)

'''制作tfrecord'''
def mk_tfrecord(path,file):
    writer = tf.python_io.TFRecordWriter(file)
    for index,class_name in enumerate(classes):
        class_path = path + class_name + '/'
        img_list = os.listdir(class_path)
        print(index, class_name)
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

'''读取tfrecord'''
def read_and_decode(filename):
    #根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [128, 128, 3])
    img = tf.cast(img, tf.float32) * (1. / 255)
    label = tf.cast(features['label'], tf.int32)

    return img, label

def get_data(tfrecord_path,batch_size,one_hot=False,label_num=0):
    min_after_dequeue = 500
    capacity = min_after_dequeue + 3*batch_size
    x, label = read_and_decode(tfrecord_path)
    x, label = tf.train.shuffle_batch([x, label],
                                      batch_size=batch_size,
                                      capacity=capacity,
                                      min_after_dequeue=min_after_dequeue,
                                      num_threads=64)
    if one_hot:
        label = tf.one_hot(label, label_num, 1, 0)

    return x, label

if __name__ == '__main__':

    train_path = '../data/seg_train/'
    val_path = '../data/seg_val/'
    test_path = '../data/seg_test/'

    record_path = '../data/record/'
    if not os.path.exists(record_path):
        os.mkdir(record_path)

    classes = os.listdir(train_path)

    mk_tfrecord(train_path, record_path + "train.tfrecords")
    mk_tfrecord(val_path, record_path + "val.tfrecords")
    mk_tfrecord(test_path, record_path + "test.tfrecords")
