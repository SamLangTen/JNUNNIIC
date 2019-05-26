import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import getopt


def clear_dir(dir):
    import shutil
    import os
    shutil.rmtree(dir)
    os.mkdir(dir)


def mk_tfrecord_v1(path, file):
    '''制作tfrecord有序'''
    writer = tf.python_io.TFRecordWriter(file)
    classes = os.listdir(path)
    for index, class_name in enumerate(classes):
        class_path = path + class_name + '/'
        img_list = os.listdir(class_path)
        print(index, class_name)
        for imgname in img_list:
            img_path = class_path + imgname
            img = Image.open(img_path)
            img = img.resize((128, 128))
            img_raw = img.tobytes()  # 转为二进制
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))  # example对象对label和image数据进行封装
            writer.write(example.SerializeToString())  # 序列化为字符串

    writer.close()


def mk_tfrecord_v2(path, file):
    '''制作tfrecord 乱序'''
    writer = tf.python_io.TFRecordWriter(file)
    img_dict = {}
    class_path_list = []
    classes = os.listdir(path)
    # 将所有图片名以及对应label加入字典
    for index, class_name in enumerate(classes):
        class_path = path + class_name + '/'
        class_path_list.append(class_path)
        img_list = os.listdir(class_path)
        for imgname in img_list:
            img_dict[imgname] = index

    key_list = list(img_dict.keys())
    # 打乱图片名顺序
    random.shuffle(key_list)
    key_num = len(key_list)

    for i in range(0, key_num):
        label = img_dict[key_list[i]]
        imgname = key_list[i]
        print("==========current label {},current image {},current num {}==========".format(
            label, imgname, i))
        img_path = class_path_list[label] + imgname
        img = Image.open(img_path)
        img = img.resize((128, 128))
        img_raw = img.tobytes()  # 转为二进制
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))  # example对象对label和image数据进行封装
        writer.write(example.SerializeToString())  # 序列化为字符串

    writer.close()


def mk_tfrecord_all(file, *paths):
    '''制作所有数据的tfrecord'''
    writer = tf.python_io.TFRecordWriter(file)
    img_dict = {}
    for index, class_name in enumerate(os.listdir(paths[0][0])):
        for path in list(paths[0]):
            class_path = path + class_name + '/'
            img_list = os.listdir(class_path)
            for imgname in img_list:
                img_dict[os.path.join(class_path, imgname)] = index
    key_list = list(img_dict.keys())
    # 打乱图片名顺序
    random.shuffle(key_list)
    key_num = len(key_list)

    #写入
    for i in range(0, key_num):
        label = img_dict[key_list[i]]
        img_path = key_list[i]
        img = Image.open(img_path)
        img = img.resize((128, 128))
        img_raw = img.tobytes()  # 转为二进制
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))  # example对象对label和image数据进行封装
        writer.write(example.SerializeToString())  # 序列化为字符串
    writer.close()


def read_and_decode(filename):
    '''读取tfrecord'''
    # 根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename],
                                                    shuffle=True)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
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


def get_data(tfrecord_path, batch_size, one_hot=False, label_num=0):
    min_after_dequeue = 500
    capacity = min_after_dequeue + 3*batch_size
    x, label = read_and_decode(tfrecord_path)
    x, label = tf.train.shuffle_batch([x, label],
                                      batch_size=batch_size,
                                      capacity=capacity,
                                      min_after_dequeue=min_after_dequeue,
                                      num_threads=64,
                                      seed=0.2)
    if one_hot:
        label = tf.one_hot(label, label_num)

    return x, label


def print_help():
    print('制作NNIIC数据集')
    print('')
    print('./tfrecord.py [-h] [-r tfrecord_path] [-c training_set_path] [-v validation_set_path] [-t testing_set_path] [-a] [data_set_path1 data_set_path2...]')
    print('')
    print('参数：')
    print('-r --tfrecord_path\t指定输出数据集的目录路径。默认为../data/record')
    print('-c --train_set\t指定训练数据的目录。默认为../data/seg_train')
    print('-v --val_set\t指定校验数据的目录。默认为../data/seg_val')
    print('-t --test_set\t指定测试数据的目录。默认为../data/test_val')
    print('-a --all_set\t任意打包数据，参数后跟若干目录')
    print('')
    print('')
    print('示例：')
    print('')
    print('制作数据集')
    print('./tfrecord.py')
    print('')
    print('自定义各数据集的目录')
    print('./tfrecord.py -r "../data/record/" -c "../data/train/" -v "../data/val/" -t "../data/test"')
    print('')
    print('打包若干数据')
    print('./tfrecord.py -r "../data/record/" -a "../data/train/" "../data/val/" "../data/test/"')
    print('')


if __name__ == '__main__':

    train_path = '../data/seg_train/'
    val_path = '../data/seg_val/'
    test_path = '../data/seg_test/'
    record_path = '../data/record/'
    is_make_all = False
    all_sets = []

    try:
        opts, args = getopt.gnu_getopt(sys.argv[1:], 'hr:c:v:t:a', ['help', 'tfrecord_path=',
                                                                    'train_set=', 'val_set=', 'test_set=', 'all_set'])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print_help()
            sys.exit(2)
        elif opt in ('-r', '--tfrecord_path'):
            record_path = arg
        elif opt in ('-c', '--train_set'):
            train_path = arg
        elif opt in ('-v', '--val_set'):
            val_path = arg
        elif opt in ('-t', '--test_set'):
            test_path = arg
        elif opt in ('-a', '--all_set'):
            is_make_all = True
        else:
            if is_make_all:
                all_sets.append(arg)
            else:
                print_help()
                sys.exit(2)
    if is_make_all:
        all_sets = args
        mk_tfrecord_all(os.path.join(record_path, "all.tfrecords"), all_sets)
    else:
        mk_tfrecord_v2(train_path, os.path.join(
            record_path + "train.tfrecords"))
        mk_tfrecord_v2(val_path, os.path.join(record_path, "val.tfrecords"))
        mk_tfrecord_v2(test_path, os.path.join(record_path, "test.tfrecords"))
