import tensorflow as tf
import numpy as np
import tempfile
import os
import sys
import time
import getopt
from data_preprocessing.tfrecord import read_and_decode, get_data, clear_dir
from model import VggNetModel


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)


image_summary = tf.summary.image
scalar_summary = tf.summary.scalar
histogram_summary = tf.summary.histogram
merge_summary = tf.summary.merge
SummaryWriter = tf.summary.FileWriter


def inference(x, training=False):
    keep_prob = tf.placeholder(tf.float32)
    # conv1_1
    with tf.variable_scope('conv1_1') as scope:
        kernel = tf.get_variable('weights', initializer=tf.truncated_normal(
            [3, 3, 3, 64], dtype=tf.float32, stddev=1e-1))
        conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', initializer=tf.constant(
            0.0, shape=[64], dtype=tf.float32))
        out = tf.nn.bias_add(conv, biases)
        conv1_1 = tf.nn.relu(out, name=scope.name)

    # conv1_2
    with tf.variable_scope('conv1_2') as scope:
        kernel = tf.get_variable('weights', initializer=tf.truncated_normal(
            [3, 3, 64, 64], dtype=tf.float32, stddev=1e-1))
        conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', initializer=tf.constant(
            0.0, shape=[64], dtype=tf.float32))
        out = tf.nn.bias_add(conv, biases)
        conv1_2 = tf.nn.relu(out, name=scope.name)

    # pool1
    pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[
                           1, 2, 2, 1], padding='SAME', name='pool1')

    # conv2_1
    with tf.variable_scope('conv2_1') as scope:
        kernel = tf.get_variable('weights', initializer=tf.truncated_normal(
            [3, 3, 64, 128], dtype=tf.float32, stddev=1e-1))
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', initializer=tf.constant(
            0.0, shape=[128], dtype=tf.float32))
        out = tf.nn.bias_add(conv, biases)
        conv2_1 = tf.nn.relu(out, name=scope.name)

    # conv2_2
    with tf.variable_scope('conv2_2') as scope:
        kernel = tf.get_variable('weights', initializer=tf.truncated_normal(
            [3, 3, 128, 128], dtype=tf.float32, stddev=1e-1))
        conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', initializer=tf.constant(
            0.0, shape=[128], dtype=tf.float32))
        out = tf.nn.bias_add(conv, biases)
        conv2_2 = tf.nn.relu(out, name=scope.name)

    # pool2
    pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[
                           1, 2, 2, 1], padding='SAME', name='pool2')

    # conv3_1
    with tf.variable_scope('conv3_1') as scope:
        kernel = tf.get_variable('weights', initializer=tf.truncated_normal(
            [3, 3, 128, 256], dtype=tf.float32, stddev=1e-1))
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', initializer=tf.constant(
            0.0, shape=[256], dtype=tf.float32))
        out = tf.nn.bias_add(conv, biases)
        conv3_1 = tf.nn.relu(out, name=scope.name)

    # conv3_2
    with tf.variable_scope('conv3_2') as scope:
        kernel = tf.get_variable('weights', initializer=tf.truncated_normal(
            [3, 3, 256, 256], dtype=tf.float32, stddev=1e-1))
        conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', initializer=tf.constant(
            0.0, shape=[256], dtype=tf.float32))
        out = tf.nn.bias_add(conv, biases)
        conv3_2 = tf.nn.relu(out, name=scope.name)

    # conv3_3
    with tf.variable_scope('conv3_3') as scope:
        kernel = tf.get_variable('weights', initializer=tf.truncated_normal(
            [3, 3, 256, 256], dtype=tf.float32, stddev=1e-1))
        conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', initializer=tf.constant(
            0.0, shape=[256], dtype=tf.float32))
        out = tf.nn.bias_add(conv, biases)
        conv3_3 = tf.nn.relu(out, name=scope.name)

    # pool3
    pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[
                           1, 2, 2, 1], padding='SAME', name='pool3')

    # conv4_1
    with tf.variable_scope('conv4_1') as scope:
        kernel = tf.get_variable('weights', initializer=tf.truncated_normal(
            [3, 3, 256, 512], dtype=tf.float32, stddev=1e-1))
        conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', initializer=tf.constant(
            0.0, shape=[512], dtype=tf.float32))
        out = tf.nn.bias_add(conv, biases)
        conv4_1 = tf.nn.relu(out, name=scope.name)

    # conv4_2
    with tf.variable_scope('conv4_2') as scope:
        kernel = tf.get_variable('weights', initializer=tf.truncated_normal(
            [3, 3, 512, 512], dtype=tf.float32, stddev=1e-1))
        conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', initializer=tf.constant(
            0.0, shape=[512], dtype=tf.float32))
        out = tf.nn.bias_add(conv, biases)
        conv4_2 = tf.nn.relu(out, name=scope.name)

    # conv4_3
    with tf.variable_scope('conv4_3') as scope:
        kernel = tf.get_variable('weights', initializer=tf.truncated_normal(
            [3, 3, 512, 512], dtype=tf.float32, stddev=1e-1))
        conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', initializer=tf.constant(
            0.0, shape=[512], dtype=tf.float32))
        out = tf.nn.bias_add(conv, biases)
        conv4_3 = tf.nn.relu(out, name=scope.name)

    # pool4
    pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[
                           1, 2, 2, 1], padding='SAME', name='pool4')

    # conv5_1
    with tf.variable_scope('conv5_1') as scope:
        kernel = tf.get_variable('weights', initializer=tf.truncated_normal(
            [3, 3, 512, 512], dtype=tf.float32, stddev=1e-1))
        conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', initializer=tf.constant(
            0.0, shape=[512], dtype=tf.float32))
        out = tf.nn.bias_add(conv, biases)
        conv5_1 = tf.nn.relu(out, name=scope.name)

    # conv5_2
    with tf.variable_scope('conv5_2') as scope:
        kernel = tf.get_variable('weights', initializer=tf.truncated_normal(
            [3, 3, 512, 512], dtype=tf.float32, stddev=1e-1))
        conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', initializer=tf.constant(
            0.0, shape=[512], dtype=tf.float32))
        out = tf.nn.bias_add(conv, biases)
        conv5_2 = tf.nn.relu(out, name=scope.name)

    # conv5_3
    with tf.variable_scope('conv5_3') as scope:
        kernel = tf.get_variable('weights', initializer=tf.truncated_normal(
            [3, 3, 512, 512], dtype=tf.float32, stddev=1e-1))
        conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', initializer=tf.constant(
            0.0, shape=[512], dtype=tf.float32))
        out = tf.nn.bias_add(conv, biases)
        conv5_3 = tf.nn.relu(out, name=scope.name)

    # pool5
    pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[
                           1, 2, 2, 1], padding='SAME', name='pool5')

    # fc6
    with tf.variable_scope('fc6') as scope:
        shape = int(np.prod(pool5.get_shape()[1:]))
        fc6w = tf.get_variable('weights', initializer=tf.truncated_normal(
            [shape, 4096], dtype=tf.float32, stddev=1e-1))
        fc6b = tf.get_variable('biases', initializer=tf.constant(
            1.0, shape=[4096], dtype=tf.float32))
        pool5_flat = tf.reshape(pool5, [-1, shape])
        fc6l = tf.nn.bias_add(tf.matmul(pool5_flat, fc6w), fc6b)
        fc6 = tf.nn.relu(fc6l)

        if training:
            fc6 = tf.nn.dropout(fc6, keep_prob)

    # fc7
    with tf.variable_scope('fc7') as scope:
        fc7w = tf.get_variable('weights', initializer=tf.truncated_normal(
            [4096, 4096], dtype=tf.float32, stddev=1e-1))
        fc7b = tf.get_variable('biases', initializer=tf.constant(
            1.0, shape=[4096], dtype=tf.float32))
        fc7l = tf.nn.bias_add(tf.matmul(fc6, fc7w), fc7b)
        fc7 = tf.nn.relu(fc7l)

        if training:
            fc7 = tf.nn.dropout(fc7, keep_prob)

    # fc8
    with tf.variable_scope('fc8') as scope:
        fc8w = tf.get_variable('weights', initializer=tf.truncated_normal(
            [4096, 6], dtype=tf.float32, stddev=1e-1))
        fc8b = tf.get_variable('biases', initializer=tf.constant(
            1.0, shape=[6], dtype=tf.float32))
        score = tf.nn.bias_add(tf.matmul(fc7, fc8w), fc8b)

    return score, keep_prob


def cnn(x):
    keep_prob = tf.placeholder(tf.float32)

    '''第一层卷积'''
    with tf.name_scope("conv1"):
        W_conv1 = weight_variable([3, 3, 3, 256])
        b_conv1 = bias_variable([256])
        h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)

    '''第一层池化'''
    with tf.name_scope("pool1"):
        h_pool1 = max_pool(h_conv1)

    '''output: [batch_size,64,64,256]'''

    '''第二层卷积'''
    with tf.name_scope("conv2"):
        W_conv2 = weight_variable([3, 3, 256, 512])
        b_conv2 = bias_variable([512])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_conv2 = tf.nn.dropout(h_conv2, keep_prob)

    '''第二层池化'''
    with tf.name_scope("pool2"):
        h_pool2 = max_pool(h_conv2)

    '''output: [batch_size,32,32,512]'''

    '''第三层卷积'''
    with tf.name_scope("conv3"):
        W_conv3 = weight_variable([3, 3, 512, 512])
        b_conv3 = bias_variable([512])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

    '''第三层池化'''
    with tf.name_scope("pool3"):
        h_pool3 = max_pool(h_conv3)

    '''output: [batch_size,16,16,512]'''

    '''第四层卷积'''
    with tf.name_scope("conv4"):
        W_conv4 = weight_variable([3, 3, 512, 256])
        b_conv4 = bias_variable([256])
        h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)

    '''第四层池化'''
    with tf.name_scope("pool4"):
        h_pool4 = max_pool(h_conv4)

    '''output: [batch_size,8,8,256]'''

    '''第五层全连接层'''
    with tf.name_scope("fc1"):
        W_fc1 = weight_variable([8*8*256, 1024])
        b_fc1 = bias_variable([1024])
        h_pool4_flat = tf.reshape(h_pool4, [-1, 8*8*256])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)

    '''dropout'''
    with tf.name_scope("dropout1"):
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    '''output: [batch_size,1024]'''

    '''第六层全连接层'''
    with tf.name_scope("fc2"):
        W_fc2 = weight_variable([1024, 6])
        b_fc2 = bias_variable([6])
        output = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    '''output: [batch_size,6]'''

    return output, keep_prob


def Loss(output, label):
    with tf.name_scope("loss"):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=label, logits=output)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss_summary = scalar_summary('loss', cross_entropy_mean)
    return cross_entropy_mean


def Optimizer(output, label):
    cross_entropy_mean = Loss(output, label)
    with tf.name_scope("Adam_Optimizer"):
        optimizer = tf.train.AdamOptimizer(0.001).minimize(cross_entropy_mean)
    return optimizer


def Evaluator(output, label):
    with tf.name_scope("Accuracy"):
        correct_prediction = tf.equal(
            tf.argmax(label, 1), tf.argmax(output, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)

        accuracy = tf.reduce_mean(correct_prediction)
        acc_summary = scalar_summary('accuracy', accuracy)
    return accuracy


def train(iter_num=20, log_iter_step=10, batch_size=32, is_restore=False, restore_path=''):
    # 总迭代次数
    # 输出间隔
    class_num = 6
    train_tfrecord_path = './data/record/train.tfrecords'
    val_tfrecord_path = './data/record/val.tfrecords'
    test_tfrecord_path = './data/record/test.tfrecords'
    model_path = './model/'
    vgg = VggNetModel(6, 0.5)
    # clear_dir(model_path)

    x = tf.placeholder(tf.float32, [None, 128, 128, 3])

    label = tf.placeholder(tf.float32, [None, class_num])
    #output, keep_prob = cnn(x)
    output, keep_prob = inference(x, True)

    # 优化器
    optimizer = Optimizer(output, label)

    # 评估器
    accuracy = Evaluator(output, label)

    # 误差
    loss = Loss(output, label)

    # 读取数据
    train_x, train_label = get_data(
        train_tfrecord_path, batch_size=batch_size, one_hot=True, label_num=class_num)
    val_x, val_label = get_data(
        val_tfrecord_path, batch_size=batch_size, one_hot=True, label_num=class_num)
    test_x, test_label = get_data(
        test_tfrecord_path, batch_size=batch_size, one_hot=True, label_num=class_num)

    '''保存模型'''
    saver = tf.train.Saver(max_to_keep=5)

    merge = tf.summary.merge_all()

    max_accuracy = 0.0
    with tf.Session() as sess:
        writer = tf.summary.FileWriter("./logs/", sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(iter_num+1):
            start_time = time.time()
            if(is_restore):
                saver.restore(sess, model_path+restore_path)
            input_train_x, input_train_label = sess.run([train_x, train_label])

            # 训练一步
            optimizer.run(session=sess, feed_dict={
                          x: input_train_x, label: input_train_label, keep_prob: 0.5})

            if i % log_iter_step == 0:

                input_val_x, input_val_label = sess.run([val_x, val_label])

                loss_value = loss.eval(session=sess, feed_dict={
                                       x: input_train_x, label: input_train_label, keep_prob: 1.0})
                train_accuracy = accuracy.eval(session=sess, feed_dict={
                                               x: input_train_x, label: input_train_label, keep_prob: 1.0})
                val_accuracy = accuracy.eval(session=sess, feed_dict={
                                             x: input_val_x, label: input_val_label, keep_prob: 1.0})
                print('===step {},loss value is {},train accuracy is {},val accuracy is {}==='.format(
                    i, loss_value, train_accuracy, val_accuracy))

                if val_accuracy > max_accuracy:
                    max_accuracy = val_accuracy
                    saver.save(sess, model_path + 'model-' + str(i) + '.ckpt')

                summary = sess.run(merge, feed_dict={
                                   x: input_train_x, label: input_train_label, keep_prob: 1.0})
                writer.add_summary(summary, i)

        input_test_x, input_test_label = sess.run([test_x, test_label])

        test_accuracy = accuracy.eval(session=sess, feed_dict={
                                      x: input_test_x, label: input_test_label, keep_prob: 1.0})
        print('===test accuracy is {}==='.format(test_accuracy))

        coord.request_stop()
        coord.join(threads)
        # sess.close()


def predict_model(test_tfrecord_path, restore_path, batch_size=32, class_num=6):

    model_path = './model/'
    x = tf.placeholder(tf.float32, [None, 128, 128, 3])
    label = tf.placeholder(tf.float32, [None, class_num])
    output, keep_prob = cnn(x)

    accuracy = Evaluator(output, label)

    test_x, test_label = get_data(
        test_tfrecord_path, batch_size=batch_size, one_hot=True, label_num=class_num)

    saver = tf.train.Saver(max_to_keep=5)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        saver.restore(sess, model_path+restore_path)
        input_test_x, input_test_label = sess.run([test_x, test_label])

        test_accuracy = accuracy.eval(session=sess, feed_dict={
            x: input_test_x, label: input_test_label, keep_prob: 1.0})
        print('===test accuracy is {}==='.format(test_accuracy))

        coord.request_stop()
        coord.join(threads)


def print_help():
    print('训练NNIIC网络')
    print('')
    print('./cnn.py [-h] [-r model_path] [-i iteration_times] [-l logging_step] [-b batch_size] [-p predict_record_path]')
    print('')
    print('参数：')
    print('-r --restore\t要加载的模型存档。如果不指定，则从头开始训练模型')
    print('-i --iteration\t训练的迭代次数。默认为10次')
    print('-l --logging_iteration_step\t记录数据的间隔。默认为10步')
    print('-b --batch_size\t每批的大小。默认为32')
    print('-p --predict_path\t使用模型对指定的数据集进行预测。如果不指定，则表示训练模型。注意此参数需要配合--restore参数加载模型存档')
    print('')
    print('')
    print('示例：')
    print('')
    print('以默认配置从头开始训练模型：')
    print('./cnn.py')
    print('')
    print('从头开始训练模型，迭代1000次，每批32：')
    print('./cnn.py -i 1000 -b 32')
    print('')
    print('从记录model.ckpt加载模型存档训练模型，迭代1000次，每批64：')
    print('./cnn.py -r model.ckpt -i 1000 -b 64')
    print('')
    print('使用模型model.ckpt对数据集all.tfrecords进行预测，每批64：')
    print('./cnn.py -r model.ckpt -b 64 -p ./all.tfrecords')


if __name__ == '__main__':
    is_restore = False
    restore_path = ''
    iter_num = 10
    log_iter = 10
    batch_size = 32
    is_predict = False
    predict_path = ''
    try:
        opts, args = getopt.gnu_getopt(sys.argv[1:], 'hr:i:l:b:p:', ['help',
                                                                     'restore=', 'iteration=', 'logging_iteration_step=', 'batch_size=', 'predict_path='])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print_help()
            sys.exit(2)
        elif opt in ('-r', '--restore'):
            is_restore = True
            restore_path = arg
        elif opt in ('-i', '--iteration'):
            iter_num = int(arg)
        elif opt in ('-l', '--logging_interation_step'):
            log_iter = int(arg)
        elif opt in ('-b', '--batch_size'):
            batch_size = int(arg)
        elif opt in ('-p', '--predict_path'):
            is_predict = True
            predict_path = arg
        else:
            print_help()
            sys.exit(2)
    if is_predict:
        if restore_path != '':
            predict_model(predict_path, restore_path, batch_size)
        else:
            print_help()
    else:
        train(iter_num, log_iter, batch_size, is_restore, restore_path)
