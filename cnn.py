import tensorflow as tf
import tempfile
import os
import sys
import time
import getopt
from data_preprocessing.tfrecord import read_and_decode, get_data, clear_dir


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

    # clear_dir(model_path)

    x = tf.placeholder(tf.float32, [None, 128, 128, 3])

    label = tf.placeholder(tf.float32, [None, class_num])
    output, keep_prob = cnn(x)

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


if __name__ == '__main__':
    is_restore = False
    restore_path = ''
    iter_num = 10
    log_iter = 10
    batch_size = 32
    try:
        opts, args = getopt.gnu_getopt(sys.argv[1:], 'r:i:l:b:', [
                                   'restore=', 'iteration=', 'logging_iteration_step=', 'batch_size='])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-r', '--restore'):
            is_restore = True
            restore_path = arg
        elif opt in ('-i', '--iteration'):
            iter_num = int(arg)
        elif opt in ('-l', '--logging_interation_step'):
            log_iter = int(arg)
        elif opt in ('-b', '--batch_size'):
            batch_size = int(arg)
    train(iter_num, log_iter, batch_size, is_restore, restore_path)
