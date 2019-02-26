# coding=utf-8
import tensorflow as tf
import os
import cv2
import random
import json
import sys

NUM_EXAMPLES_PER_EPOCH = 10000
RATIO = 0.7


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def generation_TFRecord(data_dir, tfrecord_dir):
    '''
    生成TFRecord文件
    @param data_dir: 数据所在文件夹
    @return:
    '''
    vocublary = json.load(open("./map.json", "r"))
    image_name_list = []
    for file in os.listdir(data_dir):
        if file.endswith('.jpg'):
            image_name_list.append(file)

    random.shuffle(image_name_list)
    # capacity = len(image_name_list)

    # 生成train tfrecord文件
    train_writer = tf.python_io.TFRecordWriter(os.path.join(tfrecord_dir, 'train_dataset.tfrecord'))
    train_image_name_list = image_name_list[0: int(RATIO * NUM_EXAMPLES_PER_EPOCH)]
    for train_name in train_image_name_list:
        train_image_label = []
        for s in train_name.strip('.jpg'):
            train_image_label.append(vocublary[s])

        train_image_raw = cv2.imread(os.path.join(data_dir, train_name), 1)
        if train_image_raw is None:
            continue
        height, width, channel = train_image_raw.shape
        ratio = 32 / float(height)
        train_image = cv2.resize(train_image_raw, (int(width * ratio), 32))
        # 将图片格式转换(编码)成流数据，赋值到内存缓存中;主要用于图像数据格式的压缩，方便网络传输。
        is_success, train_image_buffer = cv2.imencode('.jpg', train_image)
        if not is_success:
            continue
        train_image_byte = train_image_buffer.tostring()

        train_example = tf.train.Example(features=tf.train.Features(feature={
            'label': int64_list_feature(train_image_label),
            'image': bytes_feature(train_image_byte)}))
        train_writer.write(train_example.SerializeToString())
        sys.stdout.flush()
    sys.stdout.flush()
    train_writer.close()

    # 生成test tfrecord文件
    test_writer = tf.python_io.TFRecordWriter(os.path.join(tfrecord_dir, 'test_dataset.tfrecord'))
    test_image_name_list = image_name_list[int(NUM_EXAMPLES_PER_EPOCH * RATIO):NUM_EXAMPLES_PER_EPOCH]
    for test_name in test_image_name_list:
        test_image_label = []
        for s in test_name.strip('.jpg'):
            test_image_label.append(vocublary[s])

        # 以彩色图像方式读取
        test_image_raw = cv2.imread(os.path.join(data_dir, test_name), 1)
        if test_image_raw is None:
            continue

        height, width, channel = test_image_raw.shape
        ratio = 32 / float(height)
        test_image = cv2.resize(test_image_raw, (int(width * ratio), 32))
        is_success, test_image_buffer = cv2.imencode('.jpg', test_image)
        if not is_success:
            continue
        test_image_byte = test_image_buffer.tostring()

        test_example = tf.train.Example(features=tf.train.Features(feature={
            'label': int64_list_feature(test_image_label),
            'image': bytes_feature(test_image_byte)}))
        test_writer.write(test_example.SerializeToString())
    test_writer.close()


def read_tfrecord(filename, batch_size, is_train=True):
    if not os.path.exists(filename):
        raise ValueError('connot find tfrecord file in path')

    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialize_example = reader.read(filename_queue)
    image_features = tf.parse_single_example(serialized=serialize_example,
                                             features={
                                                 'label': tf.VarLenFeature(dtype=tf.int64),
                                                 'image': tf.FixedLenFeature([], tf.string)
                                             })
    image = tf.image.decode_jpeg(image_features['image'])
    image.set_shape([32, None, 3])
    image = tf.cast(image, tf.float32)

    label = tf.cast(image_features['label'], tf.int32)
    sequence_length = tf.cast(tf.shape(image)[-2] / 4, tf.int32)

    if is_train is True:
        train_image_batch, train_label_batch, train_sequence_length = tf.train.batch([image, label, sequence_length],
                                                                                     batch_size=batch_size,
                                                                                     dynamic_pad=True,
                                                                                     num_threads=4,
                                                                                     capacity=1000 + 3 * batch_size)
        return train_image_batch, train_label_batch, train_sequence_length
    else:
        test_image_batch, test_label_batch, test_sequence_length = tf.train.batch([image, label, sequence_length],
                                                                                  batch_size=batch_size,
                                                                                  dynamic_pad=True,
                                                                                  capacity=1000 + 3 * batch_size)
        return test_image_batch, test_label_batch, test_sequence_length


def main(argv):
    # print(vocublary)
    data_dir = "/home/mary/textGenation/english/Train_en_10000/Train"
    tfrecord_dir = '/home/mary/textGenation/english/Train_en_10000/'
    generation_TFRecord(data_dir, tfrecord_dir)
    tfrecord_files = os.path.join(tfrecord_dir, 'train_dataset.tfrecord')
    train_image, train_label, train_seq_length = read_tfrecord(tfrecord_files, 32)
    dense_label = tf.sparse_tensor_to_dense(train_label)

    with tf.Session() as session:
        session.run(tf.group(tf.global_variables_initializer(),
                             tf.local_variables_initializer()))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coord)
        for i in range(2):
            t_image, t_label, t_seq_len, t_dense_label = session.run([train_image, train_label,
                                                                      train_seq_length, dense_label])
            print(t_dense_label)
            print(t_seq_len)

        coord.request_stop()
        coord.join(threads=threads)


if __name__ == '__main__':
    tf.app.run()
