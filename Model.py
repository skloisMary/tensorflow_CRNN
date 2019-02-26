import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from TFrecorde import read_tfrecord, NUM_EXAMPLES_PER_EPOCH, RATIO
import os
import cv2
import json
import time


class CRNN(object):
    def __init__(self, batch_size, init_learning_rate, dataset_path, epochs,
                 early_stopping_step, model_dir, checkpoint_dir):
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.epochs = epochs
        # # early stop
        # self.early_stopping_step = early_stopping_step
        # self.should_early_stop = False
        # self.step = 0
        #
        self.model_dir = model_dir
        self.checkpoint_dir = checkpoint_dir

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.char_map_dict = json.load(open('./map.json', 'r'))
        self.num_classes = len(self.char_map_dict.keys()) + 1

        # 设置placeholder
        self.input_images = tf.placeholder(tf.float32, shape=[self.batch_size, 32, None, 3], name='input_images')
        self.input_labels = tf.sparse_placeholder(tf.int32, name='input_labels')
        self.input_sequence_lengths = tf.placeholder(tf.int32, shape=[self.batch_size], name='input_sequence_length')

        # network
        self.ouputs = self.build_network(self.input_images,
                                         self.input_sequence_lengths)

        # learning_rate
        self.global_step = tf.train.create_global_step()
        self.learning_rate = tf.train.exponential_decay(learning_rate=init_learning_rate, global_step=self.global_step,
                                                        decay_rate=0.8, decay_steps=1000, staircase=True)
        tf.summary.scalar('learning_rate', self.learning_rate)

        # computer the CTC(Connectionist Temporal Classification) Loss
        self.loss = tf.reduce_mean(tf.nn.ctc_loss(labels=self.input_labels, inputs=self.ouputs,
                                                  sequence_length=self.input_sequence_lengths,
                                                  ignore_longer_outputs_than_inputs=True))
        tf.summary.scalar('ctc_loss', self.loss)
        # optimizer
        self.optimizer = tf.train.AdadeltaOptimizer(self.learning_rate).minimize(self.loss, self.global_step)

        #
        self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(self.ouputs, self.input_sequence_lengths,
                                                                    merge_repeated=False)

        # tf.edit_distance()计算序列之间的编辑距离
        self.sequence_distance = tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.input_labels))
        tf.summary.scalar('seq_distance', self.sequence_distance)

        # summary
        self.summary_op = tf.summary.merge_all()

    def build_network(self, input, input_sequence_lengths):
        cnn_output = self.CNN_VGG(input)
        sequence_out = self.map_to_sequence(cnn_output)
        net_out = self.RNN(sequence_out, input_sequence_lengths)
        return net_out

    def CNN_VGG(self, inputs):
        ''' CNN extract feature from each input image, 网络架构选择的是VGG(CRNN)
        @param inputs: the input image
        @return: feature maps
        '''
        with tf.variable_scope('VGG_CNN'):
            conv1 = tf.layers.conv2d(inputs=inputs, filters=64, kernel_size=(3, 3),
                                     padding='SAME', activation=tf.nn.relu, name='conv_1')
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=(2, 2), strides=2, name='pool_1')

            #
            conv2 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=(3, 3),
                                     padding='SAME', activation=tf.nn.relu, name='conv_2')
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=(2, 2), strides=2, name='pool_2')

            #
            conv3 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=(3, 3),
                                     padding='SAME', activation=tf.nn.relu, name='conv_3')
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=(2, 1), strides=(2, 1), name='pool_3')

            #
            conv4 = tf.layers.conv2d(inputs=pool3, filters=256, kernel_size=(3, 3),
                                     padding='SAME', activation=tf.nn.relu, name='conv_4')
            bn1 = tf.layers.batch_normalization(conv4, training=True, name='bn1')
            conv5 = tf.layers.conv2d(inputs=bn1, filters=512, kernel_size=(3, 3),
                                     padding='SAME', activation=tf.nn.relu, name='conv_5')
            bn2 = tf.layers.batch_normalization(conv5, training=True, name='bn_2')
            pool4 = tf.layers.max_pooling2d(inputs=bn2, pool_size=(2, 1), strides=(2, 1), name='pool_5')

            #
            conv7 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=(2, 1),
                                     padding='VALID', activation=tf.nn.relu, name='conv_6')
            # print('conv_7', conv7.shape)
        return conv7

    def map_to_sequence(self, input_tensor):
        return tf.squeeze(input_tensor, axis=1)

    def RNN(self, input, seq_len):
        with tf.variable_scope('BiLSTM_1'):
            lstm_fw_cell_1 = rnn.BasicLSTMCell(256)
            lstm_bw_cell_1 = rnn.BasicLSTMCell(256)
            inter_output, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_1,
                                                              lstm_bw_cell_1,
                                                              input, seq_len,
                                                              dtype=tf.float32)
            inter_output = tf.concat(inter_output, 2)
        with tf.variable_scope('BiLSTM_2'):
            lstm_fw_cell_2 = rnn.BasicLSTMCell(256)
            lstm_bw_cell_2 = rnn.BasicLSTMCell(256)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_2,
                                                         lstm_bw_cell_2,
                                                         inter_output, seq_len,
                                                         dtype=tf.float32)
            rnn_output = tf.concat(outputs, 2)
        rnn_reshaped = tf.reshape(rnn_output, shape=[-1, 512])
        # doing the affine projection
        softmax_w = tf.Variable(tf.truncated_normal(shape=[512, self.num_classes], stddev=0.01), name='weight_w')
        logits = tf.matmul(rnn_reshaped, softmax_w)
        logits = tf.reshape(logits, shape=[self.batch_size, -1, self.num_classes])
        # final layer, the output of BLSTM
        net_out = tf.transpose(logits, (1, 0, 2), name='transpose_time_major')
        return net_out

    def sparse_matrix_to_list(self, sparse_matrix):
        indices = sparse_matrix.indices
        values = sparse_matrix.values
        dense_shape = sparse_matrix.dense_shape

        dense_matrix = len(self.char_map_dict.keys()) * np.ones(dense_shape, dtype=np.int32)
        for i, indice in enumerate(indices):
            dense_matrix[indice[0], indice[1]] = values[i]

        string_list = []
        for row in dense_matrix:
            string = []
            for val in row:
                string.append(self.int_to_string(val))
            string_list.append("".join(s for s in string if s != '*'))
        return string_list

    def int_to_string(self, value):
        for key in self.char_map_dict.keys():
            if self.char_map_dict[key] == int(value):
                return str(key)
            elif len(self.char_map_dict.keys()) == int(value):
                return ""

    def train(self):
        image, label, seq_len_batch = read_tfrecord(self.dataset_path, self.batch_size)
        saver = tf.train.Saver()
        # checkpoint
        train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        model_name = 'crnn_ctc_ocr_{:s}.ckpt'.format(str(train_start_time))
        model_save_path = os.path.join(self.checkpoint_dir, model_name)
        #
        with tf.Session() as session:
            # log
            summary_writer = tf.summary.FileWriter(self.model_dir, session.graph)

            session.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=session, coord=coord)

            print('start training')
            for index in range(self.epochs):
                #
                batch_image, batch_label, batch_seq_length = session.run(
                    [image, label, seq_len_batch])
                #
                _, loss, lr, seq_distance, decodeds, summary = session.run(
                    [self.optimizer, self.loss, self.learning_rate,
                     self.sequence_distance, self.decoded, self.summary_op],
                    feed_dict={self.input_images: batch_image,
                               self.input_labels: batch_label,
                               self.input_sequence_lengths: batch_seq_length})
                #
                if index % 100 == 0:
                    preds = self.sparse_matrix_to_list(decodeds[0])
                    gt_labels = self.sparse_matrix_to_list(batch_label)
                    accuracy = []
                    for j, gt_label in enumerate(gt_labels):
                        pred = preds[j]
                        #
                        if index % 2000 == 0:
                            print('prediction:', pred)
                            print('grouth_truth_label:', gt_label)
                        #
                        total_count = len(gt_label)
                        correct_count = 0
                        try:
                            for i, lab in enumerate(gt_label):
                                if lab == pred[i]:
                                    correct_count += 1
                        except IndexError:
                            continue
                        finally:
                            try:
                                accuracy.append(correct_count / total_count)
                            except ZeroDivisionError:
                                if len(pred) == 0:
                                    accuracy.append(1)
                                else:
                                    accuracy.append(0)
                    accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)
                    print('epoches:', index, ' loss:', loss, ' seq_distance:', seq_distance,
                          ' learning_rate:', lr, ' accuracy:', accuracy)
                    summary_writer.add_summary(summary=summary, global_step=index)
                if (index + 1) % 5000 == 0:
                    saver.save(sess=session, save_path=model_save_path, global_step=index)

                # #
                # if seq_distance == 0:
                #     self.step += 1
                # #
                # if self.step >= self.early_stopping_step:
                #     self.should_early_stop = True
                #     print('early stopping is trigger at step :', index)
                # #
                # if self.should_early_stop is True:
                #     saver.save(sess=session, save_path=model_save_path, global_step=index)
                #     break

            summary_writer.close()
            coord.request_stop()
            coord.join(threads=threads)

    def test(self):
        print('testing!')
        image, label, seq_len_batch = read_tfrecord(self.dataset_path, self.batch_size, is_train=False)
        saver = tf.train.Saver()
        saver_path = tf.train.latest_checkpoint(self.checkpoint_dir)

        #
        # test_sample_count = NUM_EXAMPLES_PER_EPOCH - int(RATIO * NUM_EXAMPLES_PER_EPOCH)
        test_sample_count = int(RATIO * NUM_EXAMPLES_PER_EPOCH)
        step_num = test_sample_count // self.batch_size
        print('iteration:', step_num)

        sess_config = tf.ConfigProto()
        with tf.Session(config=sess_config) as session:
            saver.restore(sess=session, save_path=saver_path)
            #

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=session, coord=coord)
            #
            mean_accuracy = []
            for index in range(step_num):
                batch_image, batch_label, batch_seq_length = session.run([image, label, seq_len_batch])
                decodes = session.run(self.decoded, feed_dict={
                    self.input_images: batch_image,
                    self.input_labels: batch_label,
                    self.input_sequence_lengths: batch_seq_length
                })
                preds = self.sparse_matrix_to_list(decodes[0])
                gt_labels = self.sparse_matrix_to_list(batch_label)

                #
                accuracy = []
                for j, gt_label in enumerate(gt_labels):
                    pred = preds[j]
                    print('predict label:', pred)
                    print('grouth_label:', gt_label)
                    #
                    total_count = len(gt_label)
                    # print('total_count:', total_count)
                    correct_count = 0
                    try:
                        for i, lab in enumerate(gt_label):
                            if lab == pred[i]:
                                correct_count += 1
                    except IndexError:
                        continue
                    finally:
                        try:
                            accuracy.append(correct_count / total_count)
                        except ZeroDivisionError:
                            if len(pred) == 0:
                                accuracy.append(1)
                            else:
                                accuracy.append(0)
                accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)
                mean_accuracy.append(accuracy)
                print('index:', index, 'test accuracy is:', accuracy)
            mean_accuracy = np.mean(np.array(mean_accuracy).astype(np.float32), axis=0)
            print('the final mean accuracy:', mean_accuracy)
            coord.request_stop()
            coord.join(threads=threads)
