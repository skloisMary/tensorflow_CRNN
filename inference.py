import tensorflow as tf
import numpy as np
import cv2
import Model
import json
import argparse
import os

parser = argparse.ArgumentParser(description='train or test the CRNN model')

parser.add_argument('--bs', dest='batch_size', type=int, default=1,
                    help='size of a batch')
parser.add_argument('--epoch', dest='epoch', type=int, default=20000,
                    help='How many iteration in training')
# parser.add_argument('--d_p', dest='dataset_path',
#                     default='/home/mary/textGenation/english/train_dataset.tfrecord',
#                     help='where the image data is ')
parser.add_argument('--d_p', dest='dataset_path',
                    default='/home/mary/textGenation/english/test_dataset.tfrecords',
                    help='where the image data is ')
parser.add_argument('--inti_lr', dest='init_learning_rate', type=float, default=0.1,
                    help='the initial learning rate when gradient')
parser.add_argument('--early_stopping_step', dest='early_stopping_step', type=int,
                    default=2000)
parser.add_argument('--moder_dir', dest='model_dir', default='./model')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./saver')
parser.add_argument('--image_dir', dest='image_dir', default='./images')

args = parser.parse_args()


def sparse_matrix_to_list(sparse_matrix, char_map_dict):
    indices = sparse_matrix.indices
    values = sparse_matrix.values
    dense_shape = sparse_matrix.dense_shape

    dense_matrix = len(char_map_dict.keys()) * np.ones(dense_shape, dtype=np.int32)
    for i, indice in enumerate(indices):
        dense_matrix[indice[0], indice[1]] = values[i]

    string_list = []
    for row in dense_matrix:
        string = []
        for val in row:
            string.append(int_to_string(val, char_map_dict))
        string_list.append("".join(s for s in string if s != '*'))
    return string_list


def int_to_string(value, char_map_dict):
    for key in char_map_dict.keys():
        if char_map_dict[key] == int(value):
            return str(key)
        elif len(char_map_dict.keys()) == int(value):
            return ""


def inference(image_dir, checkpoint_dir, char_map_dict):
    crnn_net = Model.CRNN(batch_size=args.batch_size,
                          init_learning_rate=args.init_learning_rate,
                          epochs=args.epoch,
                          dataset_path=args.dataset_path,
                          early_stopping_step=args.early_stopping_step,
                          model_dir=args.model_dir,
                          checkpoint_dir=args.checkpoint_dir
                          )
    decoded = crnn_net.decoded
    saver = tf.train.Saver()
    saver_path = tf.train.latest_checkpoint(checkpoint_dir)

    with tf.Session() as session:
        saver.restore(sess=session, save_path=saver_path)

        # read images
        names_lists = os.listdir(image_dir)
        for image_name in names_lists:
            image_path = os.path.join(image_dir, image_name)
            image = cv2.imread(image_path)
            height, width, channel = image.shape
            ratio = 32 / float(height)
            width = int(width * ratio)
            image = cv2.resize(image, (width, 32))
            image = np.expand_dims(image, axis=0)
            image = np.array(image, dtype=np.float32)
            seq_len = np.array([width / 4], dtype=np.int32)

            pred = session.run(decoded, feed_dict={
                crnn_net.input_images: image,
                crnn_net.input_sequence_lengths: seq_len
            })
            pred = sparse_matrix_to_list(pred[0], char_map_dict)
            print('image label:', image_name.strip('.jpg'))
            print('prediced label:', pred[0])


def main(_):
    char_map_dict = json.load(open("./char_map.json", "r"))
    inference(args.image_dir, args.checkpoint_dir, char_map_dict)


if __name__ == '__main__':
    tf.app.run()
