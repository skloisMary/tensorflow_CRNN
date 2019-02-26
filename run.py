# encoding = "utf8"
import tensorflow as tf
import argparse
from Model import CRNN
import os

parser = argparse.ArgumentParser(description='train or test the CRNN model')

# parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--phase', dest='phase', default='test', help='train or test')
parser.add_argument('--bs', dest='batch_size', type=int, default=32,
                    help='size of a batch')
parser.add_argument('--epoch', dest='epoch', type=int, default=10000,
                    help='How many iteration in training')
# parser.add_argument('--d_p', dest='dataset_path',
#                     default='/home/mary/textGenation/english/Train_en_10000/train_dataset.tfrecord',
#                     help='where the image data is ')
parser.add_argument('--d_p', dest='dataset_path',
                    default='/home/mary/textGenation/english/Train_en_10000/test_dataset.tfrecord',
                    help='where the image data is ')
parser.add_argument('--inti_lr', dest='init_learning_rate', type=float, default=0.1,
                    help='the initial learning rate when gradient')
parser.add_argument('--early_stopping_step', dest='early_stopping_step', type=int,
                    default=2000)
parser.add_argument('--moder_dir', dest='model_dir', default='./model')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./saver')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def main(argv):
    crnn = CRNN(batch_size=args.batch_size,
                init_learning_rate=args.init_learning_rate,
                epochs=args.epoch,
                dataset_path=args.dataset_path,
                early_stopping_step=args.early_stopping_step,
                model_dir=args.model_dir,
                checkpoint_dir=args.checkpoint_dir
                )
    if args.phase is 'train':
        crnn.train()
    elif args.phase is 'test':
        crnn.test()


if __name__ == '__main__':
    tf.app.run()
