import tensorflow as tf
import os
import json


def extract(data_dir):
    character_list = []
    for names in os.listdir(data_dir):
        name = names.strip('.jpg')
        for c in name:
            character_list.append(c)
    return sorted(list(set(character_list)))


def make_dictionary(character_list):
    vocabulary = {}
    index = 0
    for char in character_list:
        vocabulary[char] = index
        index += 1
    return vocabulary


def main(_):
    data_dir = '/home/mary/textGenation/english/Train_en_10000/Train'
    character = extract(data_dir)
    vocabulary = make_dictionary(character_list=character)
    with open('map.json', 'w') as f:
        json.dump(vocabulary, f)
        print('completely!')


if __name__ == '__main__':
    tf.app.run()
