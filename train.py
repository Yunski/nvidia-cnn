import argparse
import os
import shutil
import time
import math
import pandas as pd
import matplotlib.pyplot as plt

from cnn import CNN
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

"""
Train CNN. 
Adapted from https://github.com/naokishibuya/car-behavioral-cloning/blob/master/model.py
"""

def load_data(args):
    """
    load simulator driving training data
    """
    df = pd.read_csv(os.path.join(os.getcwd(), args.data_dir, 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
    X = df[['center', 'left', 'right']].values
    Y = df['steering'].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=args.test_size)
    X_train, Y_train = shuffle(X_train, Y_train)
    X_test, Y_test = shuffle(X_test, Y_test)
    return X_train, X_test, Y_train, Y_test

def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def main():
    """
    Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='training/track1')
    parser.add_argument('-s', help='save directory',        dest='save_dir',          type=str,   default='saved_models/track1')
    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
    parser.add_argument('-p', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',      dest='epochs',            type=int,   default=10)
    parser.add_argument('-m', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=20000)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=40)
    parser.add_argument('-v', help='number of validation batches', dest='num_validation_batches', type=int, default=18)
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-4)
    args = parser.parse_args()

    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    if not os.path.exists(args.data_dir):
        raise Exception("Data directory does not exist.") 
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    else:
        shutil.rmtree(args.save_dir)
        os.makedirs(args.save_dir)
 
    X_train, X_valid, Y_train, Y_valid = load_data(args)
    conv_layer_sizes = [(24, 5, 2), (36, 5, 2), (48, 5, 2), (64, 3, 1), (64, 3, 1)]
    dense_layer_sizes = [100, 50, 10, 1]
    model = CNN(conv_layer_sizes, dense_layer_sizes)
    print("=" * 30)
    model.summary()
    print("=" * 30)

    start = time.time()
    model.fit(X_train, X_valid, Y_train, Y_valid, args.data_dir, save_dir=args.save_dir, 
        learning_rate=args.learning_rate, epochs=args.epochs, 
        samples_per_epoch=args.samples_per_epoch, batch_size=args.batch_size, 
        p=args.keep_prob, num_validation_batches=args.num_validation_batches) 
    end = time.time()
    total = math.ceil(end - start)
    print("")
    print("FINISHED.")
    print("Training took {} minutes {} seconds.".format(total // 60, total % 60))
 

if __name__ == '__main__':
    main()