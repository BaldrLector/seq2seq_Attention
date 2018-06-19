import HyperParams as hp
import tqdm
import tensorflow as tf
from scipy.io.wavfile import write
import os
import numpy as np
from utils import *
from network import *
from model import *
from data_utils import *
from utils import *
import librosa


def inference(dir, mfcc, frame, outputname=None, mode='inference'):
    g = Graph(mode=mode)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(hp.checkpointpath + '/' + dir))

        # aduio, sr = librosa.load(hp.testaudio, sr=hp.sr)
        # mfcc = librosa.feature.mfcc(aduio, sr=sr, n_mfcc=hp.n_mfcc)

        T_y = int(hp.fps * mfcc.shape[0] * 512 / hp.sr)
        mfcc = np.expand_dims(mfcc, 0)
        y_hat = np.zeros((1, T_y, hp.output_dims))
        # check here! This way is from tactron
        for t in range(T_y):
            _y_hat = sess.run(g.y_hat, feed_dict={g.x: mfcc, g.y: y_hat})
            y_hat[:, t, :] = _y_hat[:, t, :]

        if outputname is not None:
            result_to_Morpher(y_hat[0], './output/' + outputname + '.txt')
            result_to_Morpher(frame, './output/' + outputname + '-origin.txt')


def infer_from_file(filepath, outputname, mode='inference'):
    pass
    g = Graph(mode=mode)
    saver = tf.train.Saver()

    y, sr = librosa.load(filepath, sr=hp.sr)
    mfcc = librosa.feature.mfcc(y, sr=sr, n_mfcc=hp.n_mfcc)

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(hp.checkpointpath + '/' + dir))

        # aduio, sr = librosa.load(hp.testaudio, sr=hp.sr)
        # mfcc = librosa.feature.mfcc(aduio, sr=sr, n_mfcc=hp.n_mfcc)

        T_y = int(hp.fps * mfcc.shape[0] * 1024 / hp.sr)
        mfcc = np.expand_dims(mfcc, 0)
        y_hat = np.zeros((1, T_y, hp.output_dims))
        # check here! This way is from tactron
        for t in range(T_y):
            _y_hat = sess.run(g.y_hat, feed_dict={g.x: mfcc, g.y: y_hat})
            y_hat[:, t, :] = _y_hat[:, t, :]

        if outputname is not None:
            pass
            result_to_Morpher(y_hat[0], './output/' + outputname + '.txt')
            # result_to_Morpher(frame, './output/' + outputname + '-origin.txt')
    return y_hat


if __name__ == '__main__':
    with tf.device('/cpu:0'):
        mfccs, frames = load_from_npz(npz_path=hp.sentence_npz_path)
        _, _, val_mfccs, val_frames = split_dataset(mfccs, frames)

        y_hat = inference('dict_sentecn_one_by_one', val_mfccs[0], val_frames[0], 'test')
        # inference('dict')
