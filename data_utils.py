from functools import reduce

import numpy as np
import sys
import os
import re
from lxml import etree
import glob
import librosa
import tqdm
import HyperParams as hp
import tensorflow as tf
import tensorflow.contrib.data as data
import re
from utils import *

'''
    file should be place like that:
    basedir
        000
            000.mov
            000_mf.mov
            000_小蜜蜂.wav
            000MorphFrame.txt
            000.eaf
            000_origin.eaf
            
        001
            ...
        ...
'''


def read_3dmax_all_frame(file):
    '''

    :param file: file path for the 3dmax frame
    :return:an array with 51-dimension
    '''
    pass
    # frames = []
    # with open(file, 'r') as f:
    #     for l in f.readlines():
    #         content = l.strip().split(' ')
    #         frame = []
    #         for ix in hp.mouth_pos:
    #             # denote this because ix in mouth_pos is start from 1 not 0
    #             if content[ix - 1] == '0f':
    #                 frame.append(0.0)
    #             elif content[ix - 1] == '100f':
    #                 frame.append(100.0)
    #             else:
    #                 frame.append(float(content[ix - 1]))
    #         frames.append(frame)
    #
    # frames = np.array(frames)
    frames = []
    mouth_pos = np.array(hp.mouth_pos)
    with open(file, 'r') as f:
        for l in f.readlines():
            content = l.strip().split(' ')
            content = np.array(content)
            frames.append(content[mouth_pos - 1])
    frames = np.array(frames)
    return frames

def cut_3dmax_frame(start, end, fps=30.0, MorphFrame=None):
    '''

    :param start: start time mm
    :param end: end time mm
    :param fps: sample ratio of 3dmax frame file
    :return:an array from start to end time, its shape is (1,T,D) , where D should be 51 (default)
    '''

    if MorphFrame is None:
        raise EOFError

    start /= 1000.0
    end /= 1000.0
    start_frame, end_frame = int(fps * start), int(fps * end)
    return MorphFrame[start_frame:end_frame]


def get_mfcc(filepath,
             start, end, sr=hp.sr, n_mfcc=hp.n_mfcc):
    '''

    :param filepath: wav path
    :param sr: sample rate ,default = hp.sr
    :param offset:
    :param n_mfcc: demension of mfcc
    :return:
    '''
    start /= 1000.0
    end /= 1000.0

    y, _ = librosa.load(filepath, sr=sr, offset=start, duration=end - start)  # (n,)

    audio_mfcc = librosa.feature.mfcc(y, sr=sr, n_mfcc=n_mfcc)  # (n_mfcc,t)
    audio_mfcc = audio_mfcc.transpose((1, 0))

    return audio_mfcc


def get_dir_path(Basedir):
    '''

    :param Basedir: Base dir path
    :return: text file path and eaf file path
    '''
    if not os.path.exists(Basedir):
        raise FileExistsError

    dirnames = os.listdir(Basedir)
    dirlist = [os.path.join(Basedir, dir) for dir in dirnames if os.path.isdir(os.path.join(Basedir, dir))]
    return dirlist


def parse_EAF(EAFfile):
    '''

    :param EAFfile: EAFfile path
    :return: Words [word1:[start,end],...]  mov_time_origin, wav_time_origin, is sentence
            or   [index:[start,end],...]
    '''

    Timeslots = {}
    Words = {}

    tree = etree.parse(EAFfile)
    root = tree.getroot()

    mov_time_origin, wav_time_origin = 0, 0

    if 'TIME_ORIGIN' in root[0][0].attrib.keys():
        mov_time_origin = int(root[0][0].attrib['TIME_ORIGIN'])

    if 'TIME_ORIGIN' in root[0][1].attrib.keys():
        wav_time_origin = int(root[0][1].attrib['TIME_ORIGIN'])

    timeOrder = root[1]
    tier = root[2]

    for t in timeOrder:
        Timeslots[t.values()[0]] = t.values()[1]

    for t in tier:
        Words[t[0][0].text] = t[0].values()[1:]

    for w in Words.keys():
        Words[w][0] = int(Timeslots[Words[w][0]])
        Words[w][1] = int(Timeslots[Words[w][1]])

    return Words, mov_time_origin, wav_time_origin


def parse_TEXT(TEXTfile):
    '''

    :param TEXTfile:
    :return: index2content [index1:content1,...]
    '''
    pass


def deal_single_dir(dir, MrophFrametail='MorphFrame.txt', EAFtail='-origin.eaf', WAVtail='_小蜜蜂.wav', split_by_dir=True):
    '''

    :param dir:
    :param MrophFrametail: use tail for MrophFrame file
    :param EAFtail:
    :param WAVtail:
    :param split_by_dir: wherea is split by dir
    :return:
    '''

    print(dir)

    mfccs = []
    frames = []

    if split_by_dir:
        MorphFramePath = dir + '/' + os.path.basename(dir) + MrophFrametail
        EAFPath = dir + '/' + os.path.basename(dir) + EAFtail
        WAVPath = dir + '/' + os.path.basename(dir) + WAVtail
    else:
        MorphFramePath = dir + MrophFrametail
        EAFPath = dir + EAFtail
        WAVPath = dir + WAVtail

    # print(MorphFramePath)
    # print(EAFPath)
    # print(WAVPath)

    Words, mov_time_origin, wav_time_origin = parse_EAF(EAFPath)
    MorphFrame = read_3dmax_all_frame(MorphFramePath)

    for w in Words.keys():
        start, end = Words[w][0], Words[w][1]

        frame = cut_3dmax_frame(mov_time_origin + start, mov_time_origin + end, fps=hp.fps, MorphFrame=MorphFrame)
        mfcc = get_mfcc(WAVPath, start + wav_time_origin, end + wav_time_origin, sr=hp.sr)

        mfccs.append(mfcc)
        frames.append(frame)

    # mfccs = np.array(mfccs)
    # frames = np.array(frames)

    return mfccs, frames


def load_all_data(basedir):
    dirlist = get_dir_path(basedir)

    mfccs, frames = deal_single_dir(dirlist[0])
    mfccs, frames = np.array(mfccs), np.array(frames)

    for i in range(1, len(dirlist)):
        mfccs_now, frames_now = deal_single_dir(dirlist[i])
        mfccs = np.concatenate((mfccs, mfccs_now), axis=0)
        frames = np.concatenate((frames, frames_now), axis=0)

    return mfccs, frames


def load_from_npz(npz_path):
    mfccs, frames = np.load(npz_path)['mfccs'], np.load(npz_path)['frames']
    return mfccs, frames


def save_to_npz(mfccs, frames, npz_path, cover=False):
    if cover or os.path.exists(npz_path) is False:
        np.savez(npz_path, mfccs=mfccs, frames=frames)


def make_dict_npz(basedir='dictinary'):
    mfccs, frames = load_all_data(basedir)
    np.savez(hp.dict_npz_path, mfccs=mfccs, frames=frames)


def add_reverse(mfccs, frames):
    assert len(mfccs) == len(frames)
    mfccs = np.concatenate((mfccs, mfccs[:][::-1]), axis=0)
    frames = np.concatenate((frames, frames[:][::-1]), axis=0)
    return mfccs, frames


def get_gropued_mfccs_frames(npz_path=hp.dict_npz_path):
    mfccs, frames = load_from_npz(npz_path=npz_path)
    ix = np.argsort([i.shape[0] for i in mfccs])
    mfccs, frames = mfccs[ix], frames[ix]
    num_group = np.ceil(len(mfccs) // hp.buket_size)
    for i in range(int(num_group)):
        sub_mfccs = mfccs[
                    i * hp.buket_size:(i + 1) * hp.buket_size if (i + 1) * hp.buket_size < len(mfccs) else len(mfccs)]
        sub_frames = frames[
                     i * hp.buket_size:(i + 1) * hp.buket_size if (i + 1) * hp.buket_size < len(frames) else len(
                         frames)]

        # print(sub_frames.shape)
        # print(sub_mfccs.shape)

        mfccs_maxtime = max([i.shape[0] for i in sub_mfccs])
        frames_maxtime = max([i.shape[0] for i in sub_frames])

        # print(mfccs_maxtime)
        # print(frames_maxtime)

        for i, v in enumerate(sub_mfccs):
            sub_mfccs[i] = np.concatenate((sub_mfccs[i], np.zeros((mfccs_maxtime - sub_mfccs[i].shape[0], hp.n_mfcc))),
                                          0)

        for i, v in enumerate(sub_frames):
            sub_frames[i] = np.concatenate(
                (sub_frames[i], np.zeros(((frames_maxtime - sub_frames[i].shape[0]), hp.output_dims))),
                0)
        sub_mfccs = np.stack(sub_mfccs)
        sub_frames = np.stack(sub_frames)
        yield sub_mfccs, sub_frames


def get_stack_mfccs_frames(npz_path):
    with tf.device('/cpu:0'):
        pass
        mfccs, frames = load_from_npz(npz_path=npz_path)

        if hp.usereverse:
            mfccs, frames = add_reverse(mfccs, frames)

        mfccs_maxtime = max([i.shape[0] for i in mfccs])
        mfccs_mintime = min([i.shape[0] for i in mfccs])

        frames_maxtime = max([i.shape[0] for i in frames])
        frames_mintime = min([i.shape[0] for i in frames])

        for i, v in enumerate(mfccs):
            mfccs[i] = np.concatenate((mfccs[i], np.zeros((mfccs_maxtime - mfccs[i].shape[0], hp.n_mfcc))), 0)

        for i, v in enumerate(frames):
            frames[i] = np.concatenate((frames[i], np.zeros(((frames_maxtime - frames[i].shape[0]), hp.output_dims))),
                                       0)

        # try use align-pad
        # for i, v in enumerate(mfccs):
        #     leftpad = (mfccs_maxtime - mfccs[i].shape[0]) // 2
        #     rightpad = mfccs_maxtime - mfccs[i].shape[0] - leftpad
        #     mfccs[i] = np.concatenate((np.zeros((leftpad, hp.n_mfcc)), mfccs[i]), 0)
        #     mfccs[i] = np.concatenate((mfccs[i], np.zeros((rightpad, hp.n_mfcc))), 0)
        #
        # for i, v in enumerate(frames):
        #     leftpad_frame = (frames_maxtime - frames[i].shape[0]) // 2
        #     rightpad_frame = frames_maxtime - frames[i].shape[0] - leftpad_frame
        #     frames[i] = np.concatenate((np.zeros((leftpad_frame, hp.output_dims)), frames[i]), 0)
        #     frames[i] = np.concatenate((frames[i], np.zeros((rightpad_frame, hp.output_dims))), 0)

        mfccs = np.stack(mfccs)
        frames = np.stack(frames)
        return mfccs, frames


def stack_batch_loader():
    pass
    mfccs, frames = get_stack_mfccs_frames(hp.dict_npz_path)
    num_bactch = mfccs.shape[0] // hp.batch_size
    i = 0
    while True:
        if i < num_bactch - 1:
            yield mfccs[int(i * num_bactch):int((i + 1) * num_bactch)], frames[
                                                                        int(i * num_bactch):int((i + 1) * num_bactch)]
        else:
            yield mfccs[int(i * num_bactch):-1], frames[int(i * num_bactch):-1]
        i = (i + 1) % (num_bactch)


def get_file_paths(Basedir):
    dirnames = os.listdir(Basedir)
    tails = ['wav', 'mov', 'fbx', 'eaf']
    fileset = set([dir.split('.')[0] for dir in dirnames if dir.split('.')[1] in tails])
    filepaths = [os.path.join(Basedir, file) for file in fileset]
    return filepaths


def read_txt(txtfile):
    content = []
    with open(txtfile, 'r', encoding='utf8') as f:
        for l in f.readlines():
            content.append(l.strip())

    res = []
    for i, v in enumerate(content):
        if len(content[i]) > 0:
            res.append(content[i][3:].strip())
    return res


def read_single_sentence_dir(dir):
    filepaths = get_file_paths(dir)
    print(filepaths)

    mfccs, frames = deal_single_dir(filepaths[0], MrophFrametail='MorphFrame.txt', EAFtail='.eaf',
                                    WAVtail='.wav', split_by_dir=False)

    for i in range(1, len(filepaths)):
        mfccs_now, frames_now = deal_single_dir(filepaths[i], MrophFrametail='MorphFrame.txt', EAFtail='.eaf',
                                                WAVtail='.wav', split_by_dir=False)
        # mfccs = np.concatenate((mfccs, mfccs_now), axis=0)
        # frames = np.concatenate((frames, frames_now), axis=0)
        mfccs += mfccs_now
        frames += frames_now

    mfccs = np.array(mfccs)
    frames = np.array(frames)
    return mfccs, frames


def read_all_sentence_dir(basedir=hp.datadir, save=True):
    dirlist = get_dir_path(basedir)
    mfccs, frames = read_single_sentence_dir(dirlist[0])
    if len(dirlist) > 1:
        for i in range(1, len(dirlist)):
            mfccs_now, frames_now = read_single_sentence_dir(dirlist[i])
            mfccs = np.concatenate((mfccs, mfccs_now), axis=0)
            frames = np.concatenate((frames, frames_now), axis=0)
    if os.path.exists(hp.sentence_npz_path):
        os.remove(hp.sentence_npz_path)
        np.savez(hp.sentence_npz_path, mfccs=mfccs, frames=frames)
    return mfccs, frames


def get_gropued_sentence_mfccs_frames(npz_path=hp.sentence_npz_path, Use_train_val=True):
    mfccs, frames = load_from_npz(npz_path=npz_path)

    if Use_train_val:
        mfccs_train, frames_train, mfccs_val, frames_val = split_dataset(mfccs, frames)
        mfccs = mfccs_train
        frames = frames_train

    # Here sort for bucket
    # ix = np.argsort([i.shape[0] for i in mfccs])
    # mfccs, frames = mfccs[ix], frames[ix]

    num_group = np.ceil(len(mfccs) // hp.buket_size)

    for i in range(int(num_group)):
        sub_mfccs = mfccs[
                    i * hp.buket_size:(i + 1) * hp.buket_size if (i + 1) * hp.buket_size < len(mfccs) else len(mfccs)]
        sub_frames = frames[
                     i * hp.buket_size:(i + 1) * hp.buket_size if (i + 1) * hp.buket_size < len(frames) else len(
                         frames)]

        # print(sub_frames.shape)
        # print(sub_mfccs.shape)

        mfccs_maxtime = max([i.shape[0] for i in sub_mfccs])
        frames_maxtime = max([i.shape[0] for i in sub_frames])

        # print(mfccs_maxtime)
        # print(frames_maxtime)

        for i, v in enumerate(sub_mfccs):
            sub_mfccs[i] = np.concatenate((sub_mfccs[i], np.zeros((mfccs_maxtime - sub_mfccs[i].shape[0], hp.n_mfcc))),
                                          0)

        for i, v in enumerate(sub_frames):
            sub_frames[i] = np.concatenate(
                (sub_frames[i], np.zeros(((frames_maxtime - sub_frames[i].shape[0]), hp.output_dims))),
                0)
        sub_mfccs = np.stack(sub_mfccs)
        sub_frames = np.stack(sub_frames)
        yield sub_mfccs, sub_frames


def get_random_sentence_mfccs_frames(npz_path=hp.sentence_npz_path, Use_train_val=True, Use_backet=True):
    pass
    mfccs, frames = load_from_npz(npz_path=npz_path)

    if Use_train_val:
        mfccs_train, frames_train, mfccs_val, frames_val = split_dataset(mfccs, frames)
        mfccs = mfccs_train
        frames = frames_train

    if Use_backet:
        while 1:
            ix = np.random.choice(len(mfccs), hp.buket_size)
            sub_mfccs, sub_frames = mfccs[ix], frames[ix]
            mfccs_maxtime = max([i.shape[0] for i in sub_mfccs])
            frames_maxtime = max([i.shape[0] for i in sub_frames])

            for i, v in enumerate(sub_mfccs):
                sub_mfccs[i] = np.concatenate(
                    (sub_mfccs[i], np.zeros((mfccs_maxtime - sub_mfccs[i].shape[0], hp.n_mfcc))),
                    0)

            for i, v in enumerate(sub_frames):
                sub_frames[i] = np.concatenate(
                    (sub_frames[i], np.zeros(((frames_maxtime - sub_frames[i].shape[0]), hp.output_dims))),
                    0)
            sub_mfccs = np.stack(sub_mfccs)
            sub_frames = np.stack(sub_frames)
            yield sub_mfccs, sub_frames

    else:
        while 1:
            ix = np.random.choice(len(mfccs), 1)
            yield mfccs[ix], frames[ix]


def get_smooth_cutted_data():
    mfccs, frames = load_from_npz(hp.dict_npz_path)
    for i in range(len((frames))):
        res_mfccs, res_frames = [], []
        num_ = frames[i].shape[0] // 0.7
        for i in range(num_):
            pass
            la2lv = mfccs.shape[0] / frames.shape[0]
            mfcc_width = mfccs.shape[0] // 2
            frame_witdh = frames.shape[0] // 2


def get_batch():
    """Loads training data and put them in queues"""
    with tf.device('/cpu:0'):
        # Load data
        mfccs, frames = load_from_npz(hp.dict_npz_path)

        # for i in range(len(mfccs)):
        #     mfccs[i] = mfccs[i].reshape(-1)

        mfccs_maxlen, mfccs_minlen = max([i.shape[0] for i in mfccs]), min([i.shape[0] for i in mfccs])
        frames_maxlen, frames_minlen = max([i.shape[0] for i in frames]), min([i.shape[0] for i in frames])

        # Calc total batch count
        num_batch = len(mfccs) // hp.batch_size

        # dataset = tf.data.Dataset.


def split_dataset(mfccs, frames, split=0.2):
    mfccs, frames = load_from_npz(npz_path=hp.sentence_npz_path)
    assert len(mfccs) == len(frames)
    val_len = int(len(mfccs) * split)
    return mfccs[:-val_len], frames[:-val_len], mfccs[-val_len:], frames[-val_len:]
