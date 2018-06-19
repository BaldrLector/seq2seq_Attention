import os
import sys
import re
from lxml import etree
import HyperParams as hp
from data_utils import *


def get_mov_wav_Filelist(DirPath=r'C:\Users\yaoguangming\Desktop\data', databaseNum=4):
    if DirPath is None:
        print('No path !!!')
        return None

    WavList = [DirPath + '/%03d' % (s) + '_小蜜蜂.wav' for s in range(databaseNum)]
    MovList = [DirPath + '/%03d' % (s) + '_mf.mov' for s in range(databaseNum)]

    return WavList, MovList


def getDirList(BaseDir=None, DirNum=18):
    if DirNum is None or BaseDir is None:
        print('Incorret input')
        return None

    DirList = [BaseDir + '/%03d' % (s) for s in range(DirNum)]
    return DirList


def Parsing_xml(XmlPath=None):
    """
    解析xml/eaf 文件，获取时间戳和字
    :param XmlPath:
    :return:
    """

    Timeslots = {}
    Words = {}

    tree = etree.parse(XmlPath)
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

    return Timeslots, Words, mov_time_origin, wav_time_origin


def splitHandle(input, start, end, TIME_ORIGIN=0, output=None, isMov=False):
    start_h, start_m, start_s = formateMM(start + TIME_ORIGIN)
    end_h, end_m, end_s = formateMM(end + TIME_ORIGIN)

    formated_start = '%02d:%02d:%02f' % (start_h, start_m, start_s)
    formated_end = '%02d:%02d:%02f' % (end_h, end_m, end_s)

    if isMov:
        os.system('ffmpeg -i %s -an -ss %s -to %s %s' % (input, formated_start, formated_end, output))
    else:
        os.system('ffmpeg -i %s -ss %s -to %s %s' % (input, formated_start, formated_end, output))


def formateMM(mm):
    """
    format mm ---> HH:MM:SS
    :param mm:
    :return:
    """
    h, m, s = 0, 0, 0
    s = mm / 1000
    if s >= 60:
        m = int(s / 60)
        s %= 60
    if m >= 60:
        h = int(m / 60)
        m = int(m % 60)

    return h, m, s


def split_wav_mov(BaseDir=r'C:\Users\yaoguangming\Desktop\data', outputPath=r'C:\Users\yaoguangming\Desktop\data\dict'):
    """
    切分数据
    :param BaseDir: 放置数据集的父目录
    :param outputPath: 输出切分数据的目录
    """
    DirList = getDirList(BaseDir=BaseDir, DirNum=18)
    for dir in DirList:
        wav = dir + '/' + dir[-3:] + '_小蜜蜂.wav'
        mov = dir + '/' + dir[-3:] + '.mov'

        Timeslots, Words, mov_time_origin, wav_time_origin = Parsing_xml(dir + '/' + dir[-3:] + '.eaf')

        cnt = 0
        pre_word = None

        for word in Words:

            if pre_word == None or pre_word != word:
                pre_word = word
                cnt = 1
            else:
                cnt += 1

            start = Words[word][0]
            end = Words[word][1]

            start = int(Timeslots[start])
            end = int(Timeslots[end])

            wav_path = outputPath + '/' + word + '.wav'
            mov_path = outputPath + '/' + word + '.mov'
            mix_path = outputPath + '/' + word + '_mix.mov'

            # split wav
            splitHandle(wav, start, end, TIME_ORIGIN=wav_time_origin, output=wav_path, isMov=False)
            # split mov
            splitHandle(mov, start, end, TIME_ORIGIN=mov_time_origin, output=mov_path, isMov=True)
            # mix
            os.system('ffmpeg -i %s -i %s %s' % (mov_path, wav_path, mix_path))


def split_sigle_dir(dir, outputPath):
    if os.path.exists(outputPath) is False:
        os.mkdir(outputPath)

    wav = dir + '/' + dir[-3:] + '_小蜜蜂.wav'
    mov = dir + '/' + dir[-3:] + '.mov'

    Timeslots, Words, mov_time_origin, wav_time_origin = Parsing_xml(dir + '/' + dir[-3:] + '-origin.eaf')

    cnt = 0
    pre_word = None

    for word in Words:

        if pre_word == None or pre_word != word:
            pre_word = word
            cnt = 1
        else:
            cnt += 1

        start = Words[word][0]
        end = Words[word][1]

        start = int(Timeslots[start])
        end = int(Timeslots[end])

        wav_path = outputPath + '/' + word + '.wav'
        mov_path = outputPath + '/' + word + '.mov'
        mix_path = outputPath + '/' + word + '_mix.mov'

        # split wav
        splitHandle(wav, start, end, TIME_ORIGIN=wav_time_origin, output=wav_path, isMov=False)
        # split mov
        splitHandle(mov, start, end, TIME_ORIGIN=mov_time_origin, output=mov_path, isMov=True)
        # mix
        os.system('ffmpeg -i %s -i %s %s' % (mov_path, wav_path, mix_path))


def split_by_dir(BaseDir, BaseOutput):
    if not os.path.exists(BaseDir): os.mkdir(BaseDir)

    pass
    DirList = getDirList(BaseDir=BaseDir, DirNum=19)
    print(DirList)
    from concurrent.futures import ThreadPoolExecutor
    p = ThreadPoolExecutor(20)

    for dir in DirList:
        # split_sigle_dir(dir, BaseOutput + dir[-4:])
        p.submit(split_sigle_dir, dir, BaseOutput + dir[-4:])


def split_all_sentence_dir(basedir=hp.datadir):
    from concurrent.futures import ThreadPoolExecutor
    p = ThreadPoolExecutor(10)
    dirlist = get_dir_path(basedir)
    for dir in dirlist:
        p.submit(split_single_sentence_dir, dir)


def split_single_sentence_dir(dir):
    outputdir = dir + '/split'
    # print(outputdir)
    if os.path.exists(outputdir) is False:
        os.mkdir(outputdir)
    else:
        # already split files
        return

    # split file
    files = os.listdir(dir)
    # print(files)
    movfiles = filter(lambda x: str(x).endswith('.mov'), files)
    for i in movfiles:
        # print(i[:-4])
        index = i[:-4]

        mov = os.path.join(dir, index + '.mov')
        wav = os.path.join(dir, index + '.wav')
        eaf = os.path.join(dir, index + '.eaf')

        Words, mov_time_origin, wav_time_origin = parse_EAF(eaf)

        for word in Words:
            start = Words[word][0]
            end = Words[word][1]

            wav_path = outputdir + '/' + word + '.wav'
            mov_path = outputdir + '/' + word + '.mov'
            mix_path = outputdir + '/' + word + '_mix.mov'

            # split wav
            splitHandle(wav, start, end, TIME_ORIGIN=wav_time_origin, output=wav_path, isMov=False)
            # split mov
            splitHandle(mov, start, end, TIME_ORIGIN=mov_time_origin, output=mov_path, isMov=True)
            # mix
            os.system('ffmpeg -i %s -i %s %s' % (mov_path, wav_path, mix_path))


if __name__ == '__main__':
    pass
    # mfccs, frames = load_from_npz(hp.sentence_npz_path)
    # print(mfccs.shape)
    # print(frames.shape)
    # save_to_npz(mfccs, frames, hp.sentence_npz_path)
    # split_sentence()
    # split_single_sentence_dir('./data/士兵突击')
    split_all_sentence_dir()
