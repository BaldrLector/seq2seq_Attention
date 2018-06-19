import tensorflow as tf
import os

# data param
dictdir = r"./dictinary"
datadir = './data'

# save
checkpointpath = './checkpoint'  # checkpoint path
traindir = 'train'  # suddir for train

# test
testaudio = './data/ā.wav'
testoutput = './data/a.txt'

'''
denote this ix is start from 1 not 0
'''
leeye_pos = [2, 4, 6, 8, 10, 12, 14, 16, 17, 19, 48, 51]
reeye_pos = [1, 3, 5, 7, 9, 11, 13, 15, 17, 18, 47, 50]
mouth_pos = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
             49]

morpher = ['EyeBink_L', 'EyeBink_R', 'EyeSquint_L', 'EyeSquint_R', 'EyeDown_L', 'EyeDown_R', 'EyeIn_L', 'EyeIn_R',
           'EyeOpen_L', 'EyeOpen_R', 'EyeOut_L', 'EyeOut_R', 'EyeUp_L', 'EyeUp_R', 'BrowsD_L', 'BrowsD_R', 'BrowsU_C',
           'BrowsU_L', 'BrowsU_R', 'JawOpen', 'LipsTogether', 'JawLeft', 'JawRight', 'JawFwd', 'LipsUpperUp_L',
           'LipsUpperUp_R', 'LipsLowerDown_L', 'LipsLowerDown_R', 'LipsUpperClose', 'LipsLowerClose', 'MouthSmile_L',
           'MouthSmile_R', 'MouthDimple_L', 'MouthDimple_R', 'LipsStretch_L', 'LipsStretch_R', 'MouthFrown_L',
           'MouthFrown_R', 'MouthPress_L', 'MouthPress_R', 'LipsPucker', 'LipsFunnel', 'MouthLeft', 'MouthRight',
           'ChinLowerRaise', 'ChinUpperRaise', 'Sneer_L', 'Sneer_R', 'Puff', 'CheekSquint_L', 'CheekSquint_R']

usereverse = False
addPad = True
PadTime = 100  # ms

# signal processing
sr = 22050  # Sample rate.
n_fft = 2048  # fft points (samples)
frame_shift = 0.0125  # seconds
frame_length = 0.05  # seconds
hop_length = int(sr * frame_shift)  # samples.
win_length = int(sr * frame_length)  # samples.
n_mels = 80  # Number of Mel banks to generate
power = 1.2  # Exponent for amplifying the predicted magnitude
n_iter = 50  # Number of inversion iterations
preemphasis = .97  # or None
r = 5  # Reduction factor. Paper => 2, 3, 5
max_db = 100
ref_db = 20
fps = 30  # frames per second in video

# model
n_mfcc = 39  # mfcc dimension
vocab_size = 1000
num_hidden = 300  # RNN num_hidden

output_dims = len(mouth_pos)  # outputs dimensions : 28 for only mouth_pos
batch_size = 128

# scheme
buket_size = 16
lr = 0.01
epoch = 3000
dropout_rate = 0.5
num_highwaynet_blocks = 4

if os.path.exists(checkpointpath) is False: os.mkdir(checkpointpath)

dict_npz_path = './data/dict-' + str(n_mfcc) + '.npz'
sentence_npz_path = './data/sentence.npz'
# log
logdir = './log'  # logdir path
if os.path.exists(logdir) is False: os.mkdir(logdir)
