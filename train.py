import HyperParams as hp
import tensorflow as tf
from data_utils import *
from network import *
from utils import *


def train_one_by_one():
    pass
    g = Graph()
    print("Training Graph loaded")

    saver = tf.train.Saver()

    saver_hook = tf.train.CheckpointSaverHook(hp.checkpointpath + '/train', save_secs=1, saver=saver)
    # log_hook = tf.train.LoggingTensorHook([])
    summery_hook = tf.train.SummarySaverHook(save_secs=1, summary_op=g.merged,
                                             summary_writer=tf.summary.FileWriter(hp.checkpointpath + '/train',
                                                                                  flush_secs=1))
    # step_counter_hook = tf.train.StepCounterHook()
    # feed_hook = tf.train.FeedFnHook()
    # scaffold = tf.train.Scaffold()

    with tf.train.MonitoredSession(
            session_creator=tf.train.ChiefSessionCreator(checkpoint_dir=hp.checkpointpath + '/train'),
            hooks=[saver_hook, summery_hook], ) as sess:

        for e in range(hp.epoch):
            print(e)
            mfccs, frames = load_from_npz(hp.dict_npz_path)
            for x, y in zip(mfccs, frames):
                x = np.expand_dims(x, 0)
                y = np.expand_dims(y, 0)

                _, now_loss = sess.run([g.train_op, g.loss],
                                       feed_dict={g.x: x, g.y: y})
                # print('step\t{}\t loss:{}'.format(sess.run(g.global_step), now_loss))


def train_by_pad(dir):
    g = Graph()
    print("Training Graph loaded")

    saver = tf.train.Saver()

    saver_hook = tf.train.CheckpointSaverHook(hp.checkpointpath + '/' + dir, save_secs=60, saver=saver)
    # log_hook = tf.train.LoggingTensorHook([])
    summery_hook = tf.train.SummarySaverHook(save_secs=1, summary_op=g.merged,
                                             summary_writer=tf.summary.FileWriter(hp.checkpointpath + '/' + dir,
                                                                                  flush_secs=1))
    # step_counter_hook = tf.train.StepCounterHook()
    # feed_hook = tf.train.FeedFnHook()
    # scaffold = tf.train.Scaffold()

    with tf.train.MonitoredSession(
            session_creator=tf.train.ChiefSessionCreator(checkpoint_dir=hp.checkpointpath + '/' + dir),
            hooks=[saver_hook, summery_hook], ) as sess:
        session = sess

        # while type(session).__name__ != 'Session':
        #     session = session._sess
        for e in tqdm.tqdm(range(hp.epoch)):
            # data_loader = get_gropued_mfccs_frames()
            # data_loader = get_gropued_sentence_mfccs_frames()
            # mfccs, frames = data_loader.__next__()
            mfccs, frames = load_from_npz(npz_path=hp.sentence_npz_path)
            for mfcc, frame in zip(mfccs, frames):
                frame = np.expand_dims(frame, 0)
                mfcc = np.expand_dims(mfcc, 0)
                frame += np.random.randn(*frame.shape) * 0.001
                # mfcc += np.random.randn(*mfcc.shape) * 0.0001

                _, now_loss, step = sess.run([g.train_op, g.loss, g.global_step],
                                             feed_dict={g.x: mfcc, g.y: frame})
                print('step\t{}\t loss:{}'.format(step, now_loss))


if __name__ == '__main__':
    pass
    # train_one_by_one()
    # train_by_pad('add_font_tag')
    # train_by_pad('dict_sentecn_one_by_one')
