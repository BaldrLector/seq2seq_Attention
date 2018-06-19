import HyperParams as hp
import numpy as np
import tensorflow as tf

def eval():
    '''
    Todo:

    '''
    # Load Data

    # Parse
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess,tf.train.latest_checkpoint(hp.checkpointpath))
        print("Restored!")

        writer = tf.summary.FileWriter(hp.logdir, sess.graph)
        # Feed Foward
        x = None
        y = None
        sess.run([],feed_dict={})

        writer.close()

if __name__ == '__main__':
    eval()
    print('Eval Done!')
