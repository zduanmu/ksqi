import os
import scipy.optimize
import numpy as np
from pysqoe.models import QoeModel


class Xue2014QoE(QoeModel):
    r"""
    This is an implementation of the objective QoE model described in [R1].
    Note that:
        1. we do not perform instantenuous quality smoothing at the end.

    [R1]:   J. Xue, D. Q. Zhang, H. Yu, and C. W. Chen, ``Assessing quality of
            experience for adaptive HTTP video streaming,'' in Proc. IEEE Int. Conf.
            Multimedia Expo, Chengdu, China, 2014, pp. 1–6.
    """
    def __init__(self):
        self.a = -1 / 51
        self.b = 1
        self.c = 0.05
        self.gamma = 0.71
        self.w_init = 0.5
        self.qp_init = 27

    def __call__(self, streaming_video):
        vb = np.array(streaming_video.data['video_bitrate'])
        qp = np.array(streaming_video.data['qp'])
        chunk_dur = np.array(streaming_video.data['chunk_duration'])
        rb_dur = np.array(streaming_video.data['rebuffering_duration'])
        q = self.a * qp + self.b
        # offline process, so maximum bitcount estimation is not necessary
        bitcount = np.multiply(vb, chunk_dur) * 1000
        bitcount_max = np.max(bitcount)
        # stalling experiment
        s_0 = -self.w_init * (self.a * self.qp_init + self.b)
        w = (np.log(bitcount[:-1]) + self.c) / (np.log(bitcount_max) + self.c)
        x = -w * q[:-1]
        y = np.sum(q * chunk_dur)
        y += s_0 * rb_dur[0]
        y += np.sum(x * rb_dur[1:])
        y /= (np.sum(chunk_dur) + np.sum(rb_dur))
        return y
