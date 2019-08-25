import os
import numpy as np
from pysqoe.models import QoeModel


class SQI(QoeModel):
    r"""
    This is an implementation of the objective QoE model described in [R1].
    There are two modifications to the original implementation, which do not
    affect the performance of the model much.
        1. We replaced ssimplus by vmaf because it is open-source.
        2. We saved the cumulated stalling experience S in eq. (4) as a lookup table.

    Note that the model parameters are manually tuned to optimize its performance on
    WaterlooSQoE-I dataset. Therefore, we do not provide the training code here.

    [R1]: Z. Duanmu, K. Zeng, K. Ma, A. Rehman, and Z. Wang, ``A quality-of-experience
          index for streaming video,'' IEEE J. Sel. Topics Signal Process., vol. 11,
          no. 1, pp. 154-166, Feb. 2017.
    """
    def __init__(self, num_t=10, num_p=10):
        model_dir = os.path.dirname(os.path.realpath(__file__))
        self.s_model_txt = os.path.join(model_dir, 's_model.txt')
        self.s0_model_txt = os.path.join(model_dir, 's0_model.txt')
        self.eps = 1e-3
        self.p_init = 80
        self.p_range = 100
        self.s_model = np.loadtxt(self.s_model_txt, delimiter=',')
        self.s0_model = np.loadtxt(self.s0_model_txt, delimiter=',')
        assert self.s_model is not None, 'Model weights do not exist.'
        assert self.s0_model is not None, 'Model weights do not exist.'
        self.num_p = num_p
        self.num_t = num_t
        self.p_size = self.p_range / self.num_p

    def __call__(self, streaming_video):
        s = []
        p = np.array(streaming_video.data['vmaf'])
        rb_dur = np.array(streaming_video.data['rebuffering_duration'])
        chunk_dur = np.array(streaming_video.data['chunk_duration'])
        session_dur = np.sum(chunk_dur) + np.sum(rb_dur)
        p_pre = p[0]
        for i, p_cur in enumerate(p):
            s_i = self._compute_rebuffering_experience(p_pre, rb_dur[i], i==0)
            s.append(s_i)
            p_pre = p_cur
        s_sum = np.sum(s)
        p_sum = np.sum(p * chunk_dur)
        qoe = (p_sum + s_sum) / session_dur
        return qoe

    def get_s_idx(self, p, t):
        # -eps to quantize 100 into the last bin
        si = int(max((p - self.eps), 0) / self.p_size)
        sj = int(min(max(round(t - self.eps), 0), self.num_t)) - 1
        return si, sj

    def _compute_rebuffering_experience(self, p_pre, t, is_first):
        if is_first:
            _, sj = self.get_s_idx(self.p_init, t)
            s = self.s0_model[sj] if sj != -1 else 0
        else:
            si, sj = self.get_s_idx(p_pre, t)
            s = self.s_model[si, sj] if sj != -1 else 0
        return s
