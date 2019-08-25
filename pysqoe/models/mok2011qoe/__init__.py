import os
import scipy.optimize
import numpy as np
from pysqoe.models import QoeModel


def fun(coeff, data):
    a, b, c, d = coeff
    x1, x2, x3 = data.T
    y = a - b * x1 - c * x2 - d * x3
    return y

# We use PLCC loss to alleviate the potential mismatch in scales between MOS and prediction
def objective(coeff, data, target, fun):
    prediction = fun(coeff, data)
    yhat = prediction - np.mean(prediction)
    y = target - np.mean(target)
    loss = (-1) * np.inner(y, yhat) / (np.std(y) * np.std(yhat) * y.size)
    return loss

class Mok2011QoE(QoeModel):
    r"""
    This is an implementation of the objective QoE model described in [R1].

    Input features:
        1. x1: duration of initial buffering
        2. x2: average duration of rebuffering events
        3. x3: frequency of global stalling in 1/second

    [R1]: R. K. P. Mok, E. W. W. Chan, and R. K. C. Chang, ``Measuring the
          quality of experience of HTTP video streaming,'' in Proc. IFIP/IEEE
          Int. Symp. Integr. Netw. Manag., Dublin, Ireland, 2011, pp. 485-492.
    """
    def __init__(self):
        model_dir = os.path.dirname(os.path.realpath(__file__))
        self.param_file = os.path.join(model_dir, 'params.txt')
        if os.path.isfile(self.param_file):
            self.coeff = np.loadtxt(self.param_file)
        else:
            self.coeff = None

    def __call__(self, streaming_video):
        assert self.coeff is not None, 'Model weights do not exists.'
        x1, x2, x3 = Mok2011QoE._extract(streaming_video)
        data = np.array([x1, x2, x3])
        q = fun(self.coeff, data)
        return q

    @staticmethod
    def _extract(streaming_video):
        ib_dur = streaming_video.data['rebuffering_duration'][0]
        if ib_dur == 0:
            x1 = 0
        elif ib_dur > 0 and ib_dur <= 1:
            x1 = 1
        elif ib_dur > 1 and ib_dur <= 5:
            x1 = 2
        elif ib_dur > 5:
            x1 = 3
        else:
            raise ValueError('Initial buffering duration cannot be negative.')

        num_rb = np.count_nonzero(streaming_video.data['rebuffering_duration'])
        video_dur = np.sum(streaming_video.data['rebuffering_duration']) + \
                    np.sum(streaming_video.data['chunk_duration'])
        rb_freq = num_rb / video_dur
        if rb_freq == 0:
            x2 = 0
        elif rb_freq > 0 and rb_freq <= 0.02:
            x2 = 1
        elif rb_freq > 0.02 and rb_freq <= 0.15:
            x2 = 2
        elif rb_freq > 0.15:
            x2 = 3
        else:
            raise ValueError('Rebuffering frequency cannot be negative.')

        rb_dur = np.mean(streaming_video.data['rebuffering_duration'])
        if rb_dur == 0:
            x3 = 0
        elif rb_dur > 0 and rb_dur <= 5:
            x3 = 1
        elif rb_dur > 5 and rb_dur <= 10:
            x3 = 2
        elif rb_dur > 10:
            x3 = 3
        else:
            raise ValueError('Mean rebuffering duration cannot be negative.')
        return x1, x2, x3

    def train(self, dataset_s, dataset_a):
        print('Training Mok2011QoE...')
        # disgard dataset_a since FTW is rebuffering-oriented QoE model
        coeff_0 = np.array([4.23, 0.0672, 0.742, 0.106])
        data = []
        target = []
        for i in range(len(dataset_s)):
            streaming_video, mos = dataset_s[i]
            x1, x2, x3 = Mok2011QoE._extract(streaming_video)
            data.append([x1, x2, x3])
            target.append(mos)

        data = np.array(data)
        target = np.array(target)
        general_result = scipy.optimize.minimize(objective, coeff_0,
                                                 method='Nelder-Mead',
                                                 args=(data, target, fun))
        coeff = general_result.x
        np.savetxt(self.param_file, coeff, fmt='%03.3f')
