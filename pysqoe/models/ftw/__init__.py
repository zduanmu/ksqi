import os
import scipy.optimize
import numpy as np
from pysqoe.models import QoeModel


def fun(coeff, data):
    a, b, c, d = coeff
    x1, x2 = data.T
    y = a * np.exp(-(b * x1 + c) * x2) + d
    return y

# We use PLCC loss to alleviate the potential mismatch in scales between MOS and prediction
def objective(coeff, data, target, fun):
    prediction = fun(coeff, data)
    yhat = prediction - np.mean(prediction)
    y = target - np.mean(target)
    loss = (-1) * np.inner(y, yhat) / (np.std(y) * np.std(yhat) * y.size)
    return loss

class FTW(QoeModel):
    r"""
    This is an implementation of the objective QoE model described in [R1].

    Input features:
        1. x1: average duration of rebuffering events
        2. x2: number of rebuffering events
    Model parameters: a, b, c, d
    QoE = a * exp(-(b * x1 + c) * x2) + d

    [R1]: T. Hoßfeld, R. Schatz, E. Biersack, and L. Plissonneau, ``Internet video delivery in
          YouTube: From traffic measurements to quality of experience,'' in Data Traffic Monitoring
          and Analysis. Heidelberg, Germany: Springer, Jan. 2013, pp. 264-301.
    """
    def __init__(self):
        model_dir = os.path.dirname(os.path.realpath(__file__))
        self.param_file = os.path.join(model_dir, 'params.txt')
        if os.path.isfile(self.param_file):
            self.coeff = np.loadtxt(self.param_file)
        else:
            self.coeff = None

    def __call__(self, streaming_video):
        assert self.coeff is not None, 'Model weights do not exist.'
        x1, x2 = FTW._extract(streaming_video)
        data = np.array([x1, x2])
        q = fun(self.coeff, data)
        return q

    @staticmethod
    def _extract(streaming_video):
        rb_dur = np.array(streaming_video.data['rebuffering_duration'])
        rb_dur[0] = rb_dur[0] / 3
        x1 = np.mean(rb_dur)
        x2 = np.count_nonzero(rb_dur)
        return x1, x2

    def train(self, dataset_s, dataset_a):
        print('Training FTW...')
        # disgard dataset_a since FTW is rebuffering-oriented QoE model
        coeff_0 = np.array([3.5, 0.15, 0.19, 1.5])
        data = []
        target = []
        for i in range(len(dataset_s)):
            streaming_video, mos = dataset_s[i]
            x1, x2 = FTW._extract(streaming_video)
            data.append([x1, x2])
            target.append(mos)

        data = np.array(data)
        target = np.array(target)
        general_result = scipy.optimize.minimize(objective, coeff_0,
                                                 method='Nelder-Mead',
                                                 args=(data, target, fun))
        coeff = general_result.x
        np.savetxt(self.param_file, coeff, fmt='%03.3f')
