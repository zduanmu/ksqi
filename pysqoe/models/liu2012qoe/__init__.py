import os
import scipy.optimize
import numpy as np
from pysqoe.models import QoeModel


def fun(coeff, data):
    a, b = coeff
    x1, x2 = data.T
    y = x1 / a - b * x2
    return y

# We use PLCC loss to alleviate the potential mismatch in scales between MOS and prediction
def objective(coeff, data, target, fun):
    prediction = fun(coeff, data)
    yhat = prediction - np.mean(prediction)
    y = target - np.mean(target)
    loss = (-1) * np.inner(y, yhat) / (np.std(y) * np.std(yhat) * y.size)
    return loss

class Liu2012QoE(QoeModel):
    r"""
    This is an implementation of the objective QoE model described in [R1].

    Input features:
        1. x1: average bitrate
        2. x2: rebuffering ratio
    Model parameters: a, b
    QoE = x1 / a - b * x2

    [R1]: X. Liu et al., ``A case for a coordinated Internet video control plane,''
          in Proc. ACM SIGCOMM, Helsinki, Finland, 2012, pp. 359-370.
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
        x1, x2 = Liu2012QoE._extract(streaming_video)
        data = np.array([x1, x2])
        q = fun(self.coeff, data)
        return q

    @staticmethod
    def _extract(streaming_video):
        x1 = np.mean(streaming_video.data['video_bitrate'])
        video_dur = np.sum(streaming_video.data['chunk_duration'])
        rb_durs = np.array(streaming_video.data['rebuffering_duration'])
        rb_durs[0] = rb_durs[0] / 3
        rb_dur = np.sum(rb_durs)
        x2 = rb_dur / (rb_dur + video_dur)
        return x1, x2

    def train(self, dataset_s, dataset_a):
        print('Training Liu2012QoE...')
        # disgard dataset_a since Liu2012QoE is an adaptation-agnostic QoE model
        coeff_0 = np.array([20, -3.7])
        data = []
        target = []
        for i in range(len(dataset_s)):
            streaming_video, mos = dataset_s[i]
            x1, x2 = Liu2012QoE._extract(streaming_video)
            data.append([x1, x2])
            target.append(mos)

        data = np.array(data)
        target = np.array(target)
        general_result = scipy.optimize.minimize(objective, coeff_0,
                                                 method='Nelder-Mead',
                                                 args=(data, target, fun))
        coeff = general_result.x
        np.savetxt(self.param_file, coeff, fmt='%03.3f')
