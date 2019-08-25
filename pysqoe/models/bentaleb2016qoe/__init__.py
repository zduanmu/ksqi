import os
import scipy.optimize
import numpy as np
from pysqoe.models import QoeModel


def fun(coeff, data):
    a, b, c = coeff
    x1, x2, x3, x4 = data.T
    y = x1 - a * x2 - b * x3 - c * x4
    return y

def fun_s(coeff, data):
    b, c = coeff
    x1, x3, x4 = data.T
    y = x1 - b * x3 - c * x4
    return y

def fun_a(coeff, data):
    a = coeff
    x1, x2 = data.T
    y = x1 - a * x2
    return y

# We use PLCC loss to alleviate the potential mismatch in scales between MOS and prediction
def objective(coeff, data, target, fun):
    prediction = fun(coeff, data)
    yhat = prediction - np.mean(prediction)
    y = target - np.mean(target)
    loss = (-1) * np.inner(y, yhat) / (np.std(y) * np.std(yhat) * y.size)
    return loss

class Bentaleb2016QoE(QoeModel):
    r"""
    This is an implementation of the objective QoE model described in [R1].
    We applied two modifications to the model described in [R1].
        1. we replaced ssimplus by vmaf, because ssimplus is not open source
        2. we removed the scaling factor of video presentation quality because
           vmaf already lies in [0, 100]

    Input features:
        1. x1: average vmaf
        2. x2: total variation of vmaf
        3. x3: average rebuffering duration (except for initial buffering)
        4. x4: initial buffering duration / K, where K is the number of segment
    Model parameters: a, b, c, d
    QoE = x1 - a * x2 - b * x3 - c * x4 

    [R1]:  A. Bentaleb, A. C. Begen, and R. Zimmermann. ``SDNDASH: Improving QoE
           of HTTP adaptive streaming using software defined networking.'' Proceedings
           of ACM Multimedia, 2016.
    """
    def __init__(self):
        model_dir = os.path.dirname(os.path.realpath(__file__))
        self.param_s_file = os.path.join(model_dir, 'param_s.txt')
        self.param_a_file = os.path.join(model_dir, 'param_a.txt')

        if os.path.isfile(self.param_s_file):
            self.param_s = np.loadtxt(self.param_s_file, ndmin=1)
        else:
            self.param_s = None

        if os.path.isfile(self.param_a_file):
            self.param_a = np.loadtxt(self.param_a_file, ndmin=1)
        else:
            self.param_a = None

    def __call__(self, streaming_video):
        assert self.param_s is not None, 'Model weights do not exist.'
        assert self.param_a is not None, 'Model weights do not exist.'
        coeff = np.concatenate((self.param_a, self.param_s))
        x1, x2, x3, x4 = Bentaleb2016QoE._extract(streaming_video)
        data = np.array([x1, x2, x3, x4])
        q = fun(coeff, data)
        return q

    @staticmethod
    def _extract(streaming_video):
        vmaf = np.array(streaming_video.data['vmaf'])
        rb_dur = np.array(streaming_video.data['rebuffering_duration'])
        K = vmaf.size
        x1 = np.mean(vmaf)
        x2 = np.sum(np.abs(vmaf[1:] - vmaf[:-1])) / K
        x3 = np.mean(rb_dur[1:]) / K
        x4 = rb_dur[0] / K
        return x1, x2, x3, x4

    def train(self, dataset_s, dataset_a):
        r"""
        We have also tried to train the parameter of the model jointly on dataset_s + dataset_a,
        but it does not seem to work well.
        """
        print('Training Bentaleb2016QoE...')
        self._train_s(dataset=dataset_s)
        self._train_a(dataset=dataset_a)

    def _train_s(self, dataset):
        coeff_0 = np.array([0.25, 0.25])
        data = []
        target = []
        for i in range(len(dataset)):
            streaming_video, mos = dataset[i]
            x1, _, x3, x4 = Bentaleb2016QoE._extract(streaming_video)
            data.append([x1, x3, x4])
            target.append(mos)

        data = np.array(data)
        target = np.array(target)
        general_result = scipy.optimize.minimize(objective, coeff_0,
                                                 method='Nelder-Mead',
                                                 args=(data, target, fun_s))
        coeff = general_result.x
        np.savetxt(self.param_s_file, coeff, fmt='%03.3f')

    def _train_a(self, dataset):
        coeff_0 = np.array([0.25])
        data = []
        target = []
        for i in range(len(dataset)):
            streaming_video, mos = dataset[i]
            x1, x2, _, _ = Bentaleb2016QoE._extract(streaming_video)
            data.append([x1, x2])
            target.append(mos)

        data = np.array(data)
        target = np.array(target)
        general_result = scipy.optimize.minimize(objective, coeff_0,
                                                 method='Nelder-Mead',
                                                 args=(data, target, fun_a))
        coeff = general_result.x
        np.savetxt(self.param_a_file, coeff, fmt='%03.3f')
