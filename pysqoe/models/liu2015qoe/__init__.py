import os
import scipy.optimize
import numpy as np
from pysqoe.models import QoeModel


def fun(coeff, data):
    a, b, c, d, B1, B2, C1, C2 = coeff
    I_ID, D_ST, N_ST, MVM, P1, P2 = data.T
    I_ST = a * D_ST + b * N_ST - c * np.sqrt(D_ST * N_ST) + d * MVM
    I_LV = B1 * P1 + B2 * P2
    y = 100 - I_ID - I_ST - I_LV + C1 * I_ID * np.sqrt(I_ST + I_LV) + C2 * np.sqrt(I_ST * I_LV)
    return y

# We use PLCC loss to alleviate the potential mismatch in scales between MOS and prediction
def objective(coeff, data, target, fun):
    prediction = fun(coeff, data)
    yhat = prediction - np.mean(prediction)
    y = target - np.mean(target)
    loss = (-1) * np.inner(y, yhat) / (np.std(y) * np.std(yhat) * y.size)
    return loss

class Liu2015QoE(QoeModel):
    r"""
    This is an implementation of the objective QoE model described in [R1].
    Note that we do not perform the non-linear mapping from R to DASH-MOS.

    Input features:
        1. x1: I_ID
        2. x2: D_ST
        3. x3: N_ST
        4. x4: min(AMVM, MV_threshold)
        5. x5: P1
        6. x6: P2
    Model parameters: a, b
    QoE = x1 - a * x2 - b * x3 - c * x4 

    [R1]:  Y. Liu, S. Dey, F. Ulupinar, M. Luby, and Y. Mao, ``Deriving and 
           validating user experience model for DASH video streaming,'' IEEE
           Trans. Broadcast., vol. 61, no. 4, pp. 651-665, Dec. 2015.
    """
    def __init__(self):
        self.alpha = 3.2
        self.mv_t = 0.012
        self.k = 0.02
        self.mu = 0.05
        model_dir = os.path.dirname(os.path.realpath(__file__))
        self.param_file = os.path.join(model_dir, 'params.txt')
        if os.path.isfile(self.param_file):
            self.coeff = np.loadtxt(self.param_file)
        else:
            self.coeff = None

    def __call__(self, streaming_video):
        assert self.coeff is not None, 'Model weights do not exist.'
        x1, x2, x3, x4, x5, x6 = self._extract(streaming_video)
        data = np.array([x1, x2, x3, x4, x5, x6])
        q = fun(self.coeff, data)
        return q

    def _extract(self, streaming_video):
        # validate input data
        assert 'vqm' in streaming_video.data, 'VQM is not available in the dataset.'
        assert 'rebuffering_duration' in streaming_video.data, 'Rebuffering duration is not available in the dataset.'
        assert 'mvm' in streaming_video.data, 'MVM is not available in the dataset.'
        assert 'chunk_duration' in streaming_video.data, 'Chunk duration is not available in the dataset.'
        # extract features
        rb = np.array(streaming_video.data['rebuffering_duration'])
        vqm = np.array(streaming_video.data['vqm'])
        chunk_dur = np.array(streaming_video.data['chunk_duration'])
        K = rb.size
        x1 = np.minimum(self.alpha * rb[0], 100)
        x2 = np.sum(rb[0:(K - 1)])
        x3 = np.count_nonzero(rb[0:(K - 1)])
        x4 = np.minimum(np.mean(streaming_video.data['mvm']), self.mv_t)
        D = np.zeros(K)
        x6 = 0.0
        for i in range(1, K):
            if abs(vqm[i] - vqm[i - 1] < self.mu):
                D[i] = D[i - 1] + 1
            else:
                D[i] = 0

            if vqm[i] - vqm[i - 1] > 0:
                x6 = x6 + (vqm[i] - vqm[i - 1]) * (vqm[i] - vqm[i - 1])
        x6 /= K
        
        x5 = 0
        for i in range(K):
            x5 = x5 + vqm[i] * np.exp(self.k * chunk_dur[i] * D[i])
        x5 /= K
        return x1, x2, x3, x4, x5, x6

    def train(self, dataset_s, dataset_a):
        print('Training Liu2015QoE...')
        # note that we oversample dataset_s to roughly balance the sizes of the 
        # dataset_s and dataset_a
        dataset = dataset_s + dataset_s + dataset_s + dataset_a
        coeff_0 = np.array([3.35, 3.98, 2.5, 1800, 73.6, 1608, 0.15, 0.82])
        data = []
        target = []
        for i in range(len(dataset)):
            streaming_video, mos = dataset[i]
            x1, x2, x3, x4, x5, x6 = self._extract(streaming_video)
            data.append([x1, x2, x3, x4, x5, x6])
            target.append(mos)

        data = np.array(data)
        target = np.array(target)
        general_result = scipy.optimize.minimize(objective, coeff_0,
                                                 method='Nelder-Mead',
                                                 args=(data, target, fun))
        coeff = general_result.x
        np.savetxt(self.param_file, coeff, fmt='%03.3f')
