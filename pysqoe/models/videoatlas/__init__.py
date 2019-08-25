import os
import pickle
import numpy as np
from joblib import dump, load
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from pysqoe.models import QoeModel


class VideoATLAS(QoeModel):
    def __init__(self):
        r"""
        This is an implementation of the objective QoE model described in [R1].
            1. We use SVR as the regressor.
            2. We use VMAF as the video presentation quality measure.
            3. We apply average pooling to VMAF.

        [R1]:  C. G. Bampis and A. C. Bovik, ``Learning to predict streaming video
               QoE: Distortions, rebuffering and memory,'' CoRR, vol. abs/1703.00633,
               2017. [Online]. Available: http://arxiv.org/abs/1703.00633
        """
        model_dir = os.path.dirname(os.path.realpath(__file__))
        self.model_pkl = os.path.join(model_dir, 'model.pkl')
        self.regressor = None
        self.scalar = None
        if os.path.isfile(self.model_pkl):
            self.regressor = load(self.model_pkl)[0]
            self.scaler = load(self.model_pkl)[1]

    def __call__(self, streaming_video):
        assert self.regressor is not None, 'Model weights do not exist.'
        assert self.scaler is not None, 'Model weights do not exist.'
        q, r1, r2, m, i = VideoATLAS._extract(streaming_video)
        data = np.array([[q, r1, r2, m, i]])
        x = self.scaler.transform(data)
        y = self.regressor.predict(x)
        y = np.asscalar(y)
        return y

    @staticmethod
    def _extract(streaming_video):
        q = np.mean(streaming_video.data['vmaf'])
        chunk_dur = np.array(streaming_video.data['chunk_duration'])
        media_dur = np.sum(chunk_dur)
        rb_dur = np.array(streaming_video.data['rebuffering_duration'])
        rb_dur_sum = np.sum(rb_dur)
        r1 = rb_dur_sum / (media_dur + rb_dur_sum)
        r2 = np.count_nonzero(streaming_video.data['rebuffering_duration'])
        is_best = np.array(streaming_video.data['is_best'])
        m = 0
        for idx in range(is_best.size - 1, -1, -1):
            if is_best[idx]:
                m += chunk_dur[idx]
            if rb_dur[idx] > 0 or is_best[idx] == 0:
                break
        m /= (media_dur + rb_dur_sum)
        i = (np.sum(chunk_dur[is_best == 0]) + rb_dur_sum) / (media_dur + rb_dur_sum)
        return q, r1, r2, m, i

    def train(self, dataset_s, dataset_a):
        print('Training VideoATLAS...')
        # note that we oversample dataset_s to roughly balance the sizes of the 
        # dataset_s and dataset_a
        dataset = dataset_s + dataset_s + dataset_s + dataset_a
        data = []
        target = []
        for i in range(len(dataset)):
            streaming_video, mos = dataset[i]
            q, r1, r2, m, i = VideoATLAS._extract(streaming_video)
            data.append([q, r1, r2, m, i])
            target.append(mos)

        data = np.array(data)
        target = np.array(target)
        scaler = preprocessing.StandardScaler().fit(data)
        x = scaler.transform(data)

        regressor = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=10,
                                     param_grid={'C': [1e-1, 1e0, 1e1, 1e2, 1e3],
                                     'gamma': np.logspace(-2, 2, 15)})
        regressor.fit(x, np.ravel(target))
        pickle.dump([regressor.best_estimator_, scaler], open(self.model_pkl, 'wb'))
