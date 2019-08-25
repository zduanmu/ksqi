def get_qoe_model(model_name):
    if model_name == 'FTW':
        model = FTW()
    elif model_name == 'Mok2011QoE':
        model = Mok2011QoE()
    elif model_name == 'Liu2012QoE':
        model = Liu2012QoE()
    elif model_name == 'Xue2014QoE':
        model = Xue2014QoE()
    elif model_name == 'Liu2015QoE':
        model = Liu2015QoE()
    elif model_name == 'Yin2015QoE':
        model = Yin2015QoE()
    elif model_name == 'Spiteri2016QoE':
        model = Spiteri2016QoE()
    elif model_name == 'Bentaleb2016QoE':
        model = Bentaleb2016QoE()
    elif model_name == 'SQI':
        model = SQI()
    elif model_name == 'P1203':
        model = P1203()
    elif model_name == 'VideoATLAS':
        model = VideoATLAS()
    elif model_name == 'KSQI':
        model = KSQI()
    elif model_name == 'VsQMDASH':
        raise ValueError('VsQM-DASH is only applicable to streaming videos with 3 segments.')
    else:
        raise NotImplementedError('Invalid QoE model %s.' % model_name)

    return model

def train(models, dataset_s, dataset_a):
    model_names = models.split(':')
    models = [get_qoe_model(model_name) for model_name in model_names]
    for model in models:
        model.train(dataset_s=dataset_s, dataset_a=dataset_a)

    print('Training is completed.')

class QoeModel(object):
    def __init__(self):
        pass

    def __call__(self, streaming_video):
        pass


from pysqoe.models.ftw import FTW
from pysqoe.models.mok2011qoe import Mok2011QoE
from pysqoe.models.liu2012qoe import Liu2012QoE
from pysqoe.models.xue2014qoe import Xue2014QoE
from pysqoe.models.liu2015qoe import Liu2015QoE
from pysqoe.models.yin2015qoe import Yin2015QoE
from pysqoe.models.spiteri2016qoe import Spiteri2016QoE
from pysqoe.models.bentaleb2016qoe import Bentaleb2016QoE
from pysqoe.models.sqi import SQI
from pysqoe.models.p1203 import P1203
from pysqoe.models.videoatlas import VideoATLAS
from pysqoe.models.ksqi import KSQI
