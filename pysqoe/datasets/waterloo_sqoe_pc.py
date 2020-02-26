import os
import pandas as pd
from pysqoe.datasets import StreamingDatabase
from pysqoe.utils import download, extract


class WaterlooSQoEPC(StreamingDatabase):
    # feature: extracted features and mos
    # server_video: extracted features, encoded representations, dash videos, and mos
    # full: extracted features, encoded representations, dash videos, streaming videos, and mos
    __valid_versions = ['feature', 'server_video', 'full']

    def __init__(self, root_dir, version='feature', download=True):
        if download:
            self._download(root_dir=root_dir, version=version)

        self.name = 'WaterlooSQoE-PC'
        csv_file = os.path.join(root_dir, 'data.csv')
        streaming_log_dir = os.path.join(root_dir, 'streaming_logs')
        video_element_dir = os.path.join(root_dir, 'server_videos') \
                                if version != 'feature' else None
        feature_profile_dir = os.path.join(root_dir, 'feature_profiles') \
                                if version != 'feature' else None
        super().__init__(csv_file=csv_file,
                         streaming_log_dir=streaming_log_dir,
                         video_element_dir=video_element_dir,
                         feature_profile_dir=feature_profile_dir)

    def _download(self, root_dir, version='feature'):
        assert version in self.__valid_versions
        if not os.path.isdir(root_dir):
            os.makedirs(root_dir)
        elif os.path.isfile(os.path.join(root_dir, 'data.csv')):
            df = pd.read_csv(os.path.join(root_dir, 'data.csv'))
            streaming_logs = df.to_dict(orient='list')
            if all([os.path.isfile(os.path.join(root_dir, 'streaming_logs', s))
                    for s in streaming_logs['streaming_log']]):
                print('WaterlooSQoE-PC already downloaded.')
                return
        
        url = 'http://ivc.uwaterloo.ca/database/WaterlooSQoE-PC/'
        if version == 'full':
            filename = 'waterloo_sqoe_pc_full.zip'
            url += filename
        elif version == 'feature':
            filename = 'waterloo_sqoe_pc_feature.zip'
            url += filename
        else:
            filename = 'waterloo_sqoe_pc_server_video.zip'
            url += filename

        zip_filename = os.path.join(root_dir, filename)
        if not os.path.isfile(zip_filename):
            print('Downloading WaterlooSQoE-PC database...\n')
            download(url, zip_filename)
            print('WaterlooSQoE-PC database download complete!\n')
        
        # extract zip
        extract(zip_filename, root_dir)
