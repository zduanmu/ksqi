import os
import glob
import json
import bisect
import numpy as np
import pandas as pd
from pysqoe.utils.mpd_parser import MpdParser


SUPPORTED_STREAM_TYPE = ['video', 'audio']

def get_dataset(dataset, root_dir, download=True, version='feature'):
    if dataset == 'WaterlooSQoE-I':
        dataset = WaterlooSQoE1(root_dir=root_dir, download=download, version=version)
    elif dataset == 'WaterlooSQoE-II':
        dataset = WaterlooSQoE2(root_dir=root_dir, download=download, version=version)
    elif dataset == 'WaterlooSQoE-III':
        dataset = WaterlooSQoE3(root_dir=root_dir, download=download, version=version)
    elif dataset == 'WaterlooSQoE-IV':
        dataset = WaterlooSQoE4(root_dir=root_dir, download=download, version=version)
    elif dataset == 'LIVE-NFLX-I':
        dataset = LIVENFLX1(root_dir=root_dir, download=download, version=version)
    elif dataset == 'LIVE-NFLX-II':
        dataset = LIVENFLX2(root_dir=root_dir, download=download, version=version)
    elif dataset == 'WaterlooSQoE-PC':
        dataset = WaterlooSQoEPC(root_dir=root_dir, download=download, version=version)
    else:
        raise NotImplementedError('Invalid dataset %s' % dataset)

    return dataset

class StreamingDatabase(object):
    """streaming video dataset"""
    def __init__(self, csv_file, streaming_log_dir, video_element_dir=None, feature_profile_dir=None):
        """
        StreamingDatabase provides two approaches to construct a list of streaming videos and the corresponding MOS.
        ===========================================================================================================
        The first approach only takes csv_file and streaming_log_dir as input, where csv_file only contains two
        columns: 1) streaming_log and 2) mean opinion score (with header). In this case, streaming_log should be a
        csv file (with header) containing an IxJ matrix, where I and J represents the number of segments and the
        number of features, respectively. In other words, users are responsible to generate the streaming_log by
        themselves.
        ===========================================================================================================
        The second approach takes csv_file, streaming_log_dir, and video_element_dir as input, where csv_file
        contains six columns (with header): 1) streaming log, 2) mean opinion score, 3) video name, 4) encoding
        profile name, 5) video feature profile csv, and 6) audio feature profile csv (set to None if there is no
        audio). In this case, streaming_log should be a csv file (with header) containing an Ix2 matrix, where I
        represents the number of segments. Streaming video will extract the features specified in video feature
        profile csv and audio feature profile csv from instances of VideoElement, according to the video/audio
        representation index specified in streaming logs.
        ===========================================================================================================
        Args:
            csv_file (string):      Path to the csv file.
                                    Each row of csv_file contains two entries, which are
                                        1. streaming_log (e.g. BufferOccupancy_H264_document_hdtv_1.csv)
                                        2. mos: mean opinion score of the streaming video
                                    Argument used when video_element_dir != None:
                                        3. video_name (e.g. document)
                                        4. profile_name (e.g. H264)
                                        5. vf_profile (e.g. example/mini_test/vf_profile.txt)
                                        6. af_profile (e.g. None)
                                    QoeController will introduce new columns into the csv_file, with the name of the
                                    objective QoE models being headers.

                                    Note that the order of the columns does not matter, since the values are accessed
                                    by headers.
                                    ===============================================================================
                                        Each streaming_log is a csv file with headers. It contains an IxJ matrix
                                        (excluding headers), where I and J represent the number of segment, the number
                                        of features, respectively. J >= 2
                                        The header of the columns should be
                                            1. video_representation_index
                                            2. rebuffering_duration (rebuffering duration when downloading the segment)
                                            (3. optional: audio_representation_index)
                                        More features will be injected into the streaming_video_data file during the
                                        execution of the program, according to the features required by the QoE models.

            streaming_log_dir (string):
                                    Directory storing all streaming_video_data.                            
            
            video_element_dir (string):  
                                    Directory storing all video elements. Find more details in video_element.py.
        """
        df = pd.read_csv(csv_file, dtype=str)
        self.streaming_logs = df.to_dict(orient='list')
        self.streaming_log_dir = streaming_log_dir
        self.video_element_dir = video_element_dir
        self.feature_profile_dir = feature_profile_dir

    def __len__(self):
        return len(self.streaming_logs['streaming_log'])

    def __getitem__(self, idx):
        csv_file = self.streaming_logs['streaming_log'][idx]
        log_name = os.path.join(self.streaming_log_dir, csv_file)
        mos = self.streaming_logs['mos'][idx] if 'mos' in self.streaming_logs else None
        device = self.streaming_logs['device'][idx]
        if self.feature_profile_dir is None:
            streaming_video = StreamingVideo(streaming_log=log_name, device=device)
        else:
            assert self.video_element_dir is not None, 'Please specify the video element path.'
            vf_profile = os.path.join(self.feature_profile_dir, 'video.csv')
            assert os.path.isfile(vf_profile), 'Video feature profile is not found.'
            af_profile = os.path.join(self.feature_profile_dir, 'audio.csv')
            af_profile = af_profile if os.path.isfile(af_profile) else None

            root_dir = os.path.join(self.video_element_dir, self.streaming_logs['content'][idx])
            profile_name = self.streaming_logs['encoding_profile'][idx]

            video_element = VideoElement(root_dir=root_dir, profile_name=profile_name)
            streaming_video = StreamingVideo(streaming_log=log_name,
                                             video_element=video_element,
                                             vf_profile=vf_profile,
                                             af_profile=af_profile,
                                             device=device)

        return streaming_video, mos

    def __add__(self, other):
        return ConcatDataset([self, other])


class ConcatDataset(StreamingDatabase):
    """
    Dataset to concatenate multiple streaming datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]


class StreamingVideo():
    """streaming video consists of a sequence of video segments transmitted during adaptive streaming"""
    def __init__(self, streaming_log, video_element=None, vf_profile=None, af_profile=None, device=None):
        """
        Args:
            streaming_log (string):      Path to the streaming log.
        """
        self.streaming_log = streaming_log
        self.video_element = video_element
        self.device = device
        df = pd.read_csv(streaming_log)
        self.data = df.to_dict(orient='list')
        if vf_profile is not None:
            assert video_element is not None, 'Video element is not specified.'
            assert device is not None, 'Viewing device is not specified.'
            vfp = pd.read_csv(vf_profile)
            vf = vfp.to_dict(orient='list')
            # set device
            for i, d in enumerate(vf['feature_arg']):
                if d == 'DEVICE':
                    vf['feature_arg'][i] = device

            for fn, fa, fe in zip(vf['feature_name'], vf['feature_arg'], vf['identifier']):
                self.get_feature(fn, fa, fe, 'video')

        if af_profile is not None:
            afp = pd.read_csv(af_profile)
            af = afp.to_dict(orient='list')

            for fn, fa, fe in zip(af['feature_name'], af['feature_arg'], af['identifier']):
                self.get_feature(fn, fa, fe, 'audio')

    def get_feature(self, feature_name, feature_arg, identifier, stream_type):
        self.data[identifier] = []
        feature = self.video_element.get_feature_value(stream_type, feature_name, feature_arg)
        for i, rep_idx in enumerate(self.data['representation_index']):
            self.data[identifier].append(feature[rep_idx][i])

    def get_video_name(self):
        _, video_name = os.path.split(self.streaming_log)
        return video_name

    def get_display_resolution(self):
        # we will need to find a more scalable way to get the display information
        if self.device == 'phone':
            width = 2436
            height = 1126
        elif self.device == 'hdtv':
            width = 1920
            height = 1080
        elif self.device == 'uhdtv':
            width = 3840
            height = 2160
        else:
            width = 1920
            height = 1080

        return width, height

    def dump2csv(self, output=None):
        output_log = self.streaming_log if output is None else output
        df = pd.DataFrame.from_dict(self.data)
        df.to_csv(path_or_buf=output_log, index=False)


class VideoElement(object):
    """
    Each (source_video, profile_name) pair and its corresponding encoded streams,
    dash videos, and features are collectively represented as VideoElement.
    The file system of each video element should contain the following items:
    video_name/
        source.mp4
        profile_name/
            encoded/
                video/
                    v01.mp4
                    v02.mp4
                    ...
                audio/
                    a01.mp4
                    a02.mp4
                    ...
            DASH/
                manifest.mpd
                v01/
                    1.m4s
                    2.m4s
                    ...
                v02/
                    1.m4s
                    2.m4s
                    ...
                ...
            features/
                v01/
                    1.json
                    2.json
                    ...
                v02/
                    1.json
                    2.json
                    ...
                ...
    """
    def __init__(self, root_dir, profile_name):
        self.element_root = root_dir
        self.source_video = os.path.join(root_dir, 'source.mp4')
        self.profile_name = profile_name
        self.profile_root = os.path.join(root_dir, profile_name)
        self.encoded_dir = os.path.join(self.profile_root, 'encoded')
        self.dash_dir = os.path.join(self.profile_root, 'DASH')
        self.mpd = os.path.join(self.dash_dir, 'manifest.mpd')
        self.feature_dir = os.path.join(self.profile_root, 'features')
        self.encoded_streams = {}
        self.encoded_streams['video'] = self.get_encoded_streams('video')
        self.encoded_streams['audio'] = self.get_encoded_streams('audio')
        self.dash_streams = {}
        self.dash_streams['video'] = self.get_dash_streams('video')
        self.dash_streams['audio'] = self.get_dash_streams('audio')
        self.feature_jsons = {}
        self.feature_jsons['video'] = self.get_feature_jsons('video')
        self.feature_jsons['audio'] = self.get_feature_jsons('audio')

    def get_encoded_streams(self, stream_type):
        stream_path = os.path.join(self.encoded_dir, stream_type)
        if not os.path.exists(stream_path):
            return []
        
        pattern = stream_path + '/*.mp4'
        encoded_streams = glob.glob(pattern)
        return encoded_streams

    def get_dash_streams(self, stream_type):
        assert stream_type in SUPPORTED_STREAM_TYPE, 'Invalid stream type %s.' % stream_type

        if not os.path.exists(self.mpd):
            return []

        dash_streams = MpdParser.get_chunk_list(self.mpd, stream_type)
        return dash_streams
        
    def get_feature_jsons(self, stream_type):
        assert stream_type in SUPPORTED_STREAM_TYPE, 'Invalid stream type %s.' % stream_type

        jsons = []
        for representation in self.dash_streams[stream_type]:
            json_rep = []
            for segment_stream in representation:
                rep_name, stream_name = os.path.split(segment_stream)
                base_name, _ = os.path.splitext(stream_name)
                json_name = base_name + '.json'
                json_seg = os.path.join(self.feature_dir, rep_name, json_name)
                json_rep.append(json_seg)
            jsons.append(json_rep)

        return jsons
        
    def get_feature_value(self, stream_type, feature_name, feature_arg):
        assert stream_type in SUPPORTED_STREAM_TYPE, 'Invalid stream type %s.' % stream_type

        feature = []
        for representation in self.feature_jsons[stream_type]:
            feature_rep = []
            for json_seg in representation:
                if not os.path.exists(json_seg):
                    return []
                with open(json_seg, 'r') as f:
                    data = json.load(f)

                if feature_name not in data:
                    raise ValueError("Feature name %s not found in %s." % (feature_name, json_seg))
                if feature_arg not in data[feature_name]:
                    raise ValueError("Feature argument %s not found in %s." % (feature_arg, json_seg))

                feature_seg = data[feature_name][feature_arg]
                feature_rep.append(feature_seg)
            feature.append(feature_rep)

        return feature


# import existing datasets
from pysqoe.datasets.waterloo_sqoe1 import WaterlooSQoE1
from pysqoe.datasets.waterloo_sqoe2 import WaterlooSQoE2
from pysqoe.datasets.waterloo_sqoe3 import WaterlooSQoE3
from pysqoe.datasets.waterloo_sqoe4 import WaterlooSQoE4
from pysqoe.datasets.live_nflx1 import LIVENFLX1
from pysqoe.datasets.live_nflx2 import LIVENFLX2
from pysqoe.datasets.waterloo_sqoe_pc import WaterlooSQoEPC
