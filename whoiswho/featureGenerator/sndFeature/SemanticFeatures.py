import os
import random
import sys
import time
from collections import defaultdict
import numpy as np

from whoiswho.config import RNDFilePathConfig, configs, version2path
from whoiswho.character.feature_process import featureGeneration
from whoiswho.utils import load_json, save_json, load_pickle, save_pickle
from whoiswho.dataset import load_utils
# debug_mod = True if sys.gettrace() else False
debug_mod = True

class AdhocFeatures:
    """
    This class is for generating adhoc features.
    1. Configure parameters according to dataset type
    2. Get features
    """
    def __init__(self,version, raw_data_root = None, processed_data_root = None,hand_feat_root = None):
        self.v2path = version2path(version)
        self.name = self.v2path['name']
        self.task = self.v2path['task'] #RND SND
        assert self.task == 'RND' , 'This features' \
                                    'only support RND task'
        self.type = self.v2path['type'] #train valid test

        #Modifying arguments when calling from outside
        if raw_data_root:
            self.raw_data_root = '../../dataset/'+self.v2path['raw_data_root']
        if processed_data_root:
            self.processed_data_root = '../../dataset/'+self.v2path["processed_data_root"]
        if hand_feat_root:
            self.hand_feat_root = '../'+self.v2path['hand_feat_root']

        # self.data = ret
        if self.type == 'train':
            self.config = {
                'name2aid2pid_path': self.processed_data_root + 'train/offline_profile.json',
                'whole_pub_info_path': self.raw_data_root + RNDFilePathConfig.train_pubs,
                'unass_candi_path': self.processed_data_root + RNDFilePathConfig.unass_candi_offline_path,
                'unass_pubs_path': self.raw_data_root + RNDFilePathConfig.train_pubs,
            }
            self.feat_save_path = self.hand_feat_root + 'pid2aid2hand_feat.offline.pkl'

        elif self.type == 'valid':
            self.config = {
                'name2aid2pid_path': self.processed_data_root + RNDFilePathConfig.whole_name2aid2pid,
                'whole_pub_info_path': self.processed_data_root + RNDFilePathConfig.whole_pubsinfo,
                'unass_candi_path': self.processed_data_root + RNDFilePathConfig.unass_candi_v1_path,
                'unass_pubs_path': self.raw_data_root + RNDFilePathConfig.unass_pubs_info_v1_path,
            }
            self.feat_save_path = self.hand_feat_root + 'pid2aid2hand_feat.onlinev1.pkl'
        else:
            self.config = {
                'name2aid2pid_path'  : self.processed_data_root + RNDFilePathConfig.whole_name2aid2pid,
                'whole_pub_info_path': self.processed_data_root + RNDFilePathConfig.whole_pubsinfo,
                'unass_candi_path'   : self.processed_data_root + RNDFilePathConfig.unass_candi_v2_path,
                'unass_pubs_path'    : self.raw_data_root + RNDFilePathConfig.unass_pubs_info_v2_path,
            }
            self.feat_save_path = self.hand_feat_root + 'pid2aid2hand_feat.onlinev2.pkl'

        self.genAdhocFeat = ProcessFeature(**self.config)

