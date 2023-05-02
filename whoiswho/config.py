import time
import os
from typing import Tuple, List, Union, Dict, Callable, Any, Optional


log_time = time.strftime("%m%d_%H%M%S")

def version2path(version: dict) -> dict:
    """
    Map the dataset information to the corresponding path
    """
    name, task, type=list(version.values())
    data_root = f"data/{name}/{task}/" #using in ./dataset
    feat_root = f"feat/{name}/{task}/" #using in ./featureGenerator


    #data
    raw_data_root = data_root
    processed_data_root = os.path.join(data_root, 'processed_data/')
    whoiswhograph_extend_processed_data = os.path.join(data_root,'whoiswhograph_extend_processed_data/')

    #feat
    hand_feat_root = os.path.join(feat_root, 'hand/')
    bert_feat_root = os.path.join(feat_root, 'bert/')
    graph_feat_root = os.path.join(feat_root, 'graph/')

    v2path={
        'name': name,
        'task': task,
        'type': type,
        'raw_data_root': raw_data_root,
        'processed_data_root': processed_data_root,
        'whoiswhograph_extend_processed_data': whoiswhograph_extend_processed_data,
        'hand_feat_root': hand_feat_root,
        'bert_feat_root': bert_feat_root,
        'graph_feat_root': graph_feat_root}
    return v2path



paper_idf_dir = 'saved/paper_idf/'
pretrained_oagbert_path = "saved/oagbert-v2-sim/"

configs = {

    "train_neg_sample"              : 19,
    "test_neg_sample"               : 19,

    # "train_ins"                     : 9622,
    # "test_ins"                      : 1480,

    "train_max_papers_each_author"  : 100,
    "train_min_papers_each_author"  : 5,

    "train_max_semi_len"            : 24,
    "train_max_whole_semi_len"      : 256,
    "train_max_per_len"             : 128,

    "train_max_semantic_len"        : 64,
    "train_max_whole_semantic_len"  : 512,
    "train_max_whole_len"           : 512,
    "raw_feature_len"               : 41,
    "feature_len"                   : 36 + 41,
    "bertsimi_graph_handfeature_len": 36 + 41 + 41 + 41,
    "str_len"                       : 36,
    "dl_len"                        : 44,
    # "train_knrm_learning_rate"    : 6e-5,
    "train_knrm_learning_rate"      : 2e-3,
    "local_accum_step"              : 32,

    "hidden_size"                   : 768,
    "n_epoch"                       : 15,
    "show_step"                     : 1,
    "padding_num"                   : 1,
}


class RNDFilePathConfig:
    # offline data
    train_name2aid2pid = "train/train_author.json"
    train_pubs = "train/train_pub.json"
    train_name2aid2pid_4train_bert_smi = "train/offline_profile.json"
    # valid
    database_name2aid2pid = "valid/whole_author_profiles.json"
    database_pubs = "valid/whole_author_profiles_pub.json"

    # train+valid所有已有的 name2aid2pid 信息
    # get_name2aid2pid方法使用
    whole_name2aid2pid = 'database/name2aid2pid.whole.json'
    whole_pubsinfo = 'database/pubs.info.json'

    unass_candi_offline_path = 'train/unass_candi.whole.json'

    unass_candi_v1_path = 'onlinev1/unass_candi.json'

    unass_candi_v2_path = 'onlinev2/unass_candi.json'

    unass_pubs_info_v1_path = 'valid/cna_valid_unass_pub.json'
    unass_pubs_info_v2_path = 'test/cna_test_unass_pub.json'

    # feat_dict
    feat_dict_path = 'feat/'

    '''
    #hand feat
    offline_hand_feat_path = hand_feat_root + 'pid2aid2hand_feat.offline.pkl'
    cna_v1_hand_feat_path = hand_feat_root + 'pid2aid2hand_feat.onlinev1.pkl'
    cna_v2_hand_feat_path = hand_feat_root + 'pid2aid2hand_feat.onlinev2.pkl'
    whoiswhograph_offline_hand_feat_path = hand_feat_root +'whoiswhograph_pid2aid2hand_feat.offline.pkl'

    #bert simi feat
    offline_bert_simi_feat_path = bert_feat_root + 'pid2aid2bert_feat.offline.pkl'
    cna_v1_bert_simi_feat_path = bert_feat_root + 'pid2aid2bert_feat.onlinev1.pkl'
    cna_v2_bert_simi_feat_path = bert_feat_root + 'pid2aid2bert_feat.onlinev2.pkl'

    tmp_offline_bert_emb_save_path = bert_feat_root + 'train/'
    tmp_cna_v1_bert_emb_feat_save_path = bert_feat_root + 'online_testv1/'
    tmp_cna_v2_bert_emb_feat_save_path = bert_feat_root + 'online_testv2/'

    tmp_offline_bert_simi_feat_save_path = bert_feat_root + 'train/'
    tmp_cna_v1_bert_simi_feat_save_path = bert_feat_root + 'online_testv1/'
    tmp_cna_v2_bert_simi_feat_save_path = bert_feat_root + 'online_testv2/'


    # graph simi feat 预备文件  "Hong_Li.pickle" is [(aid,author_path)...]
    tmp_offline_graph_save_path = graph_feat_root + 'train/'
    tmp_cna_v1_graph_feat_save_path = graph_feat_root + 'online_testv1/'
    tmp_cna_v2_graph_feat_save_path = graph_feat_root + 'online_testv2/'

    #graph simi feat  tmp片段
    tmp_offline_graph_simi_feat_save_path = graph_feat_root + 'train/'
    tmp_cna_v1_graph_simi_feat_save_path = graph_feat_root + 'online_testv1/'
    tmp_cna_v2_graph_simi_feat_save_path = graph_feat_root + 'online_testv2/'
    #graph simi feat 合并文件
    offline_graph_simi_feat_path = graph_feat_root + 'pid2aid2graph_feat_gat.offline.pkl'
    cna_v1_graph_simi_feat_path = graph_feat_root + 'pid2aid2graph_feat_gat.onlinev1.pkl'
    cna_v2_graph_simi_feat_path = graph_feat_root + 'pid2aid2graph_feat_gat.onlinev2.pkl'
    '''

