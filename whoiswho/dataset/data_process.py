import os
import random
import copy
from collections import defaultdict
import multiprocessing
from tqdm import tqdm
from pprint import pprint
from whoiswho.utils import load_json, save_json, get_author_index, dname_l_dict
from whoiswho.character.name_match.tool.is_chinese import cleaning_name
from whoiswho.character.name_match.tool.interface import FindMain
from whoiswho.config import RNDFilePathConfig, configs, version2path
from whoiswho.dataset import load_utils
'''This module is only used to split the RND data. '''

def printInfo(dicts):
    aNum = 0
    pNum = 0
    for name, aidPid in dicts.items():
        aNum += len(aidPid)
        for aid, pids in aidPid.items():
            pNum += len(pids)

    print("#Name %d, #Author %d, #Paper %d" % (len(dicts), aNum, pNum))


def split_train2dev(data: list,processed_data_root: str, unass_ratio=0.2):
    '''将 train 划分为训练集和测试集'''

    def _get_last_n_paper(name, paper_ids, paper_info, ratio=0.2):
        cnt_unfind_author_num = 0  # 未找到作者 index 的数量
        name = cleaning_name(name)
        years = set()
        now_years = defaultdict(list)
        for pid in paper_ids:
            year = paper_info[pid].get('year', '0')
            if year == '':
                year = 0
            else:
                year = int(year)
            if year < 1500 or year > 2022:
                year = 0
            years.add(year)
            authors = paper_info[pid].get('authors', [])
            author_names = [a['name'] for a in authors]
            author_res = FindMain(name, author_names)[0]
            if len(author_res) > 0:
                aids = author_res[0][1]
            else:
                aids = get_author_index(name, author_names, False)
                if aids < 0:
                    aids = len(authors)
                    cnt_unfind_author_num += 1
            assert aids >= 0
            # if aids == len(authors):
            #     cnt_unfind_author_num += 1
            # assert aids >= 0, f"{name} 's paper {pid}"
            now_years[year].append((pid, aids,))
        years = list(years)
        years.sort(reverse=False)
        papers_list = []
        assert len(years) > 0
        for y in years:
            papers_list.extend(now_years[y])
        # 取后 ratio 的作为未分配论文
        split_gap = int((1 - ratio) * len(papers_list))
        unass_list = papers_list[split_gap:]
        prof_list = papers_list[0:split_gap]
        assert len(unass_list) > 0
        assert len(prof_list) > 0
        assert len(unass_list) + len(prof_list) == len(papers_list)
        return prof_list, unass_list, cnt_unfind_author_num

    def _split_unass(names, authors_info, papers_info, unass_info, dump_info):
        sum_unfind_author_num = 0
        unass_candi_list = []
        for name in names:
            unass_info[name] = {}
            dump_info[name] = {}
            for aid in authors_info[name]:
                papers = authors_info[name][aid]
                prof_list, unass_list, cnt_unfind_num = _get_last_n_paper(name, papers, papers_info, unass_ratio)
                sum_unfind_author_num += cnt_unfind_num

                unass_info[name][aid] = [f"{p[0]}-{p[1]}" for p in unass_list if
                                         'authors' in papers_info[p[0]] and 0 <= p[1] < len(
                                             papers_info[p[0]]['authors'])]
                dump_info[name][aid] = [f"{p[0]}-{p[1]}" for p in prof_list]
                for pid in unass_info[name][aid]:
                    unass_candi_list.append((pid, name))
        print('The number of papers that could not find the author name : ', sum_unfind_author_num)
        return unass_candi_list

    papers_info = data[1]
    authors_info = data[0]
    names = []
    for name in authors_info:
        names.append(name)
    random.shuffle(names)

    train_unass_info = {}
    train_dump_info = {}
    train_unass_candi = _split_unass(names, authors_info, papers_info, train_unass_info, train_dump_info)
    save_json(train_dump_info, processed_data_root, "train/offline_profile.json")
    save_json(train_unass_info, processed_data_root, "train/offline_unass.json")
    save_json(train_unass_candi, processed_data_root, 'train/unass_candi.whole.json')




def get_author_index_father(params):
    ''' Functions wrapped by multiprocessing  '''
    unass_pid, name, dnames = params
    author_res = FindMain(name, dnames)[0]
    if len(author_res) > 0:
        return unass_pid, author_res[0][1], 'find', name
    res = get_author_index(name, dnames, True)
    return unass_pid, res, 'doudi', name


def get_name2aid2pid(raw_data_root,processed_data_root,name2aid2pids_path):
    ''' Merge all the information from the train set and valid set '''

    whole_pros = load_json(raw_data_root, RNDFilePathConfig.database_name2aid2pid)
    whole_pubs_info = load_json(raw_data_root, RNDFilePathConfig.database_pubs)

    train_pros = load_json(raw_data_root, RNDFilePathConfig.train_name2aid2pid)
    train_pubs_info = load_json(raw_data_root, RNDFilePathConfig.train_pubs)

    whole_pubs_info.update(train_pubs_info)
    save_json(whole_pubs_info, processed_data_root, RNDFilePathConfig.whole_pubsinfo)

    this_year = 2022
    # Merge all authors under the same name.
    name_aid_pid = defaultdict(dict)
    for aid, ainfo in whole_pros.items():
        name = ainfo['name']
        pubs = ainfo['pubs']
        name_aid_pid[name][aid] = pubs
    # print(name_aid_pid)
    # Find the main author index for each paper
    key_names = list(name_aid_pid.keys())
    new_name2aid2pids = defaultdict(dict)

    for i in range(len(key_names)):
        name = key_names[i]
        aid2pid = name_aid_pid[name]
        for aid, pids in aid2pid.items():
            tmp_pubs = []
            for pid in pids:
                coauthors = [tmp['name'] for tmp in whole_pubs_info[pid]['authors']]
                coauthors = [n.replace('.', ' ').lower() for n in coauthors]
                if 'year' in whole_pubs_info[pid]:
                    year = whole_pubs_info[pid]['year']
                    year = int(year) if year != '' else this_year
                else:
                    year = this_year

                aidx = get_author_index_father((pid, name, coauthors))[1]
                if aidx < 0:
                    aidx = len(coauthors)
                new_pid = pid + '-' + str(aidx)
                tmp_pubs.append((new_pid, year))
            tmp_pubs.sort(key=lambda x: x[1], reverse=True)
            tmp_pubs = [p[0] for p in tmp_pubs]
            new_name2aid2pids[name][aid] = tmp_pubs
    printInfo(new_name2aid2pids)

    for name, aid2pid in train_pros.items():
        assert name not in key_names
        for aid, pids in aid2pid.items():
            tmp_pubs = []
            for pid in pids:
                coauthors = [tmp['name'].lower() for tmp in train_pubs_info[pid]['authors']]
                coauthors = [n.replace('.', ' ').lower() for n in coauthors]
                if 'year' in train_pubs_info[pid]:
                    year = train_pubs_info[pid]['year']
                    year = int(year) if year != '' else this_year
                else:
                    year = this_year

                aidx = get_author_index_father((pid, name, coauthors))[1]
                if aidx < 0:
                    aidx = len(coauthors)
                new_pid = pid + '-' + str(aidx)

                tmp_pubs.append((new_pid, year))
            tmp_pubs.sort(key=lambda x: x[1], reverse=True)
            tmp_pubs = [p[0] for p in tmp_pubs]
            new_name2aid2pids[name][aid] = tmp_pubs
    new_name2aid2pids = dict(new_name2aid2pids)
    printInfo(new_name2aid2pids)

    save_json(new_name2aid2pids, processed_data_root, name2aid2pids_path)


def pretreat_unass(raw_data_root,processed_data_root,unass_candi_path, unass_list_path, unass_paper_info_path):

    name_aid_pid = load_json(processed_data_root, RNDFilePathConfig.whole_name2aid2pid)

    unass_list = load_json(raw_data_root, unass_list_path)
    unass_paper_info = load_json(raw_data_root, unass_paper_info_path)
    whole_candi_names = list(name_aid_pid.keys())
    print("#Unass: %d #candiNames: %d" % (len(unass_list), len(whole_candi_names)))

    unass_candi = []
    not_match = 0

    num_thread = int(multiprocessing.cpu_count() / 1.3)
    pool = multiprocessing.Pool(num_thread)

    ins = []
    for unass_pid in unass_list:
        pid, aidx = unass_pid.split('-')
        candi_name = unass_paper_info[pid]['authors'][int(aidx)]['name']

        ins.append((unass_pid, candi_name, whole_candi_names))

    multi_res = pool.map(get_author_index_father, ins)
    pool.close()
    pool.join()
    for i in multi_res:
        pid, aidx, typ, name = i
        if aidx >= 0:
            unass_candi.append((pid, whole_candi_names[aidx]))
        else:
            not_match += 1
            print(i)   #Print the information of papers that were not found ambiguous names
    print("Matched: %d Not Match: %d" % (len(unass_candi), not_match))

    # print(unass_candi_path)
    save_json(unass_candi, processed_data_root, unass_candi_path)
    save_json(whole_candi_names, processed_data_root, 'whole_candi_names.json')


def split_list2kfold(s_list, k, start_index=0):
    # Partition the input list into k parts
    num = len(s_list)
    each_l = int(num / k)
    result = []
    remainer = num % k
    random.shuffle(s_list)
    last_index = 0
    for i in range(k):
        if (k + i - start_index) % k < remainer:
            result.append(s_list[last_index:last_index + each_l + 1])
            last_index += each_l + 1
        else:
            result.append(s_list[last_index:last_index + each_l])
            last_index += each_l
    return result, (start_index + remainer) % k


def kfold_main_func(processed_data_root,offline_whole_profile, offline_whole_unass, k=5):
    kfold_path = f"{processed_data_root}/train/kfold_dataset/"
    os.makedirs(kfold_path, exist_ok=True)
    # Get the list of author names in the training set and the number of candidates
    name_weight = []
    for name, aid2pids in offline_whole_profile.items():
        assert len(aid2pids.keys()) == len(offline_whole_unass[name].keys())
        name_weight.append((name, len(aid2pids.keys())))
    # name_weight.sort(key=lambda x: x[1])

    both_name_weight = []
    unused_name_weight = []
    for name, weight in name_weight:
        if weight < 20:
            unused_name_weight.append((name, weight))
        else:
            both_name_weight.append((name, weight))
    # Partition the set of names into k groups
    start_index = 0
    split_res = [[] for i in range(k)]
    tmp, start_index = split_list2kfold(unused_name_weight, k, start_index)
    for i in range(k):
        split_res[i].extend(tmp[i])

    tmp, start_index = split_list2kfold(both_name_weight, k, start_index)
    for i in range(k):
        split_res[i].extend(tmp[i])
    # Generate the training data set of four-tuples
    for i in range(k):
        this_root = os.path.join(kfold_path, f'kfold_v{i + 1}')
        os.makedirs(this_root, exist_ok=True)
        dev_names = split_res[i]
        train_names = []
        for j in range(k):
            if j != i:
                train_names.extend(split_res[j])

        train_ins = []
        for na_w in train_names:
            name = na_w[0]
            whole_candi_aids = list(offline_whole_unass[name].keys())
            if len(whole_candi_aids) < configs['train_neg_sample'] + 1:
                continue
            for pos_aid, pids in offline_whole_unass[name].items():
                for pid in pids:
                    neg_aids = copy.deepcopy(whole_candi_aids)
                    neg_aids.remove(pos_aid)
                    neg_aids = random.sample(neg_aids, configs['train_neg_sample'])
                    train_ins.append((name, pid, pos_aid, neg_aids))
        save_json(train_ins, this_root, 'train_ins.json')

        dev_ins = []
        for na_w in dev_names:
            name = na_w[0]
            whole_candi_aids = list(offline_whole_unass[name].keys())
            if len(whole_candi_aids) < configs['test_neg_sample'] + 1:
                continue
            for pos_aid, pids in offline_whole_unass[name].items():
                for pid in pids:
                    neg_aids = copy.deepcopy(whole_candi_aids)
                    neg_aids.remove(pos_aid)
                    neg_aids = random.sample(neg_aids, configs['test_neg_sample'])
                    dev_ins.append((name, pid, pos_aid, neg_aids))
        save_json(dev_ins, this_root, 'test_ins.json')
    print(name_weight)


def splitdata_RND(ret,version):

    random.seed(66)
    v2path = version2path(version)
    pprint(v2path)
    raw_data_root = v2path['raw_data_root']
    processed_data_root = v2path["processed_data_root"]

    # Partition train set by year
    split_train2dev(data=ret,
                    processed_data_root=processed_data_root,
                    unass_ratio=0.2)
    offline_profile = load_json(processed_data_root, "train/offline_profile.json")
    offline_unass = load_json(processed_data_root, "train/offline_unass.json")

    kfold_main_func(processed_data_root,offline_profile, offline_unass, 5)

    # train+valid name2aid2pid
    get_name2aid2pid(raw_data_root,processed_data_root,name2aid2pids_path=RNDFilePathConfig.whole_name2aid2pid)

    # Papers that have not been assigned
    pretreat_unass(raw_data_root,processed_data_root,RNDFilePathConfig.unass_candi_v1_path, "valid/cna_valid_unass.json",
                   "valid/cna_valid_unass_pub.json")
    pretreat_unass(raw_data_root,processed_data_root,RNDFilePathConfig.unass_candi_v2_path, "test/cna_test_unass.json",
                   "test/cna_test_unass_pub.json")


if __name__ == '__main__':
    train, version = load_utils.LoadData(name="v3", type="train", task='RND', partition=None)
    splitdata_RND(train,version)
