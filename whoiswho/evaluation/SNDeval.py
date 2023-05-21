import numpy as np
import os
from os.path import join
from tqdm import tqdm
from datetime import datetime
from whoiswho.dataset.data_process import read_pubs,read_raw_pubs
from whoiswho.utils import load_json, save_json

def pairwise_evaluate(correct_labels, pred_labels):
    """Pairwise evaluation.

    Args:
        correct_labels: ground-truth labels (Numpy Array).
        pred_labels: predicted labels (Numpy Array).

    Returns:
        pairwise_precision (Float).
        pairwise_recall (Float).
        pairwise_f1 (Float).

    """

    TP = 0.0  # Pairs Correctly Predicted To SameAuthor
    TP_FP = 0.0  # Total Pairs Predicted To SameAuthor
    TP_FN = 0.0  # Total Pairs To SameAuthor

    for i in range(len(correct_labels)):
        for j in range(i + 1, len(correct_labels)):
            if correct_labels[i] == correct_labels[j]:
                TP_FN += 1
            if pred_labels[i] == pred_labels[j]:
                TP_FP += 1
            if (correct_labels[i] == correct_labels[j]) and (pred_labels[i] == pred_labels[j]):
                TP += 1

    if TP == 0:
        pairwise_precision = 0
        pairwise_recall = 0
        pairwise_f1 = 0
    else:
        pairwise_precision = TP / TP_FP
        pairwise_recall = TP / TP_FN
        pairwise_f1 = (2 * pairwise_precision * pairwise_recall) / (pairwise_precision + pairwise_recall)

    return pairwise_precision, pairwise_recall, pairwise_f1


def evaluate(name, pubs, pred, labels, mode, cur_time):
    """Evaluating with ground-truth.

    Args:
        name: disambiguating name (str).
        pubs: papers of this name (List).
        pred: predicted labels (Numpy Array).

    Returns:
        pairwise_precision (Float).
        pairwise_recall (Float).
        pairwise_f1 (Float).

    """
    labels = np.array(labels)
    pred = np.array(pred)
    pred_label_num = len(set(pred))
    true_label_num = len(set(labels))
    pairwise_precision, pairwise_recall, pairwise_f1 = pairwise_evaluate(labels, pred)

    save_dir = './whoiswho/training/snd_result'
    os.makedirs(save_dir, exist_ok=True)
    log_path = join(save_dir, 'log', mode)
    os.makedirs(log_path, exist_ok=True)
    log_file = join(log_path, f'log_{cur_time}.txt')

    with open(log_file, 'a') as f:
        f.write(f'name: {name}, prec: {pairwise_precision: .4}, recall: {pairwise_recall: .4}, '
                f'f1: {pairwise_f1: .4}, pred label num : {pred_label_num}/{true_label_num}\n')
    f.close()

    return pairwise_precision, pairwise_recall, pairwise_f1


def total_evaluate(assignment,ground_truth,type):
    # 1.assignment 加载预测结果 snd_result中的json
    # 2.for循环中提供label 与 evaluate
    if isinstance(assignment, str):
        assignment=load_json(assignment)
    if isinstance(ground_truth, str):
        ground_truth=load_json(ground_truth)

    submit_data=assignment

    pubs = read_pubs(type)
    raw_pubs = read_raw_pubs(type)
    result = {}
    precisions, recalls, f1s = [], [], []
    cur_time = datetime.now().strftime("%m%d%H%M")
    for n, name in enumerate(tqdm(raw_pubs)):
        if type == 'train':
            pubs = []
            ilabel = 0
            labels = []
            for aid in raw_pubs[name]:
                pubs.extend(raw_pubs[name][aid])
                labels.extend([ilabel] * len(raw_pubs[name][aid]))
                ilabel += 1
        elif type == 'valid' or 'test':
            pubs = raw_pubs[name]
        else:
            print("Invalid type!")



        # 逐name evaluate指标
        if type == 'test' or type == 'valid':
            pass
        else:
            # 'train'
            pairwise_prec, pairwise_rec, pairwise_f1 = evaluate(name, pubs, pred, labels, mode, cur_time)
            precisions.append(pairwise_prec)
            recalls.append(pairwise_rec)
            f1s.append(pairwise_f1)

    if type == 'test' or type == 'valid':
        pass
    else:
        #'train'
        log_file = join('./whoiswho/training/snd_result', 'log', type, f'log_{cur_time}.txt')
        with open(log_file, 'a') as f:
            f.write(f'AVG, prec: {np.mean(precisions): .4}, recall: {np.mean(recalls): .4}, '
                    f'f1: {np.mean(f1s): .4}\n')  #各name的平均聚类指标 np.mean

