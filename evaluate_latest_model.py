#!/usr/bin/env python3
"""
Script to evaluate the latest CDistNet model on the LMDB dataset we created.
"""

import os
import sys
import torch
from mmcv import Config

# Add the project root to the path
sys.path.insert(0, '/home/homeai/Documents/GitHub/CDistNet')

from cdistnet.data.data import make_lmdb_data_loader_test
from cdistnet.model.translator import Translator
from cdistnet.model.model import build_CDistNet
import codecs
import csv
from tqdm import tqdm
import glob


def load_vocab(vocab=None, vocab_size=None):
    """
    Load vocab from disk. The fisrt four items in the vocab should be <PAD>, <UNK>, <S>, </S>
    """
    vocab = [' ' if len(line.split()) == 0 else line.split()[0] for line in codecs.open(vocab, 'r', 'utf-8')]
    vocab = vocab[:vocab_size]
    assert len(vocab) == vocab_size
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


def get_alphabet(dict_path):
    with open(dict_path, "r") as f:
        data = f.readlines()
        data = list(map(lambda x: x.strip(), data))
        data = data[4:]
    return data


def get_pred_gt_name(translator, idx2word, b_image, b_gt, b_name, num, dict_path, rotate, rgb2gray, is_test_gt=True):
    gt_list, name_list, pred_list = [], [], []
    alphabet = get_alphabet(dict_path)  # not used
    if rotate:
        batch_hyp, batch_scores = translator.translate_batch(
            images=b_image.view(-1, b_image.shape[-2], b_image.shape[-1]).unsqueeze(dim=1)
        )
        batch_scores = torch.cat(batch_scores, dim=0).view(-1, 3)
        _, idx = torch.max(batch_scores, 1)
        idx = torch.arange(0, idx.shape[0], dtype=torch.long) * 3 + idx.cpu()
        batch_hyp_ = []
        for id, v in enumerate(batch_hyp):
            if id in idx:
                batch_hyp_.append(v)
        batch_hyp = batch_hyp_
    else:
        if rgb2gray == False:
            batch_hyp, batch_scores = translator.translate_batch(images=b_image[:, :3, :, :])
        else:
            batch_hyp, batch_scores = translator.translate_batch(images=b_image[:, 0:1, :, :])
    for idx, seqs in enumerate(batch_hyp):
        for seq in seqs:
            seq = [x for x in seq if x != 3]
            pred = [idx2word[x] for x in seq]
            pred = ''.join(pred)
        flag = False
        if is_test_gt == False:
            num += 1
            pred_list.append('word_{}.png'.format(num) + ', "' + pred + '"\n')
            gt_list.append('word_{}.png'.format(num) + ', "' + b_gt[idx] + '"\n')
            name_list.append(b_name[idx] + '\n')
        else:
            num += 1
            pred_list.append('{}'.format(b_name[idx]) + ', "' + pred + '"\n')
            gt_list.append('{}'.format(b_name[idx]) + ', "' + b_gt[idx] + '"\n')
            name_list.append(b_name[idx] + '\n')
    return gt_list, name_list, pred_list, num


def write_to_file(file_name, datas):
    with open(file_name, "w") as f:
        f.writelines(datas)


def evaluate_model_on_lmdb():
    # Configuration for evaluation
    cfg = Config.fromfile('/home/homeai/Documents/GitHub/CDistNet/configs/eval_config.py')
    
    # Find the latest model in the tps_persian_cdistnet directory
    model_dir = "/home/homeai/Documents/GitHub/CDistNet/models/tps_persian_cdistnet"
    model_files = glob.glob(os.path.join(model_dir, "*.pth"))
    
    # Find the best accuracy model
    best_acc_models = [f for f in model_files if "best_acc" in f]
    if best_acc_models:
        # Use the latest best accuracy model
        latest_model_path = max(best_acc_models, key=os.path.getctime)
    else:
        # If no best_acc model found, use the most recent model
        latest_model_path = max(model_files, key=os.path.getctime)
    
    print(f"Using model: {latest_model_path}")
    
    # Build the model
    model = build_CDistNet(cfg)
    model.load_state_dict(torch.load(latest_model_path, map_location='cpu'))
    device = torch.device(cfg.test.device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Create translator
    translator = Translator(cfg, model=model)
    word2idx, idx2word = load_vocab(cfg.dst_vocab, cfg.dst_vocab_size)
    
    # Evaluate on the LMDB dataset
    data_name = "/home/homeai/Documents/GitHub/CDistNet/dataset/eval_lmdb"
    print("dataset name: {}".format(data_name))
    
    test_dataloader = make_lmdb_data_loader_test(cfg, [data_name])

    gt_list, name_list, pred_list = [], [], []
    num = 0

    print("Starting evaluation...")
    # Evaluate
    for iteration, batch in enumerate(tqdm(test_dataloader)):
        b_image, b_gt, b_name = batch[0], batch[1], batch[2]
        gt_list_, name_list_, pred_list_, num = get_pred_gt_name(
            translator, idx2word, b_image, b_gt, b_name, num, cfg.dst_vocab, 
            cfg.test.rotate, cfg.rgb2gray, cfg.test.is_test_gt
        )
        gt_list += gt_list_
        name_list += name_list_
        pred_list += pred_list_

    # Write results to files
    gt_file = os.path.join(model_dir, 'gt.txt')
    pred_file = os.path.join(model_dir, 'submit.txt')
    name_file = os.path.join(model_dir, 'name.txt')
    write_to_file(gt_file, gt_list)
    write_to_file(pred_file, pred_list)
    write_to_file(name_file, name_list)
    
    # Calculate accuracy
    correct = 0
    total = len(gt_list)
    for i in range(total):
        # Extract the text from the ground truth and prediction
        gt_text = gt_list[i].split('"')[1] if '"' in gt_list[i] else gt_list[i].split(',')[1].strip()
        pred_text = pred_list[i].split('"')[1] if '"' in pred_list[i] else pred_list[i].split(',')[1].strip()
        
        if gt_text == pred_text:
            correct += 1
    
    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"Evaluation completed!")
    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Save results to CSV
    result_path = os.path.join(model_dir, 'result.csv')
    with open(result_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([os.path.basename(latest_model_path), accuracy])
    
    return accuracy


if __name__ == "__main__":
    evaluate_model_on_lmdb()