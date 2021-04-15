
import os
import numpy as np
import copy
import torch
import transformers

import warnings
import pickle

import torch.nn as nn

from gates_models import apply_cdrp_on_single_model
from utils import get_sentiment_on_examples

warnings.filterwarnings("ignore")


def trojan_detector(model_filepath, cls_token_is_first, tokenizer_filepath, embedding_filepath, result_filepath, scratch_dirpath, examples_dirpath):

    # load the classification model and move it to the GPU
    model = torch.load(model_filepath, map_location=device).eval()

    tokenizer = torch.load(tokenizer_filepath)

    # identify the max sequence length for the given embedding
    max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path]

    # set the padding token if its undefined
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    embedding = torch.load(embedding_filepath, map_location=device).eval()

    ### get the embeddings on the given data samples ###
    _, clean_embeddings, clean_labels = get_sentiment_on_examples(examples_dirpath, tokenizer, embedding, model, max_input_length, cls_token_is_first, use_amp, device)


    ### extract the features from the gates ###
    cdrp_params = {'threshold':(0.95, 0.7), 'start':0.05, 'iter':50, 'lr':0.1, 'eps':1e-2, 'gate_granularity':'all', 'subset':'all', 'gate_type':'input'}

    cdrp = lambda params: apply_cdrp_on_single_model(model, clean_embeddings, clean_labels, params, use_amp, device) 
    d = lambda x: np.linalg.norm(x - 1, ord=1)
    dd = lambda x, y: np.linalg.norm(x - y, ord=1)
    s = lambda x: len(np.where(x == 0)[0])/len(x)

    cdrp_params['reg_type'] = 'l1'
    (l1_c0_gates, l1_c1_gates, _), (l1_c0_accs, l1_c1_accs, l1_preserve_accs), (l1_c0_gammas, l1_c1_gammas, _),  _ = cdrp(cdrp_params)

    cdrp_params['reg_type'] = 'l2'
    (l2_c0_gates, l2_c1_gates, _), (l2_c0_accs, l2_c1_accs, l2_preserve_accs), (l2_c0_gammas, l2_c1_gammas, _),  _ = cdrp(cdrp_params)

    cdrp_params['reg_type'] = 'sparse'
    (sp_c0_gates, sp_c1_gates, sp_pr_gates), (sp_c0_accs, sp_c1_accs, sp_preserve_accs), (sp_c0_gammas, sp_c1_gammas, sp_pr_gammas),  _ = cdrp(cdrp_params)

    # confidence of the pruned models
    l1_confs = (l1_c0_accs[1], l1_c1_accs[1], l1_preserve_accs[1])
    l2_confs = (l2_c0_accs[1], l2_c1_accs[1], l2_preserve_accs[1])
    sp_confs = (sp_c0_accs[1], sp_c1_accs[1], sp_preserve_accs[1])

    # gamma values for the pruned models
    l1_gammas = (l1_c0_gammas, l1_c1_gammas)
    l2_gammas = (l2_c0_gammas, l2_c1_gammas)
    sp_gammas = (sp_c0_gammas, sp_c1_gammas, sp_pr_gammas)

    # distance (for l1 and l2 pruning) or sparsity (for sparsity pruning) gates
    l1_gates = (d(l1_c0_gates), d(l1_c1_gates), dd(l1_c0_gates, l1_c1_gates))
    l2_gates = (d(l2_c0_gates), d(l2_c1_gates), dd(l2_c0_gates, l2_c1_gates))
    sp_gates = (s(sp_c0_gates), s(sp_c1_gates), s(sp_pr_gates))

    features = np.array([[*l1_confs, *l1_gammas, *l1_gates, *l2_confs, *l2_gammas, *l2_gates, *sp_confs, *sp_gammas, *sp_gates]])

    ## query the model ###
    trojan_prob = clf.predict_proba(features)[0][1]
    
    ### write the result in the file ###
    print(f'Trojan Probability: {trojan_prob:.4f}')

    with open(result_filepath, 'w') as fh:
        fh.write("{}".format(trojan_prob))

if __name__ == "__main__":
    with open(os.path.join(os.sep, 'detector', 'clf.pickle'), 'rb') as handle:
        clf = pickle.load(handle)

    # with open('clf.pickle', 'rb') as handle:
    #     clf = pickle.load(handle)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = False # True if torch.cuda.is_available() else False # attempt to use mixed precision to accelerate embedding conversion process
    torch.backends.cudnn.enabled=False

    import argparse

    parser = argparse.ArgumentParser(description='TrojAI - Trojan Detector of UMD (ICSI).')
    parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.', default='./model/model.pt')
    parser.add_argument('--cls_token_is_first', help='Whether the first embedding token should be used as the summary of the text sequence, or the last token.', action='store_true', default=False)
    parser.add_argument('--tokenizer_filepath', type=str, help='File path to the pytorch model (.pt) file containing the correct tokenizer to be used with the model_filepath.', default='./model/tokenizer.pt')
    parser.add_argument('--embedding_filepath', type=str, help='File path to the pytorch model (.pt) file containing the correct embedding to be used with the model_filepath.', default='./model/embedding.pt')
    parser.add_argument('--result_filepath', type=str, help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.', default='./output.txt')
    parser.add_argument('--scratch_dirpath', type=str, help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.', default='./scratch')
    parser.add_argument('--examples_dirpath', type=str, help='File path to the folder of examples which might be useful for determining whether a model is poisoned.', default='./model/clean_example_data')

    args = parser.parse_args()

    trojan_detector(args.model_filepath, args.cls_token_is_first, args.tokenizer_filepath, args.embedding_filepath, args.result_filepath, args.scratch_dirpath, args.examples_dirpath)


