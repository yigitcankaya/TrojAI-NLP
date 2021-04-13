
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

    # Inference the example images in data
    fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith('.txt')]
    fns.sort()  # ensure file ordering


    tokenizer = torch.load(tokenizer_filepath)

    # identify the max sequence length for the given embedding
    max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path]

    # set the padding token if its undefined
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    embedding = torch.load(embedding_filepath, map_location=device).eval()

    ### get the embeddings on the given data samples ###
    _, clean_embeddings, clean_labels = get_sentiment_on_examples(examples_dirpath, tokenizer, embedding, model, max_input_length, cls_token_is_first, use_amp, device, None)

    cdrp_params = {'threshold':(0.95, 0.7), 'start':0.05, 'iter':50, 'lr':0.1, 'eps':1e-2, 'gate_granularity':'all', 'reg_type':'l1', 'subset':'all'}

    # l1 reg gates
    cdrp_params['gate_type'] = 'hidden' 
    _, (c0_haccs, c1_haccs, preserve_haccs), _, _ = apply_cdrp_on_single_model(model, clean_embeddings, clean_labels, cdrp_params, use_amp, device) 
    cdrp_params['gate_type'] = 'input'
    _, (c0_iaccs, c1_iaccs, preserve_iaccs), _, _ = apply_cdrp_on_single_model(model, clean_embeddings, clean_labels, cdrp_params, use_amp, device) 

    features = np.array([[c0_haccs[1], c1_haccs[1], preserve_haccs[1], c0_iaccs[1], c1_iaccs[1], preserve_iaccs[1]]])

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
    use_amp = True if torch.cuda.is_available() else False # attempt to use mixed precision to accelerate embedding conversion process
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


