

import os
import numpy as np
import copy
import torch
import transformers
import pandas
import math
import pickle

import warnings
import json

warnings.filterwarnings("ignore")


def get_sentiment(text, tokenizer, embedding, model, max_input_length, cls_token_is_first, use_amp, device):

    # tokenize the text
    results = tokenizer(text, max_length=max_input_length - 2, padding=True, truncation=True, return_tensors="pt")
    # extract the input token ids and the attention mask
    input_ids = results.data['input_ids']
    attention_mask = results.data['attention_mask']

    # convert to embedding
    with torch.no_grad():
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        if use_amp:
            with torch.cuda.amp.autocast():
                embedding_vector = embedding(input_ids, attention_mask=attention_mask)[0]
        else:
            embedding_vector = embedding(input_ids, attention_mask=attention_mask)[0]
        

        # ignore all but the first embedding since this is sentiment classification
        if cls_token_is_first:
            embedding_vector = embedding_vector[:, 0, :]
            embedding_vector = embedding_vector.cpu().detach().numpy()

        else:
            embedding_vector = embedding_vector.cpu().detach().numpy()
            attn_mask = attention_mask.detach().cpu().detach().numpy()
            emb_list = list()
            for i in range(attn_mask.shape[0]):
                idx = int(np.argwhere(attn_mask[i, :] == 1)[-1])
                emb_list.append(embedding_vector[i, idx, :])
            embedding_vector = np.stack(emb_list, axis=0)

        # reshape embedding vector to create batch size of 1
        embedding_vector = np.expand_dims(embedding_vector, axis=0)
        # embedding_vector is [1, 1, <embedding length>]

        embedding_vector = torch.from_numpy(embedding_vector).to(device)

            
    # predict the text sentiment
    if use_amp:
        with torch.cuda.amp.autocast():
            logits = model(embedding_vector).cpu().detach().numpy()
    else:
        logits = model(embedding_vector).cpu().detach().numpy()

    embedding_vector_np = embedding_vector.cpu().detach().numpy().flatten()

    return embedding_vector_np, logits

def get_sentiment_on_examples(examples_dirpath, tokenizer, embedding, model, max_input_length, cls_token_is_first, use_amp, device):
   # Inference the example images in data
    fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith('.txt')]
    fns.sort()  # ensure file ordering

    embeddings = []
    logits = []
    labels = []


    class_idx = -1
    while True:
        class_idx += 1
        fn = 'class_{}_example_{}.txt'.format(class_idx, 1)
        if not os.path.exists(os.path.join(examples_dirpath, fn)):
            break

        example_idx = 0
        while True:
            example_idx += 1
            fn = 'class_{}_example_{}.txt'.format(class_idx, example_idx)
            if not os.path.exists(os.path.join(examples_dirpath, fn)):
                break

            # load the example
            
            with open(os.path.join(examples_dirpath, fn), 'r') as fh:
                try:
                    text = fh.read()
                except:
                    continue
            
            cur_embedding, cur_logits = get_sentiment(text, tokenizer, embedding, model, max_input_length, cls_token_is_first, use_amp, device)

            embeddings.append(cur_embedding)
            logits.append(cur_logits)
            labels.append(class_idx)

    return np.vstack(logits), np.vstack(embeddings), np.asarray(labels)


def get_all_embeddings(model_filepath, cls_token_is_first, tokenizer_filepath, embedding_filepath, examples_dirpath, use_amp):
    # load the classification model and move it to the GPU

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.load(model_filepath, map_location=device).eval()

    # load the specified embedding
    embedding = torch.load(embedding_filepath, map_location=device).eval()
  
    tokenizer = torch.load(tokenizer_filepath)
    # set the padding token if its undefined
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # identify the max sequence length for the given embedding
    max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path]

    logits, embeddings, labels = get_sentiment_on_examples(examples_dirpath, tokenizer, embedding, model, max_input_length, cls_token_is_first, use_amp, device)
    return logits, embeddings, labels


def read_model(df, model_idx, main_path, models_path):

    model_name = df['model_name'][model_idx]
    cur_path = os.path.join(models_path, model_name)
    
    embedding =  df['embedding'][model_idx]
    embedding_flavor = df['embedding_flavor'][model_idx]
    embedding_filepath = os.path.join(main_path, 'embeddings', f'{embedding}-{embedding_flavor}.pt')
    tokenizer_filepath = os.path.join(main_path, 'tokenizers', f'{embedding}-{embedding_flavor}.pt')
    arch = df['model_architecture'][model_idx]
    poisoned = df['poisoned'][model_idx]
    model_filepath = os.path.join(cur_path, 'model.pt')
    cls_token_is_first = df['cls_token_is_first'][model_idx]
    examples_dirpath = os.path.join(cur_path, 'clean_example_data')
    exist = os.path.exists(model_filepath)

    return exist, arch, poisoned, model_filepath, cls_token_is_first, tokenizer_filepath, embedding_filepath, examples_dirpath


def write_embeddings_on_file(df, main_path, models_path, round_suffix, use_amp):

    all_embeddings, all_logits, all_labels = [], [], []
    filename = os.path.join(f'data_{round_suffix}', 'embeddings.pickle')
    model_labels = []

    for idx, _ in enumerate(df['model_name']):
        params = read_model(df, idx, main_path, models_path)
        print(f'Idx: {idx} - Poisoned: {params[2]} - Embedding: {os.path.basename(params[6])} - Arch: {params[1]}')
        
        model_labels.append(int(params[2]))

        logits, embeddings, labels = get_all_embeddings(*params[3:], use_amp=use_amp)
        all_embeddings.append(embeddings); all_logits.append(logits); all_labels.append(labels)

    data = {}
    data['embeddings'], data['logits'], data['instance_labels'] = all_embeddings, all_logits, all_labels
    data['model_labels'] = model_labels

    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
