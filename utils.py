

import os
import numpy as np
import copy
import torch
import transformers
import pandas
import math

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
        else:
            # for GPT-2 use last token as the text summary
            embedding_vector = embedding_vector[:, -1, :]

        embedding_vector = embedding_vector.to('cpu')
        embedding_vector = embedding_vector.numpy()

        # reshape embedding vector to create batch size of 1
        embedding_vector_np = np.expand_dims(embedding_vector, axis=0)

    embedding_vector = torch.from_numpy(embedding_vector_np).to(device)
            
    # predict the text sentiment
    if use_amp:
        with torch.cuda.amp.autocast():
            logits = model(embedding_vector).cpu().detach().numpy()
    else:
        logits = model(embedding_vector).cpu().detach().numpy()

    embedding_vector_np = embedding_vector_np.flatten()

    return embedding_vector_np, logits

def get_sentiment_on_examples(examples_dirpath, tokenizer, embedding, model, max_input_length, cls_token_is_first, use_amp, device, trigger_texts):
   # Inference the example images in data
    fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith('.txt')]
    fns.sort()  # ensure file ordering

    embeddings = []
    logits = []
    labels = []

    for fn in fns:
        # load the example
        with open(fn, 'r') as fh:
            try:
                text = fh.readline()
            except:
                continue
        
        fn_base = os.path.basename(fn)

        if 'source' in fn_base:
            true_label = int(fn_base.split('_')[2])
        else:
            true_label = int(fn_base.split('_')[1])

        text_clean = text

        if trigger_texts is not None:
            for trigger_text in trigger_texts:
                text_clean = text_clean.replace(trigger_text, '')

        cur_embedding, cur_logits = get_sentiment(text_clean, tokenizer, embedding, model, max_input_length, cls_token_is_first, use_amp, device)

        embeddings.append(cur_embedding)
        logits.append(cur_logits)
        labels.append(true_label)

    return np.vstack(logits), np.vstack(embeddings), np.asarray(labels)


def get_sentiment_on_embeddings(embeddings, labels, model, use_amp, device):

    embeddings_ = np.expand_dims(embeddings, axis=1) # sequence length 1
    embeddings_ = torch.from_numpy(embeddings_).to(device)
                
    # predict the text sentiment
    if use_amp:
        with torch.cuda.amp.autocast():
            logits = model(embeddings_).cpu().detach().numpy()
    else:
        logits = model(embeddings_).cpu().detach().numpy()

    preds = np.argmax(logits, axis=1)
    is_correct = np.zeros(len(logits))
    is_correct[np.where(np.equal(preds, labels))[0]] = 1

    return logits, is_correct

def embedding_distance(model_filepath, cls_token_is_first, tokenizer_filepath, embedding_filepath, examples_dirpath, poison_examples_dirpath, trigger_texts):
    # load the classification model and move it to the GPU

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.load(model_filepath, map_location=device)

    model.eval()

  
    tokenizer = torch.load(tokenizer_filepath)
    # set the padding token if its undefined
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # load the specified embedding
    embedding = torch.load(embedding_filepath, map_location=device)

    embedding.eval()

    # identify the max sequence length for the given embedding
    max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path]

    use_amp = True  # attempt to use mixed precision to accelerate embedding conversion process

    clean_logits, clean_embeddings, clean_labels = get_sentiment_on_examples(examples_dirpath, tokenizer, embedding, model, max_input_length, cls_token_is_first, use_amp, device, None)

    if trigger_texts is None:
        return clean_logits, clean_embeddings, clean_labels

    triggered_logits, triggered_embeddings, triggered_labels = get_sentiment_on_examples(poison_examples_dirpath, tokenizer, embedding, model, max_input_length, cls_token_is_first, use_amp, device, None)
    trig_removed_logits, trig_removed_embeddings, trig_removed_labels = get_sentiment_on_examples(poison_examples_dirpath, tokenizer, embedding, model, max_input_length, cls_token_is_first, use_amp, device, trigger_texts)

    return clean_logits, clean_embeddings, clean_labels, triggered_logits, triggered_embeddings, triggered_labels, trig_removed_logits, trig_removed_embeddings, trig_removed_labels


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
    config_file = os.path.join(cur_path, 'config.json')
    poison_examples_dirpath = os.path.join(cur_path, 'poisoned_example_data')

    exist = os.path.exists(model_filepath)

    if poisoned:
        with open(config_file) as f:
            config = json.load(f)

        trigger_texts = [t['text'] for t in config['triggers']]
    
    else:
        trigger_texts = None
    

    return exist, arch, poisoned, model_filepath, cls_token_is_first, tokenizer_filepath, embedding_filepath, examples_dirpath, poison_examples_dirpath, trigger_texts