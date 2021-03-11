
import os
import numpy as np
import copy
import torch
import transformers

import warnings
import pickle

import torch.nn as nn

from torch.optim import Adam
from scipy.special import softmax

warnings.filterwarnings("ignore")

############## COLLECT THE EMBEDDINGS #############

def get_sentiment(text, tokenizer, embedding, model, max_input_length, cls_token_is_first):
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


def get_sentiment_on_examples(examples_dirpath, tokenizer, embedding, model, max_input_length, cls_token_is_first):
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

        true_label = int(fn_base.split('_')[1])

        cur_embedding, cur_logits = get_sentiment(text, tokenizer, embedding, model, max_input_length, cls_token_is_first)

        embeddings.append(cur_embedding)
        logits.append(cur_logits)
        labels.append(true_label)

    return np.vstack(logits), np.vstack(embeddings), np.asarray(labels)

########## PRUNING FEATURES #################
class GatedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()

        self.model = model
        self.input_size = model.rnn.__dict__['input_size']
        bidirectional = model.rnn.__dict__['bidirectional']
        self.hidden_size = model.rnn.__dict__['hidden_size'] * (int(bidirectional) + 1)
        self.input_gates = torch.nn.Parameter(torch.ones((1, 1, self.input_size), requires_grad=True))
        self.hidden_gates = torch.nn.Parameter(torch.ones((1, self.hidden_size), requires_grad=True))
        self.forward = self.forward_no_gate


    def forward_w_input_gates(self, data):
        data = data * self.input_gates
        hidden = self.model.get_hidden(data)
        output = self.model.linear(hidden)
        return output

    def forward_w_hidden_gates(self, data):
        hidden = self.model.get_hidden(data)
        hidden = hidden * self.hidden_gates
        output = self.model.linear(hidden)
        return output

    def reset_gates(self):
        device = self.input_gates.device

        self.input_gates = torch.nn.Parameter(torch.ones((1, 1, self.input_size), requires_grad=True).to(device))
        self.hidden_gates = torch.nn.Parameter(torch.ones((1, self.hidden_size), requires_grad=True).to(device))

    def forward_no_gate(self, data):
        output = self.model(data)
        return output

def query(gated_model, embeddings, gates=True, use_amp=True):
    # predict the text sentiment
    if use_amp:
        with torch.cuda.amp.autocast():
            logits = gated_model.forward(embeddings) if gates else gated_model.forward_no_gate(embeddings)
    else:
        logits = gated_model.forward(embeddings) if gates else gated_model.forward_no_gate(embeddings)

    preds = nn.functional.log_softmax(logits, dim=1)

    return preds

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def cdrp_loss(gates, gamma, preds, targets, oh=False):
    xent_loss = cross_entropy_softlabels(preds, targets) if not oh else cross_entropy_onehot(preds, targets)
    sparsity_loss = torch.norm(gates, p=1)   
    loss = xent_loss + gamma*sparsity_loss
    return loss


def cross_entropy_softlabels(preds, targets):
    return -(targets * preds).sum(dim=1).mean()

def cross_entropy_onehot(preds, targets):
    return nn.NLLLoss()(preds, targets)


def cdrp_for_embedding(gated_model, embeddings, params, use_amp, gate_type='hidden', target=None, device='cpu'):
    gamma = params['gamma']
    iters = params['iter']
    lr = params['lr']
    eps = params['eps']

    gated_model.eval()
    freeze_model(gated_model)

    embeddings_ = np.expand_dims(embeddings, axis=1) # sequence length 1
    embeddings_ = torch.from_numpy(embeddings_).to(device)
    
    gated_model.reset_gates()
    gated_model.forward = gated_model.forward_w_hidden_gates if gate_type == 'hidden' else gated_model.forward_w_input_gates
    gates = gated_model.hidden_gates if gate_type == 'hidden' else gated_model.input_gates


    preds_wo_gates = query(gated_model, embeddings_, False, use_amp)
    optimizer = Adam([gates], lr=lr, amsgrad=True)


    for _ in range(iters):
        optimizer.zero_grad()  #clear gradients for this training step

        preds_w_gates = query(gated_model, embeddings_, True, use_amp)

        if target is None:
            loss = cdrp_loss(gates, gamma, preds_w_gates, preds_wo_gates, oh=False)

        else:
            target_ = torch.ones(preds_w_gates.shape[0]).to(device).long() * target
            loss = cdrp_loss(gates, gamma, preds_w_gates, target_, oh=True)

        loss.backward(retain_graph=True)
        optimizer.step()
        gates.data.clamp_(0, 5) # clip between 0 and 10
        gates.data[gates.data <= eps] = 0 # round down

    preds_w_gates = query(gated_model, embeddings_, True, use_amp)

    return gates.detach().cpu().numpy().flatten(), np.exp(preds_wo_gates.detach().cpu().numpy()), np.exp(preds_w_gates.detach().cpu().numpy())


############################ CONVERT GATES FEATURES ###############################################
def get_gates_data(gated_model, clean_embeddings, cdrp_hgates_params, cdrp_igates_params):

    c0_h = cdrp_for_embedding(gated_model, clean_embeddings, cdrp_hgates_params, use_amp, gate_type='hidden', target=0, device=device)
    c1_h = cdrp_for_embedding(gated_model, clean_embeddings, cdrp_hgates_params, use_amp, gate_type='hidden', target=1, device=device)
    preserve_h = cdrp_for_embedding(gated_model, clean_embeddings, cdrp_hgates_params, use_amp, gate_type='hidden', target=None, device=device)

    c0_i = cdrp_for_embedding(gated_model, clean_embeddings, cdrp_igates_params, use_amp, gate_type='input', target=0, device=device)
    c1_i = cdrp_for_embedding(gated_model, clean_embeddings, cdrp_igates_params, use_amp, gate_type='input', target=1, device=device)
    preserve_i = cdrp_for_embedding(gated_model, clean_embeddings, cdrp_igates_params, use_amp, gate_type='input', target=None, device=device)

    c0_hgates, c0_hgates_preds, full_preds = c0_h[0].reshape(1, len(c0_h[0])), c0_h[2].reshape(1, *c0_h[2].shape), c0_h[1].reshape(1, *c0_h[1].shape)
    c1_hgates, c1_hgates_preds = c1_h[0].reshape(1, len(c1_h[0])), c1_h[2].reshape(1, *c1_h[2].shape)
    preserve_hgates, preserve_hgates_preds = preserve_h[0].reshape(1, len(preserve_h[0])), preserve_h[2].reshape(1, *preserve_h[2].shape)

    c0_igates, c0_igates_preds = c0_i[0].reshape(1, len(c0_i[0])), c0_i[2].reshape(1, *c0_i[2].shape)
    c1_igates, c1_igates_preds = c1_i[0].reshape(1, len(c1_i[0])), c1_i[2].reshape(1, *c1_i[2].shape)
    preserve_igates, preserve_igates_preds = preserve_i[0].reshape(1, len(preserve_i[0])), preserve_i[2].reshape(1, *preserve_i[2].shape)


    data = {}
    
    data['c0_hgates'], data['c1_hgates'], data['preserve_hgates'] = c0_hgates, c1_hgates, preserve_hgates
    data['c0_hgates_preds'], data['c1_hgates_preds'] , data['preserve_hgates_preds']  = c0_hgates_preds, c1_hgates_preds, preserve_hgates_preds

    data['c0_igates'], data['c1_igates'], data['preserve_igates']  = c0_igates, c1_igates, preserve_igates
    data['c0_igates_preds'], data['c1_igates_preds'], data['preserve_igates_preds'] = c0_igates_preds, c1_igates_preds, preserve_igates_preds

    data['full_preds']  = full_preds

    return data

def convert_data(data,  num_bins, conversion='moments'):

    converter = get_moments if conversion == 'moments' else get_histogram

    all_c0_hgates, all_c1_hgates, all_preserve_hgates = data['c0_hgates'], data['c1_hgates'], data['preserve_hgates']
    all_c0_igates, all_c1_igates, all_preserve_igates = data['c0_igates'], data['c1_igates'], data['preserve_igates'] 

    all_c0_hgates_preds, all_c1_hgates_preds, all_preserve_hgates_preds = data['c0_hgates_preds'], data['c1_hgates_preds'] , data['preserve_hgates_preds'] 
    all_c0_igates_preds, all_c1_igates_preds, all_preserve_igates_preds = data['c0_igates_preds'], data['c1_igates_preds'], data['preserve_igates_preds']
    all_full_preds = data['full_preds']

    accs_hgates = np.hstack((get_accs(all_c0_hgates_preds, 0), get_accs(all_c1_hgates_preds, 1), get_accs(all_preserve_hgates_preds, all_full_preds)))
    accs_igates = np.hstack((get_accs(all_c0_igates_preds, 0), get_accs(all_c1_igates_preds, 1), get_accs(all_preserve_igates_preds, all_full_preds)))

    data_c0_hgates = converter(all_c0_hgates, num_bins=num_bins)
    data_c1_hgates = converter(all_c1_hgates, num_bins=num_bins)
    data_preserve_hgates = converter(all_preserve_hgates, num_bins=num_bins)

    data_c0_igates = converter(all_c0_igates, num_bins=num_bins)
    data_c1_igates = converter(all_c1_igates, num_bins=num_bins)
    data_preserve_igates = converter(all_preserve_igates, num_bins=num_bins)

    coverted_data = np.hstack((data_c0_hgates, data_c1_hgates, data_preserve_hgates, data_c0_igates, data_c1_igates, data_preserve_igates, accs_hgates, accs_igates))

    return coverted_data


def get_accs(preds, targets):

    accs = []
    avg_confs = []

    for model_idx, cur_preds in enumerate(preds):
        classes = np.argmax(cur_preds, axis=1)

        if isinstance(targets, int):
            t = targets
            avg_confs.append(np.mean(cur_preds[:, t]))
        else:
            t = np.argmax(targets[model_idx], axis=1)
            avg_confs.append(np.mean(cur_preds[np.arange(len(cur_preds)), t]))

        tot_correct = np.sum(classes == t)
        accs.append(tot_correct/len(classes))

    accs = np.array(accs)
    avg_confs = np.array(avg_confs)        
    return np.vstack((accs, avg_confs)).T
    


def get_histogram(data, num_bins=20):
    
    new_data = np.zeros((len(data), num_bins))
    for model_idx, l in enumerate(data):
        num_zeros = np.sum(l == 0)/len(l)
        nonzeros = (np.histogram(l, np.linspace(1e-12, 5, num_bins))[0])/len(l)
        
        new_data[model_idx][0] = num_zeros
        new_data[model_idx][1:] = nonzeros


    return new_data


def get_moments(data, num_bins=20):
    new_data = np.zeros((len(data), num_bins))
    for sample_idx, sample in enumerate(data):
        for moment in range(1, num_bins+1):
            new_data[sample_idx][moment-1] = (np.sum(sample**moment)**(1/moment))/len(sample)
    
    return new_data

##########################################  DETECTOR  ####################################################################################################


def trojan_detector(model_filepath, cls_token_is_first, tokenizer_filepath, embedding_filepath, result_filepath, scratch_dirpath, examples_dirpath):

    # load the classification model and move it to the GPU
    model = torch.load(model_filepath, map_location=device)

    # Inference the example images in data
    fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith('.txt')]
    fns.sort()  # ensure file ordering


    tokenizer = torch.load(tokenizer_filepath)

    # identify the max sequence length for the given embedding
    max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path]

    # set the padding token if its undefined
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    embedding = torch.load(embedding_filepath, map_location=device)

    ### get the embeddings on the given data samples ###
    _, clean_embeddings, _ = get_sentiment_on_examples(examples_dirpath, tokenizer, embedding, model, max_input_length, cls_token_is_first)


    #### query the gates model #####
    NUM_BINS = 10
    CONVERSION = 'histogram'

    gated_model = GatedModel(model)
    gated_model = gated_model.to(device)

    cdrp_hgates_params_weakest = {'gamma':0.0025, 'iter':50, 'lr':0.1, 'eps':1e-3}
    cdrp_igates_params_weakest = {'gamma':0.025, 'iter':50, 'lr':0.1, 'eps':1e-3}
    weakest_gates_data = convert_data(get_gates_data(gated_model, clean_embeddings, cdrp_hgates_params_weakest, cdrp_igates_params_weakest), num_bins=NUM_BINS, conversion=CONVERSION)

    cdrp_hgates_params_weak = {'gamma':0.005, 'iter':50, 'lr':0.1, 'eps':1e-3}
    cdrp_igates_params_weak = {'gamma':0.05, 'iter':50, 'lr':0.1, 'eps':1e-3}
    weak_gates_data = convert_data(get_gates_data(gated_model, clean_embeddings, cdrp_hgates_params_weak, cdrp_igates_params_weak), num_bins=NUM_BINS, conversion=CONVERSION)

    cdrp_hgates_params_medium = {'gamma':0.0075, 'iter':50, 'lr':0.1, 'eps':1e-3}
    cdrp_igates_params_medium = {'gamma':0.075, 'iter':50, 'lr':0.1, 'eps':1e-3}
    medium_gates_data = convert_data(get_gates_data(gated_model, clean_embeddings, cdrp_hgates_params_medium, cdrp_igates_params_medium), num_bins=NUM_BINS, conversion=CONVERSION)

    cdrp_hgates_params_strong = {'gamma':0.01, 'iter':50, 'lr':0.1, 'eps':1e-3}
    cdrp_igates_params_strong = {'gamma':0.1, 'iter':50, 'lr':0.1, 'eps':1e-3}
    strong_gates_data = convert_data(get_gates_data(gated_model, clean_embeddings, cdrp_hgates_params_strong, cdrp_igates_params_strong), num_bins=NUM_BINS, conversion=CONVERSION)

    cdrp_hgates_params_strongest = {'gamma':0.0125, 'iter':50, 'lr':0.1, 'eps':1e-3}
    cdrp_igates_params_strongest = {'gamma':0.125, 'iter':50, 'lr':0.1, 'eps':1e-3}
    strongest_gates_data = convert_data(get_gates_data(gated_model, clean_embeddings, cdrp_hgates_params_strongest, cdrp_igates_params_strongest), num_bins=NUM_BINS, conversion=CONVERSION)

    gates_features = np.hstack((weakest_gates_data, weak_gates_data, medium_gates_data, strong_gates_data, strongest_gates_data))

    trojan_prob = softmax(clf.predict_proba(gates_features)*2, axis=1)[0][1] # soften the predictions 
    
    ### write the result in the file ###
    print(f'Trojan Probability: {trojan_prob:.2f}')

    with open(result_filepath, 'w') as fh:
        fh.write("{}".format(trojan_prob))


if __name__ == "__main__":


    with open(os.path.join(os.sep, 'detector', 'clf_gates.pickle'), 'rb') as handle:
        clf = pickle.load(handle)

    # with open('clf_gates.pickle', 'rb') as handle:
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


