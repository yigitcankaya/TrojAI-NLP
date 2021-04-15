import numpy as np

import torch
import os

import torch.nn as nn
import numpy as np
import pickle


from torch.optim import Adam, SGD
import utils as ut

import model_factories as m
from collections.abc import Iterable


class GatedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()

        self.model = model
        self.input_size = model.fc_layers[0].in_features if isinstance(model, m.FCLinearModel) else model.rnn.__dict__['input_size']
        bidirectional = 0 if isinstance(model, m.FCLinearModel) else model.rnn.__dict__['bidirectional']
        self.hidden_size = model.fc_layers[-1].out_features if isinstance(model, m.FCLinearModel) else model.rnn.__dict__['hidden_size'] * (int(bidirectional) + 1)
        self.input_gates = None
        self.hidden_gates = None
        self.forward = self.forward_no_gate


    def forward_w_input_gates(self, data):
        assert self.input_gates is not None, "Please call set_gates first"

        data = data * self.input_gates.reshape((self.input_gates.shape[0], 1, self.input_gates.shape[-1]))

        hidden = self.model.get_hidden(data)
        
        output = self.model.linear(hidden)
        
        return output
  
    def forward_w_hidden_gates(self, data):

        assert self.hidden_gates is not None, "Please call set_gates first"

        hidden = self.model.get_hidden(data)

        hidden = hidden * self.hidden_gates

        output = self.model.linear(hidden)

        return output

    def set_gates(self, ng, device):
        self.input_gates = torch.nn.Parameter(torch.ones((ng, self.input_size), requires_grad=True).to(device))

        self.hidden_gates = torch.nn.Parameter(torch.ones((ng, self.hidden_size), requires_grad=True).to(device))

    def forward_no_gate(self, data):
        output = self.model(data)
        return output


########################## CDRP FUNCTIONS #######################################
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


def cdrp_loss(gates, gamma, preds, targets, oh=False, mean=True, reg_type='l1'):
    xent_loss = cross_entropy_softlabels(preds, targets, mean) if not oh else cross_entropy_onehot(preds, targets, mean)
    reg = regularizer_loss(gates, gamma, mean, reg_type)
    loss = xent_loss + reg
    return loss

def regularizer_loss(gates, gamma, mean, reg_type):
    
    if reg_type == 'l1':
        ones = torch.ones(gates.shape).to(gates.device)
        loss = gamma*(torch.norm(gates - ones, p=1, dim=1).mean()) if mean else (gamma*torch.norm(gates - ones, p=1, dim=1)).sum()

    elif reg_type == 'l2':
        ones = torch.ones(gates.shape).to(gates.device)
        loss = gamma*(torch.norm(gates - ones, p=2, dim=1).mean()) if mean else (gamma*torch.norm(gates - ones, p=2, dim=1)).sum()

    elif reg_type == 'sparse':
        loss = gamma*(torch.norm(gates, p=1, dim=1).mean()) if mean else (gamma*torch.norm(gates, p=1, dim=1)).sum()

    return loss


def cross_entropy_softlabels(preds, targets, mean):
    loss = -(targets * preds).sum(dim=1).mean() if mean else -(targets * preds).sum(dim=1).sum()
    return loss

def cross_entropy_onehot(preds, targets, mean):
    loss = nn.NLLLoss(reduction='mean')(preds, targets) if mean else nn.NLLLoss(reduction='sum')(preds, targets)
    return loss

def update_gamma(gamma, last_good_gamma, upper, lower, acc_threshold, conf_threshold, acc_and_conf):

    if isinstance(gamma, Iterable):
        for idx, g in enumerate(gamma):
            if acc_and_conf[idx][0] < acc_threshold:# or acc_and_conf[idx][1] < conf_threshold:
                gamma[idx] = g - ((g - lower[idx])/2) # decrease, makes pruning less strict
            else:
                last_good_gamma[idx] = g
                gamma[idx] = g + ((upper[idx] - g)/2) # increase, makes pruning more strict

    else:
        if acc_and_conf[0] < acc_threshold:# or acc_and_conf[1] < conf_threshold:
            gamma = gamma - ((gamma - lower)/2)
        else:
            last_good_gamma = gamma
            gamma = gamma + ((upper - gamma)/2)

    return gamma, last_good_gamma

def find_gamma_range(gamma, last_good_gamma, acc_threshold, conf_threshold, acc_and_conf):

    if isinstance(gamma, Iterable):
        for idx, g in enumerate(gamma):
            if acc_and_conf[idx][0] < acc_threshold:# or acc_and_conf[idx][1] < conf_threshold:
                gamma[idx] = g/10
            else:
                last_good_gamma[idx] = g
                gamma[idx] = g*10 # increase, makes pruning more strict

    else:
        if acc_and_conf[0] < acc_threshold:# or acc_and_conf[1] < conf_threshold:
            gamma = gamma/10
        else:
            last_good_gamma = gamma
            gamma = gamma*10

    return gamma, last_good_gamma
    


def apply_cdrp_for_gamma(gated_model, embeddings, gamma, params, target, preds_wo_gates, use_amp, device):
    iters = params['iter']
    lr = params['lr']
    eps = params['eps']
    gate_type = params['gate_type']
    gate_granularity = params['gate_granularity']
    reg_type = params['reg_type']

    ng = len(embeddings) if gate_granularity == 'per_sample' else 1
    mean = (gate_granularity != 'per_sample')

    gated_model.set_gates(ng, device)
    gates = gated_model.hidden_gates if gate_type == 'hidden' else gated_model.input_gates
    optimizer = Adam([gates], lr=lr, amsgrad=True)
    for _ in range(iters):
        optimizer.zero_grad()  #clear gradients for this training step

        preds_w_gates = query(gated_model, embeddings, True, use_amp)

        if target is None:
            loss = cdrp_loss(gates, gamma, preds_w_gates, preds_wo_gates, oh=False, mean=mean, reg_type=reg_type)

        else:
            target_ = torch.ones(preds_w_gates.shape[0]).to(device).long() * target
            loss = cdrp_loss(gates, gamma, preds_w_gates, target_, oh=True, mean=mean, reg_type=reg_type)

        loss.backward(retain_graph=True)
        optimizer.step()
        if reg_type == 'sparse':
            gates.data.clamp_(0, 5) # use sparse l1 descent, only alter a few dimensions
            gates.data[torch.abs(gates.data) <= eps] = 0 # round to zero
        else:
            gates.data.clamp_(-5, 5) # use sparse l1 descent, only alter a few dimensions
    
    gates.data[torch.abs(gates.data) <= eps] = 0 # round to zero
    gates = gates.detach().cpu().numpy()

    if ng == 1:
        gates = gates.flatten()
    else:
        gates = gates.reshape((gates.shape[0], gates.shape[-1]))

    targets = preds_wo_gates.detach().cpu().numpy() if target is None else target
    preds_w_gates = np.exp(query(gated_model, embeddings, True, use_amp).detach().cpu().numpy())
    acc_and_conf = get_acc_ind(preds_w_gates, targets) if gate_granularity == 'per_sample' else get_acc_mean(preds_w_gates, targets)

    return gates, acc_and_conf

def cdrp_with_binary_search(gated_model, embeddings, params, use_amp, target=None, device='cpu'):

    gate_granularity = params['gate_granularity']
    mean = (gate_granularity != 'per_sample')
    gate_type = params['gate_type']
    ng = len(embeddings) if gate_granularity == 'per_sample' else 1


    acc_threshold, conf_threshold = params['threshold']
    gamma_cur = params['start'] if mean else torch.ones(ng).to(device)*params['start']
    last_good_gamma = 0 if mean else torch.zeros(ng).to(device)

    max_range_find_steps, max_gamma_find_steps = 6, 8


    gated_model.eval()
    freeze_model(gated_model)
    gated_model.forward = gated_model.forward_w_hidden_gates if gate_type == 'hidden' else gated_model.forward_w_input_gates
    preds_wo_gates = query(gated_model, embeddings, False, use_amp)

    all_gates = []
    all_accs = []

    cur_step = 0
    while cur_step < max_range_find_steps:
        _, acc_and_conf = apply_cdrp_for_gamma(gated_model, embeddings, gamma_cur, params, target, preds_wo_gates, use_amp, device)
        gamma_cur, last_good_gamma = find_gamma_range(gamma_cur, last_good_gamma, acc_threshold, conf_threshold, acc_and_conf)
        cur_step += 1

    upper = last_good_gamma*10 if mean else (last_good_gamma.clone().detach().to(device))*10
    lower = last_good_gamma/10 if mean else (last_good_gamma.clone().detach().to(device))/10
    gamma_cur = (upper + lower)/2

    cur_step = 0
    while cur_step < max_gamma_find_steps:
        _, acc_and_conf = apply_cdrp_for_gamma(gated_model, embeddings, gamma_cur, params, target, preds_wo_gates, use_amp, device)
        gamma_cur, last_good_gamma = update_gamma(gamma_cur, last_good_gamma, upper, lower, acc_threshold, conf_threshold, acc_and_conf)
        cur_step += 1

    gates, acc_and_conf = apply_cdrp_for_gamma(gated_model, embeddings, last_good_gamma, params, target, preds_wo_gates, use_amp, device)
    
    last_good_gamma = last_good_gamma.cpu().numpy() if gate_granularity == 'per_sample' else last_good_gamma
    return gates, acc_and_conf, last_good_gamma


def apply_cdrp_on_single_model(model, embeddings, labels, cdrp_params, use_amp, device):

    gate_granularity = cdrp_params['gate_granularity']

    gated_model = GatedModel(model)
    gated_model = gated_model.to(device)

    embeddings_ = np.expand_dims(embeddings, axis=1) # sequence length 1
    embeddings_ = torch.from_numpy(embeddings_).to(device)

    c1_indices, c0_indices = np.where(labels == 1)[0], np.where(labels == 0)[0]

    if gate_granularity == 'per_class':
        c1_embeddings, c0_embeddings = embeddings_[c1_indices], embeddings_[c0_indices]
    else:
        c1_embeddings, c0_embeddings = embeddings_, embeddings_

    c0_gates, c0_accs, c0_gammas = cdrp_with_binary_search(gated_model, c1_embeddings, cdrp_params, use_amp, target=0, device=device)
    c1_gates, c1_accs, c1_gammas = cdrp_with_binary_search(gated_model, c0_embeddings, cdrp_params, use_amp, target=1, device=device)
    preserve_gates, preserve_accs, preserve_gammas = cdrp_with_binary_search(gated_model, embeddings_, cdrp_params, use_amp, target=None, device=device)

    return (c0_gates, c1_gates, preserve_gates), (c0_accs, c1_accs, preserve_accs), (c0_gammas, c1_gammas, preserve_gammas), (c0_indices, c1_indices)


def apply_cdrp_on_all_models(df, main_path, models_path, cdrp_params, round_suffix, use_amp, device):

    gate_type = cdrp_params['gate_type']
    gate_granularity = cdrp_params['gate_granularity']
    subset = cdrp_params['subset'] if ('subset' in cdrp_params and cdrp_params['subset'] != 'all') else len(df['model_name'])

    print(f'Collecting CDRPs for {subset} models with {cdrp_params["reg_type"]} regularization...')

    with open(os.path.join(f'data_{round_suffix}', 'embeddings.pickle'), 'rb') as handle:
        data =  pickle.load(handle)
    
    all_embeddings, all_labels = data['embeddings'], data['instance_labels']
    
    all_gates = []
    all_accs = []
    all_gammas = []
    trigger_targets = []
    class_indices = []

    for idx, _ in enumerate(df['model_name'][:subset]):
        params = ut.read_model(df, idx, main_path, models_path)
        print(f'Idx: {idx} - Poisoned: {params[2]} - Embedding: {os.path.basename(params[6])} - Arch: {params[1]} - Gate Type: {gate_type} - Gate Granularity: {gate_granularity}')

        model = torch.load(params[3], map_location=device)
        embeddings, labels = all_embeddings[idx], all_labels[idx]

        gates_data, accs_data, gammas_data, c_indices = apply_cdrp_on_single_model(model, embeddings, labels, cdrp_params, use_amp, device)
        all_gates.append(gates_data); all_accs.append(accs_data); all_gammas.append(gammas_data); class_indices.append(c_indices)

        trigger_targets.append(params[-1])

    return all_gates, all_accs, all_gammas, class_indices, data['model_labels'], trigger_targets
    

def get_acc_mean(preds, targets):
    classes = np.argmax(preds, axis=1)

    if isinstance(targets, int):
        t = targets
        conf = np.mean(preds[:, t])
    else:
        t = np.argmax(targets, axis=1)
        conf = np.mean(preds[np.arange(len(preds)), t])

    tot_correct = np.sum(classes == t)
    acc = tot_correct/len(classes)

    return np.array([acc, conf])

def get_acc_ind(preds, targets):
    classes = np.argmax(preds, axis=1)

    if isinstance(targets, int):
        t = targets
        conf = preds[:, t]
    else:
        t = np.argmax(targets, axis=1)
        conf = preds[np.arange(len(preds)), t]

    corrects = (classes == t).astype(int)

    ret = np.vstack((corrects, conf)).T
    
    return ret