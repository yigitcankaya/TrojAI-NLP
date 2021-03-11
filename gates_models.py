import numpy as np

import torch
import os

import torch.nn as nn
import numpy as np


from torch.optim import Adam
import utils as ut


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



def apply_cdrp_on_dataset(df, main_path, models_path, cdrp_hgates_params, cdrp_igates_params, use_amp, device):
    all_c0_hgates, all_c1_hgates, all_c0_igates, all_c1_igates = [], [], [], []
    all_c0_hgates_preds, all_c1_hgates_preds, all_c0_igates_preds, all_c1_igates_preds, all_full_preds = [], [], [], [], []

    all_preserve_hgates, all_preserve_igates = [], []
    all_preserve_hgates_preds, all_preserve_igates_preds = [], []


    model_labels = []
    sample_labels = []

    for idx, _ in enumerate(df['model_name']):
        params = ut.read_model(df, idx, main_path, models_path)
        print(f'Idx: {idx} - Poisoned: {params[2]} - Embedding: {os.path.basename(params[6])} - Arch: {params[1]}')

        model_labels.append(int(params[2]))

        model = torch.load(params[3], map_location=device)
        gated_model = GatedModel(model)
        gated_model = gated_model.to(device)
        params = list(params)
        params[-1] = None
        params = tuple(params)
        _, clean_embeddings, clean_labels = ut.embedding_distance(*params[3:])
        sample_labels.append(clean_labels)

        use_amp = True if torch.cuda.is_available() else False # attempt to use mixed precision to accelerate embedding conversion process
        c0_hgates = cdrp_for_embedding(gated_model, clean_embeddings, cdrp_hgates_params, use_amp, gate_type='hidden', target=0, device=device)
        c1_hgates = cdrp_for_embedding(gated_model, clean_embeddings, cdrp_hgates_params, use_amp, gate_type='hidden', target=1, device=device)
        preserve_hgates = cdrp_for_embedding(gated_model, clean_embeddings, cdrp_hgates_params, use_amp, gate_type='hidden', target=None, device=device)

        all_c0_hgates.append(c0_hgates[0]); all_c1_hgates.append(c1_hgates[0])
        all_full_preds.append(c0_hgates[1]); all_c0_hgates_preds.append(c0_hgates[2]); all_c1_hgates_preds.append(c1_hgates[2])
        all_preserve_hgates.append(preserve_hgates[0]); all_preserve_hgates_preds.append(preserve_hgates[2])


        c0_igates = cdrp_for_embedding(gated_model, clean_embeddings, cdrp_igates_params, use_amp, gate_type='input', target=0, device=device)
        c1_igates = cdrp_for_embedding(gated_model, clean_embeddings, cdrp_igates_params, use_amp, gate_type='input', target=1, device=device)
        preserve_igates = cdrp_for_embedding(gated_model, clean_embeddings, cdrp_igates_params, use_amp, gate_type='input', target=None, device=device)

        all_c0_igates.append(c0_igates[0]); all_c1_igates.append(c1_igates[0])
        all_c0_igates_preds.append(c0_igates[2]); all_c1_igates_preds.append(c1_igates[2])
        all_preserve_igates.append(preserve_igates[0]); all_preserve_igates_preds.append(preserve_igates[2])
    
        data = {}
        
        data['c0_hgates'], data['c1_hgates'], data['preserve_hgates'] = all_c0_hgates, all_c1_hgates, all_preserve_hgates
        data['c0_hgates_preds'], data['c1_hgates_preds'] , data['preserve_hgates_preds']  = all_c0_hgates_preds, all_c1_hgates_preds, all_preserve_hgates_preds

        data['c0_igates'], data['c1_igates'], data['preserve_igates']  = all_c0_igates, all_c1_igates, all_preserve_igates
        data['c0_igates_preds'], data['c1_igates_preds'], data['preserve_igates_preds'] = all_c0_igates_preds, all_c1_igates_preds, all_preserve_igates_preds

        data['full_preds'], data['model_labels'], data['sample_labels']  = all_full_preds, model_labels, sample_labels


    return data

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
    

def convert_data(data,  num_bins, conversion='moments'):

    converter = get_moments if conversion == 'moments' else get_histogram

    all_c0_hgates, all_c1_hgates, all_preserve_hgates = data['c0_hgates'], data['c1_hgates'], data['preserve_hgates']
    all_c0_igates, all_c1_igates, all_preserve_igates = data['c0_igates'], data['c1_igates'], data['preserve_igates'] 

    all_c0_hgates_preds, all_c1_hgates_preds, all_preserve_hgates_preds = data['c0_hgates_preds'], data['c1_hgates_preds'] , data['preserve_hgates_preds'] 
    all_c0_igates_preds, all_c1_igates_preds, all_preserve_igates_preds = data['c0_igates_preds'], data['c1_igates_preds'], data['preserve_igates_preds']
    all_full_preds, model_labels = data['full_preds'], data['model_labels']


    accs_hgates = np.hstack((get_accs(all_c0_hgates_preds, 0), get_accs(all_c1_hgates_preds, 1), get_accs(all_preserve_hgates_preds, all_full_preds)))
    accs_igates = np.hstack((get_accs(all_c0_igates_preds, 0), get_accs(all_c1_igates_preds, 1), get_accs(all_preserve_igates_preds, all_full_preds)))

    data_c0_hgates = converter(all_c0_hgates, num_bins=num_bins)
    data_c1_hgates = converter(all_c1_hgates, num_bins=num_bins)
    data_preserve_hgates = converter(all_preserve_hgates, num_bins=num_bins)

    data_c0_igates = converter(all_c0_igates, num_bins=num_bins)
    data_c1_igates = converter(all_c1_igates, num_bins=num_bins)
    data_preserve_igates = converter(all_preserve_igates, num_bins=num_bins)

    coverted_data = {}
    coverted_data['data'] = np.hstack((data_c0_hgates, data_c1_hgates, data_preserve_hgates, data_c0_igates, data_c1_igates, data_preserve_igates, accs_hgates, accs_igates))
    coverted_data['labels'] = model_labels

    return coverted_data

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