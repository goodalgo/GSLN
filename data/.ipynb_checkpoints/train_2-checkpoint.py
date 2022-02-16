import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

import random
random.seed(0)

import numpy as np
np.random.seed(0)

import pandas as pd

import dgl
from dgl.data.utils import load_graphs
from dgl.nn.pytorch import edge_softmax

import copy
import argparse
import time
import os
from sys import path
from os.path import dirname as dir
path.append(dir(path[0]))

import warnings
warnings.filterwarnings('ignore')

from pt_modules.graphsage import SAGEModel
from pt_modules.loss import F1_Loss, BCE_Loss, ROC_Loss, Regression_Loss, DiceBCELoss
from utils.data import prepare_node_feat_data, prepare_node_label_data, train_val_test_mask,load_subtensor
from utils.evaluation import binary_clf_evaluate_batch,regression_evaluate_batch

# from tensorboardX import SummaryWriter


def train_model(args):
    g,features,labels,train_mask,val_mask,test_mask,neighbors,ratio,task_type,agg,self_loop,n_layers = args
    # g = g.to('cuda:0')
    g = dgl.as_heterograph(g)
    g.create_formats_()
    g.in_degree(0)
    g.out_degree(0)
    g.find_edges([0])
    n_edges = g.number_of_edges()
    n_nodes = g.number_of_nodes()
    print(f"Data statistics: "+\
      f"#Nodes {n_nodes:d} | "+\
      f"#Edges {n_edges:d} | "+\
      f"#Train samples {train_mask.int().sum().item():d} | "+\
      f"#Val samples {val_mask.int().sum().item():d} | "+\
      f"#Test samples {test_mask.int().sum().item():d} | "+\
      f"#Black ratio {(labels[test_mask].sum().item()/labels[test_mask].shape[0]):.4f}")

    train_nid = torch.nonzero(train_mask, as_tuple=True)[0] # train sample index list
    val_nid = torch.nonzero(val_mask, as_tuple=True)[0] # val sample index list
    test_nid = torch.nonzero(test_mask, as_tuple=True)[0] # val sample index list

    ### <!--start--> Create PyTorch DataLoader for constructing blocks ###
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(n_layers,return_eids=True) # return_eids: retain eid in original graph as a edge feature
    train_dataloader = dgl.dataloading.NodeDataLoader(
        g,
        train_nid,
        sampler,
        batch_size=len(train_nid),
        shuffle=False,
        drop_last=False,
        num_workers=8)

    val_dataloader = dgl.dataloading.NodeDataLoader(
        g,
        val_nid,
        sampler,
        batch_size=len(val_nid),
        shuffle=False,
        drop_last=False,
        num_workers=8)
    
    test_dataloader = dgl.dataloading.NodeDataLoader(
        g,
        test_nid,
        sampler,
        batch_size=len(test_nid),
        shuffle=False,
        drop_last=False,
        num_workers=8)
    ### <!--end--> Create PyTorch DataLoader for constructing blocks ###
    
    in_feats = features.shape[1]
    out_size = 1
    model = prepare_model(in_feats,out_size,g,neighbors,ratio,agg,self_loop,n_layers)
    
    ### <!--start--> cuda ###
    torch.cuda.set_device(0)
    features = features.cuda()
    labels = torch.LongTensor(labels).cuda()
    # if ratio:
    #     g.edata['w'] = g.edata['w'].cuda()
    model.cuda()
    ### <!--end--> cuda ###

    if task_type == 'classification':
        # black_ratio = labels[train_mask].sum().item()/labels[train_mask].shape[0]
        # white_ratio = 1-black_ratio
        # max_ratio = max(black_ratio,white_ratio)
        # weight = torch.softmax(torch.cuda.FloatTensor([black_ratio-max_ratio,white_ratio-max_ratio]),axis=0)
        # loss_fcn = nn.CrossEntropyLoss(weight=torch.cuda.FloatTensor([0.45,0.55]))
        loss_fcn = DiceBCELoss()
    if task_type == 'regression':
        loss_fcn = Regression_Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-3,amsgrad=True)
    
    dur = []
    best_score = 9999999999 # store min loss on validation set
    best_epoch = None
    best_model = None
    epoch_increase = []
    for epoch in range(10000):
        model.train()
        t0 = time.time()
        
        for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
            blocks = list(map(lambda x: x.to('cuda:0'), blocks))
            if ratio:
                for block in blocks:
                    # assgin edge weight
                    # print(f"block device {block.device}")
                    # print(f"g device {g.device}")
                    block.edata['w'] = g.edata['w'][block.edata[dgl.EID]].cuda()
                # print(block.device)
            # Load the input features as well as output labels
            batch_inputs, batch_labels = load_subtensor(features, labels, seeds, input_nodes)
            
            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            dur.append(time.time() - t0)

            model.eval()
            with torch.no_grad():
                val_loss = 0
                for step, (input_nodes, seeds, blocks) in enumerate(val_dataloader):
                    # Load the input features as well as output labels
                    blocks = list(map(lambda x: x.to('cuda:0'), blocks))
                    if ratio:
                        for block in blocks:
                            # assgin edge weight
                            # print(f"block device {block.device}")
                            # print(f"g device {g.device}")
                            block.edata['w'] = g.edata['w'][block.edata[dgl.EID]].cuda()
                        # print(block.device)
                    batch_inputs, batch_labels = load_subtensor(features, labels, seeds, input_nodes)
                    batch_pred = model(blocks, batch_inputs)
                    val_loss = loss_fcn(batch_pred, batch_labels).item()

            # print(f"Epoch {epoch:4d} | Time(s) {dur[-1]:.4f} | train_loss {loss.item():12.4f} | val_loss {val_loss:12.4f}")
            
        # early stopping
        if val_loss < best_score:
            best_score = val_loss
            best_epoch = epoch
            epoch_increase.append(1) # cur epoch performance better
            best_model = copy.deepcopy(model).state_dict() # deepcopy model trained weights
        else:
            epoch_increase.append(0) # cur epoch doesn't performance better

        if len(epoch_increase)>20+1 and np.sum(epoch_increase[-20:]) == 0: # past 10 epoch all doesn't performance better
            # print(f"early stopping at best epoch {best_epoch}, best loss on val set {best_score} .")
            break # break epoch iteration

    # eval on test set
    # print('eval on test set')
    model = prepare_model(in_feats,out_size,g,neighbors,ratio,agg,self_loop,n_layers)
    model.cuda()
    model.load_state_dict(best_model)
    
    if task_type == 'classification':
        best_f1,best_acc,best_macro_f1,auc,best_f1_threshold,best_acc_threshold,best_macro_f1_threshold\
                                 = binary_clf_evaluate_batch(model,val_dataloader, features, labels, \
                                is_test_phase=False,is_ratio=ratio,edata=g.edata['w'] if ratio else None)
        # print(f"Val Set: best_f1 {best_f1:.4f} | best_acc {best_acc:.4f} | best_macro_f1 {best_macro_f1:.4f} | auc {auc:.4f}")
        # print(f"Val Set: f1_threshold {best_f1_threshold:.4f} | acc_threshold {best_acc_threshold:.4f} | macro_f1_threshold {best_macro_f1_threshold:.4f}")
        
        f1_test,acc_test,macro_f1_test,auc = binary_clf_evaluate_batch(model,test_dataloader, features, labels, \
                                is_test_phase=True,is_ratio=ratio,edata=g.edata['w'] if ratio else None,
                                f1_threshold=best_f1_threshold,acc_threshold=best_acc_threshold,macro_f1_threshold=best_macro_f1_threshold)
        print(f"Test Set: f1 {f1_test:.4f} | acc {acc_test:.4f} | macro_f1 {macro_f1_test:.4f} | auc {auc:.4f}")
    
    if task_type == 'regression':
        mae,mse,med,evs = regression_evaluate_batch(model,test_dataloader, features, labels, \
                                is_test_phase=True,is_ratio=ratio,edata=g.edata['w'] if ratio else None)
        print(f"Test Set: mae {mae:.4f} | mse {mse:.4f} | med {med:.4f} | evs {evs:.4f}")

def prepare_model(in_feats,out_size,g,neighbors,ratio,agg,add_self_loop,n_layers):
    '''
    '''
    model = SAGEModel(in_size=in_feats,
            n_hidden=64,
            out_size=out_size,
            n_layers=n_layers,
            use_bias=False,
            activation = nn.PReLU(),
            norm=nn.BatchNorm1d,
            dropout=0.0,
            aggregator_type=agg,
            use_weighted_edge= ratio,
            neighbors = neighbors,
            self_loop=self_loop)

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Trainning')
    parser.add_argument("--ratio",  action="store_true", default=False)
    parser.add_argument("--neighbors",  action="store_true", default=False)
    parser.add_argument("--task",  type=str, default='classification')
    parser.add_argument("--agg",  type=str, default='mean')
    parser.add_argument("--self-loop", action="store_true", default=False)
    parser.add_argument("--n-layers", type=int, default=1)
    args = parser.parse_args()

    edge_type='invested_by'
    ratio = args.ratio
    neighbors = args.neighbors
    task_type = args.task
    agg = args.agg
    self_loop = args.self_loop
    n_layers = args.n_layers

    glist, _ = load_graphs(f'break_comps_data/dglgraphs/co_legal.1.break.dgl.bin')
    g = glist[0]

    if ratio == True:
        weights = torch.FloatTensor(pd.read_csv(f'break_comps_data/edatas/co_legal.1.edges')['weight'].values.reshape(-1,1))
        # g.edata['w'] = weights
        g.edata['w'] = edge_softmax(g, weights) # softmax weight
    
    ndata_df = pd.read_csv('break_comps_data/ndatas/co_legal.1.clean.csv').rename(columns={'onecomp_id':'node_id'})

    columns_self =['reg_cap_fmt_basic','ent_status_fmt_basic','biz_revenue_new_fin',\
                                      'equity_fin','net_profit_fin','oper_revenue_fin','tax_fin','total_assets_fin',\
                                      'total_liability_fin','total_profit_fin','employee_num','oper_status_annual']
    ndata_df = ndata_df.drop(columns=columns_self)
    features = ndata_df.drop(columns=['node_id','break_label','train_flag']).values
    features = torch.FloatTensor(features)
    print(features.shape)

    labels = ndata_df.break_label.values.astype(int)
    labels = torch.LongTensor(labels)

    for random_seed in range(10):
        node_indegrees = g.in_degrees(np.arange(g.number_of_nodes())) # node in-degrees, aiming to discard nodes that has no in-neighbor
        industrys_mask = (ndata_df['train_flag']==1).values

        train_mask, val_mask, test_mask  = train_val_test_mask(num_samples=features.shape[0],\
                                                                random_seed=random_seed,train_size=0.6,\
                                                                node_in_degrees=node_indegrees,industry_mask=industrys_mask)

        train_mask = torch.BoolTensor(train_mask)
        val_mask = torch.BoolTensor(val_mask)
        test_mask = torch.BoolTensor(test_mask)

        all_black_cnt = labels[train_mask].sum() + labels[val_mask].sum() + labels[test_mask].sum()
        if all_black_cnt < 500: # at least there exists 500 black samples, so that approximately 100 samples in test set.
            break
        print(f"Namespace: label_name='break', industrys=-1, clf_threshold=-1, random_seed={random_seed}, all_black_cnt={all_black_cnt}")
        # exit()
        args = g,features,labels,train_mask,val_mask,test_mask,neighbors,ratio,task_type,agg,self_loop,n_layers # packed params
        
        
        train_model(args) # start training and evaluating
