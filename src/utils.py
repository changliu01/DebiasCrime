import logging
import numpy as np
import torch
import gc
import sys
import torch.utils.data as Data
from tqdm import tqdm
import argparse
from sklearn import metrics



def run_epoch(args, mode, loader, model, descriminator, enable_tqdm=True):
    gc.collect()
    torch.cuda.empty_cache()

    model.to(args.device)
    descriminator.to(args.device)
    if mode == 'train':
        model.train()
        descriminator.train()
    else:
        model.eval()
        descriminator.eval()

    collect = dict(batch_size=[], loss=[], crime_pred=[], crime_true=[], demo_pred=[], demo_true=[])    
    for idx, (crime_feat, crime_label, demo_feat, demo_label, adj, corr) in tqdm(enumerate(loader), total=len(loader), desc=mode, disable=not enable_tqdm):
        """
        - crime_feat: (batch_size, obsv_len, region_num, crime_feature_num)
        - crime_label: (batch_size, pred_len, region_num, crime_feature_num)
        - demo_feat: (batch_size, region_num, demo_feat_num)
        - demo_label: (batch_size, region_num), in {0, 1, 2}
        - adj: (region_num, region_num)
        - corr: (crime_feature_num, crime_feature_num)
        """
        crime_feat  = crime_feat.to(args.device)
        crime_label = crime_label.to(args.device)
        demo_feat   = demo_feat.to(args.device)
        demo_label  = demo_label.to(args.device)
        adj         = adj.to(args.device)
        corr        = corr.to(args.device)

        # Forward
        with torch.set_grad_enabled(mode == 'train'):
            output = model(crime_feat, adj, demo_feat, corr, crime_label)
            adv_output = descriminator(output['filter_vectors'], demo_label)
            if args.no_adversary:
                loss = output['loss']
            else:
                loss = output['loss'] - args.reg_weight * adv_output['loss']
        
        # Update model
        if mode == 'train':
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
        
        # Update descriminator
        if mode == 'train':
            for _ in range(args.d_steps):
                adv_output2 = descriminator(output['filter_vectors'].detach(), demo_label)
                descriminator.optimizer.zero_grad()
                adv_output2['loss'].backward()
                descriminator.optimizer.step()
        
        # Record result
        batch_size = crime_feat.shape[0]
        collect['batch_size'].append(batch_size)
        collect['loss'].append(loss.item() * batch_size)
        collect['crime_pred'].append(output['prediction'].detach().cpu().numpy())
        collect['crime_true'].append(crime_label.detach().cpu().numpy())
        collect['demo_pred'].append(adv_output['output'].softmax(dim=-1).detach().cpu().numpy())
        collect['demo_true'].append(demo_label.detach().cpu().numpy())
    
    # Collect result
    collect['crime_pred'] = np.concatenate(collect['crime_pred'], axis=0).flatten()
    collect['crime_true'] = np.concatenate(collect['crime_true'], axis=0).flatten()
    collect['demo_pred'] = np.concatenate(collect['demo_pred'], axis=0).reshape(-1, 3)  # .flatten(0, 1)
    collect['demo_true'] = np.concatenate(collect['demo_true'], axis=0).flatten()
    result = dict(
        sample_num = np.sum(collect['batch_size']),
        mean_loss = np.sum(collect['loss']) / np.sum(collect['batch_size']),
        crime_acc = metrics.accuracy_score(collect['crime_true'], collect['crime_pred'] > 0.178),
        crime_f1 = metrics.f1_score(collect['crime_true'], collect['crime_pred'] > 0.178, average='macro'),
        crime_auc = metrics.roc_auc_score(collect['crime_true'], collect['crime_pred'], average='macro'),
        demo_acc = metrics.accuracy_score(collect['demo_true'], collect['demo_pred'].argmax(axis=-1)),
        demo_f1 = metrics.f1_score(collect['demo_true'], collect['demo_pred'].argmax(axis=-1), average='macro'),
        demo_auc = metrics.roc_auc_score(collect['demo_true'], collect['demo_pred'], multi_class='ovo'),
    )
    return result

class Dataset(Data.Dataset):
    def __init__(self, data_path, obsv_len, pred_len):
        super().__init__()
        data = np.load(data_path, allow_pickle=True)
        crime = data['incident'] # (time, region, crime-feature)
        adj = data['s_adj'] # (region, region)
        corr = data['c_cor'] # (crime-feature, crime-feature)
        demography = data['feature'] # (region, demography-feat)
        demo_type = np.argmax(demography[:, (1,3,5)], axis=-1) # (region,)

        self.time_num, self.region_num, self.crime_feature_num = crime.shape
        self.demo_feat_num = demography.shape[1]
        self.demo_class_num = 3
        self.obsv_len = obsv_len
        self.pred_len = pred_len

        self.crime = torch.from_numpy(crime).float()
        self.adj = torch.from_numpy(adj).float()
        self.corr = torch.from_numpy(corr).float()
        self.demo_feat = torch.from_numpy(demography).float()
        self.demo_label = torch.from_numpy(demo_type).long()

    def __getitem__(self, idx):
        crime_feat = self.crime[idx:idx+self.obsv_len, :, :]
        crime_label = self.crime[idx+self.obsv_len:idx+self.obsv_len+self.pred_len, :, :]
        crime_label = crime_label[0, :, :].any(dim=-1).float()
        return (crime_feat, crime_label, self.demo_feat, self.demo_label, self.adj, self.corr)
    
    def __len__(self):
        return self.time_num - self.obsv_len - self.pred_len + 1
