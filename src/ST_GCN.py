import torch
import torch.nn as nn
import math
import numpy as np
import logging
import os

class GCNConv(nn.Module):
    def __init__(self, input_feat, output_feat):    
        super().__init__()
        self.input_feat      = input_feat
        self.output_feat     = output_feat
        self.neighbor_linear = nn.Linear(input_feat, output_feat)
        self.self_linear     = nn.Linear(input_feat, output_feat)
    
    def forward(self, x, adj):
        """
        x: tensor of shape [batch_size, num_nodes, num_features]
        """
        neighbor = torch.bmm(adj, x)
        x = (self.self_linear(x) + self.neighbor_linear(neighbor)).relu()
        return x

class PositionalEncoder(nn.Module):
    def __init__(self, dropout=0.1, max_seq_len: int=5000, d_model: int=512, batch_first: bool=True):
        super().__init__()
        self.d_model     = d_model
        self.dropout     = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        self.x_dim       = 1 if batch_first else 0
        position         = torch.arange(max_seq_len).unsqueeze(1)
        div_term         = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe               = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2]      = torch.sin(position * div_term)
        pe[:, 1::2]      = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x = x + self.pe.squeeze(1)[:x.size(self.x_dim)]
        d_model = x.shape[-1]
        position = torch.arange(x.shape[1], device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=x.device) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(x.shape[1], d_model, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        if True:
            x = x + pe
        else:
            x = x + pe.unsqueeze(1)
        return self.dropout(x)

class ST_GCN(torch.nn.Module):
    def __init__(self, args, num_regions, data_reader):
        super().__init__()
        self.model_path     = args.model_path
        self.input_feat     = args.input_feat
        self.output_feat    = args.output_feat
        self.gcn_emb_size   = args.gcn_emb_size
        self.demo_feat_size = args.demo_feat_size
        self.gcn_layer_num  = args.gcn_layer_num
        self.region_num     = num_regions
        self.demo_feat      = args.demo_feat_size
        self.demo_hidden    = args.demo_hidden
        self.window_size    = args.window_size
        self.heads          = args.heads
        self.trans_layers   = args.trans_layers
        self.wo_demo        = args.wo_demo
        self.gcns           = nn.ModuleList()
        self.gcns.append(GCNConv(self.input_feat*self.window_size, self.gcn_emb_size))
        for _ in range(self.gcn_layer_num - 1):
            self.gcns.append(GCNConv(self.gcn_emb_size, self.gcn_emb_size))
        if True:
            self.positional_encoding_layer = PositionalEncoder(
                d_model=self.input_feat*self.region_num, 
                dropout=0.1, 
                max_seq_len=args.window_size, 
                batch_first=True
            )
        else:
            self.positional_encoding_layer = PositionalEncoder(
                d_model=self.input_feat, 
                dropout=0.1, 
                max_seq_len=args.window_size, 
                batch_first=True
            )
        if True:
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_feat * self.region_num, nhead=self.heads)
        else:
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_feat, nhead=4)
        self.temp = nn.TransformerEncoder(encoder_layer, num_layers=self.trans_layers)

        self.cat_gcn_layer_num = args.cat_gcn_layer_num
        self.demo_linear = nn.Linear(self.demo_feat, self.demo_hidden, bias=False)
        self.pred_MLP = nn.Sequential(
            nn.Linear(self.input_feat, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid())
        self.W_Q = nn.Linear(self.input_feat*self.region_num, self.input_feat*self.region_num, bias=False)
        self.W_K = nn.Linear(self.input_feat*self.region_num, self.input_feat*self.region_num, bias=False)
        self.W_V = nn.Linear(self.input_feat*self.region_num, self.input_feat*self.region_num, bias=False)
        self.query = nn.Parameter(torch.randn(self.input_feat*self.region_num, 1))
        self.att_linear = nn.Linear(self.input_feat*self.region_num, self.input_feat*self.region_num)
        self.merge_linear = nn.Linear(2*self.input_feat*self.region_num, self.input_feat*self.region_num)
        self.cat_gcns = nn.ModuleList()
        self.cat_gcns.append(GCNConv(self.input_feat, self.input_feat))
        for _ in range(self.cat_gcn_layer_num - 1):
            self.cat_gcns.append(GCNConv(self.input_feat, self.input_feat))
        self.data_reader = data_reader
        self.filter_mode = args.filter_mode
        self.pred_dense = nn.Linear(self.input_feat + self.demo_hidden, self.input_feat)
        self.optimizer = None
        self.criterion = nn.BCELoss()

        self.num_features = 1  # Race
        self.filter_num = self.num_features if self.filter_mode == 'combine' else (2 ** self.num_features)
        self.filter_dict = nn.ModuleDict(
            {str(i + 1): nn.Sequential(
                nn.Linear(self.input_feat, self.input_feat * 2),
                nn.LeakyReLU(),
                nn.Linear(self.input_feat * 2, self.input_feat),
                nn.LeakyReLU(),
            ) for i in range(self.filter_num)}
        )

    def forward(self, x, adj, demography, corr, label, filter_mask=[True]):
        """
        - x: [batch_size, time_steps, region_num, feature_num(crime)]
        - adj: [batch_size, region_num, region_num]
        - demography: [batch_size, region_num, feature_num(demo)]
        - corr: [batch_size, feature_num(crime), feature_num(crime)]
        - label: [batch_size, region_num]
        - filter_mask: 
        - return:
            - prediction: [batch_size, region_num]
            - filter_vectors: [batch_size, region_num, embed_dim]
            - ori_vectors: [batch_size, region_num, embed_dim]
            - loss: float
        """
        _x = x.permute(0, 2, 1, 3) # (batch_size, region_num, time_steps, num_features)
        vector = self.SpatialConv(_x, adj)
        vector = self.TemporalConv(vector)
        filter_vectors = self.apply_filter(vector, filter_mask)
        output = self.pred_MLP(filter_vectors).squeeze(-1) # (batch_size, region_num)
        loss = self.criterion(output, label)
        out_dict = {
            'prediction':output, # (batch_size, region_num)
            'filter_vectors':filter_vectors, # (batch_size, region_num, embed_dim)
            'ori_vectors':vector, # (batch_size, region_num, embed_dim)
            'loss':loss,
        }
        return out_dict

    def l2(self):
        return sum(map(lambda x: (x ** 2).sum(), self.parameters()))

    @staticmethod
    def init_weights(m):
        """
        initialize nn weights，called in main.py
        :param m: parameter or the nn
        :return:
        """
        if type(m) == torch.nn.Linear:
            torch.nn.init.kaiming_uniform_(m.weight, a=1, nonlinearity='sigmoid')
            if m.bias is not None:
                m.bias.data.zero_()
        elif type(m) == torch.nn.Embedding:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def SpatialConv(self, x, adj):
        """
        x: [batch_size, region_num, time_steps, feature_num]
        adj: [batch_size, region_num, region_num]
        return: [batch_size, region_num, embed_dim]
        """
        shape_x = x.shape
        x = x.flatten(-2, -1)
        for gcn in self.gcns:
            x = gcn(x, adj)
        x = x.view(*shape_x[:3], -1)
        return x
    
    def TemporalConv(self, x):
        """
        x: [batch_size, region_num, time_steps, embed_dim]
        return: [batch_size, region_num, embed_dim]
        """
        x = x.permute(0, 2, 1, 3) # (batch_size, time_steps, region_num, num_features)
        if True:
            x = x.flatten(-2, -1)
            x = self.positional_encoding_layer(x)
            all_time_hidden = self.temp(x)
        else:
            x = self.positional_encoding_layer(x)
            all_time_hidden = self.temp(x.permute(0, 2, 1, 3).flatten(0, 1))

        Q = self.W_Q(all_time_hidden)
        K = self.W_K(all_time_hidden)
        V = self.W_V(all_time_hidden)
        att = torch.bmm(Q, K.permute(0, 2, 1)).softmax(dim=-1)
        time_feat_after_att = torch.bmm(att, V) #(batch_size, time_steps,num_feat)
        keys = time_feat_after_att.tanh()
        time_att = torch.bmm(keys, self.query.unsqueeze(0).repeat(keys.shape[0], 1, 1))
        time_att = time_att.softmax(dim=1)
        time_att = torch.bmm(time_att.permute(0, 2, 1), time_feat_after_att).squeeze(1)

        x = torch.concat((time_att, all_time_hidden[:, -1, :]), dim=-1)
        x = self.merge_linear(x)
        x = x.reshape(x.shape[0], self.region_num, -1)
        return x
    

    # def Sideinfo(self, demography):
    #     demo_emb = self.demo_linear(demography)
    #     return demo_emb
    
    # def CatSpatialConv(self, cat_corr):
    #     one_hot_matrix = torch.eye(self.input_feat).repeat(cat_corr.shape[0], 1, 1).to(cat_corr.device)
    #     for gcn in self.cat_gcns:
    #         cat_corr = gcn(one_hot_matrix, cat_corr)
    #     return cat_corr
    
    def apply_filter(self, vectors, filter_mask):
        if self.filter_mode == 'separate' and np.sum(filter_mask) != 0:
            filter_mask = np.asarray(filter_mask)
            idx = filter_mask.dot(2**np.arange(filter_mask.size))
            sens_filter = self.filter_dict[str(idx)]
            result = sens_filter(vectors)
        elif self.filter_mode == 'combine' and np.sum(filter_mask) != 0:
            result = None
            for idx, val in enumerate(filter_mask):
                if not val: continue
                sens_filter = self.filter_dict[str(idx + 1)]
                result = sens_filter(vectors) if result is None else result + sens_filter(vectors)
            result = result / np.sum(filter_mask)   # average the embedding
        else:
            result = vectors
        return result
    
    def freeze_model(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
 
    def unfreeze_model(self):
        self.train()
        for param in self.parameters():
            param.requires_grad = True
    
    def load_model(self, model_path=None):
        model_path = model_path or self.model_path
        self.load_state_dict(torch.load(model_path))
        logging.info('Load model from ' + model_path)

    def save_model(self, model_path=None):
        model_path = model_path or self.model_path
        dir_path = os.path.dirname(model_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        torch.save(self.state_dict(), model_path)
        logging.info('Save model to ' + model_path)
