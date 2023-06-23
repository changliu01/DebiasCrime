import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, time_step:int, region_num:int, crime_feat_size:int, demo_feat_size:int, embedding_dim:int):
        super().__init__()
        self.encoder = nn.Linear(time_step * crime_feat_size + demo_feat_size, embedding_dim)
        self.filter = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 1), 
            nn.Sigmoid()
        )
        self.criterion = nn.BCELoss()
    
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
        input = torch.concat([x.permute(0, 2, 1, 3).flatten(-2, -1), demography], dim=-1)
        ori_vector = self.encoder(input) # [batch_size, region_num, embed_dim]
        filter_vector = self.filter(ori_vector)
        prediction = self.decoder(filter_vector).squeeze(dim=-1)
        loss = self.criterion(prediction, label)
        return dict(
            prediction=prediction,
            filter_vectors=filter_vector,
            ori_vectors=ori_vector,
            loss=loss
        )

