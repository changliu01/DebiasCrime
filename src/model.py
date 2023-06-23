import torch
import torch.nn as nn

class Descriminator(nn.Module):
    def __init__(self, embed_dim, class_num, neg_slope, dropout):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * 2), bias=True),
            nn.LeakyReLU(neg_slope),
            nn.Dropout(p=dropout),
            nn.Linear(int(embed_dim * 2), int(embed_dim * 4), bias=True),
            nn.LeakyReLU(neg_slope),
            nn.Dropout(p=dropout),
            nn.Linear(int(embed_dim * 4), int(embed_dim * 2), bias=True),
            nn.LeakyReLU(neg_slope),
            nn.Dropout(p=dropout),
            nn.Linear(int(embed_dim * 2), int(embed_dim * 2), bias=True),
            nn.LeakyReLU(neg_slope),
            nn.Dropout(p=dropout),
            nn.Linear(int(embed_dim * 2), int(embed_dim), bias=True),
            nn.LeakyReLU(neg_slope),
            nn.Dropout(p=dropout),
            nn.Linear(int(embed_dim), int(embed_dim / 2), bias=True),
            nn.LeakyReLU(neg_slope),
            nn.Dropout(p=dropout),
            nn.Linear(int(embed_dim / 2), class_num, bias=True)
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, label):
        """
        - x: (batch_size, retion_num, crime_feature_num)
        - label: (batch_size, region_num)
        - return: 
            - output: (batch_size, region_num, demo_class_num)
            - pred: (batch_size, region_num)
        """
        output = self.network(x)
        pred = output.argmax(dim=-1)
        loss = self.criterion(output.permute(0, 2, 1), label)
        result_dict = dict(
            output=output,
            pred=pred,
            loss=loss
        )
        return result_dict
