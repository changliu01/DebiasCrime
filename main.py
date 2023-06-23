import pandas as pd
import numpy as np
import torch
import random, sys, os, time, datetime, argparse
from setproctitle import setproctitle
if 'ipython' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm
from src_dev.utils import run_epoch, Dataset
from src_dev.model import Descriminator
from src_dev.ST_GCN import ST_GCN
from src_dev.STCGNN import STCGNN
from src_dev.Linear import Linear

parser = argparse.ArgumentParser(description='Model Params')
parser.add_argument('--model', type=str, default='Linear', choices=['ST-GCN', 'STCGNN', 'Linear'])
parser.add_argument('--num_workers', type=int, default=0 if 'win' in sys.platform else 4)
parser.add_argument('--data_path', type=str, default='./dataset/CHI.npz')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--window_size', type=int, default=7)
parser.add_argument('--input_feat', type=int, default=4)
parser.add_argument('--embedding_dim', type=int, default=4)
parser.add_argument('--output_feat', type=int, default=1)
parser.add_argument('--dim_val', type=int, default=1)
parser.add_argument('--gcn_emb_size', type=int, default=28)
parser.add_argument('--demo_feat_size', type=int, default=10)
parser.add_argument('--demo_hidden', type=int, default=32)
parser.add_argument('--gcn_layer_num', type=int, default=3)
parser.add_argument('--region_num', type=int, default=300)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--trans_layers', type=int, default=1)
parser.add_argument('--cat_gcn_layer_num', type=int, default=3)
parser.add_argument('--wo_demo', type=bool, default=False)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--neg_slope', type=float, default=0.2)
parser.add_argument('--model_path', type=str, default='./model_ckpt/base_model.pt')
parser.add_argument('--l2_weight', type=float, default=1e-5)
parser.add_argument('--lr_attack', type=float, default=1e-3)
parser.add_argument('--l2_attack', type=float, default=1e-5)
parser.add_argument('--print_metrics', type=str, default='rmse')
parser.add_argument('--filter_mode', type=str, default='combine')
parser.add_argument('--load', type=int, default=0)
parser.add_argument('--d_steps', type=int, default=20)
parser.add_argument('--check_epoch', type=int, default=1)
parser.add_argument('--reg_weight',type=float, default=1)
parser.add_argument('--attacker_epoch', type=int, default=10)
parser.add_argument('--no_adversary', action='store_true', help='Whether to use adversary')
parser.add_argument('--fix_one', action='store_true', help='fix one feature for evaluation.')
args = parser.parse_args()

if __name__ == '__main__':
    start_time = time.time()
    setproctitle('MDMProject')

    # set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # load data
    dataset = Dataset(args.data_path, args.window_size, 1)
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.75, 0.15, 0.1])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # load model
    if args.model == 'ST-GCN':
        model = ST_GCN(
            args, 
            dataset.region_num, 
            dataset
        ).to(args.device)
        model.apply(model.init_weights)
    elif args.model == 'STCGNN':
        model = STCGNN(
            num_nodes      = dataset.region_num,
            num_categories = dataset.crime_feature_num,
            Ks             = 2,
            Kc             = 2,
            input_dim      = 1,
            hidden_dim     = 16,
            num_layers     = 2,
            out_horizon    = 1,
            feature_dim    = dataset.demo_feat_num,
            embedding_dim  = args.embedding_dim
        ).to(args.device)
    elif args.model == 'Linear':
        model = Linear(
            time_step        = args.window_size,
            region_num       = dataset.region_num,
            crime_feat_size  = dataset.crime_feature_num,
            demo_feat_size   = dataset.demo_feat_num,
            embedding_dim    = args.embedding_dim
        ).to(args.device)
    else:
        raise ValueError('Invalid model name.')
    descriminator = Descriminator(args.embedding_dim, dataset.demo_class_num, args.neg_slope, args.dropout)
    model.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    descriminator.optimizer = torch.optim.Adam(descriminator.parameters(), lr=args.lr_attack, weight_decay=args.weight_decay)
    print('#Parameter =', sum([p.numel() for p in model.parameters() if p.requires_grad]))
    
    # train
    for epoch in range(args.epoch):
        print(f'--- Epoch {epoch} ---')
        train_result = run_epoch(args, 'train', train_loader, model, descriminator, enable_tqdm=False)
        print('\t'.join([f'{k}:{v:.4f}' for k, v in train_result.items()]))
        valid_result = run_epoch(args, 'valid', valid_loader, model, descriminator, enable_tqdm=False)
        print('\t'.join([f'{k}:{v:.4f}' for k, v in valid_result.items()]))
    print(f'--- Test ---')
    test_result = run_epoch(args, 'test', test_loader, model, descriminator, enable_tqdm=False)
    print('\t'.join([f'{k}:{v:.4f}' for k, v in test_result.items()]))
