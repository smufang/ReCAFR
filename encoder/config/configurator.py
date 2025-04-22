import os
import yaml
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn

def parse_configure(model=None, dataset=None):
    parser = argparse.ArgumentParser(description='RLMRec')
    parser.add_argument('--model', type=str, default='lightgcn_plus', help='Model name')
    parser.add_argument('--dataset', type=str, default='amazon', help='Dataset name')
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    parser.add_argument('--seed', type=int, default=None, help='Device number')
    parser.add_argument('--cuda', type=str, default='0', help='Device number')
    args, _ = parser.parse_known_args()

    # cuda
    if args.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    # model name
    if model is not None:
        model_name = model.lower()
    elif args.model is not None:
        model_name = args.model.lower()
    else:
        model_name = 'default'

    # dataset
    if dataset is not None:
        args.dataset = dataset

    with open('./encoder/config/modelconf/lightgcn_plus.yml') as f:
        config_data = f.read()
        configs = yaml.safe_load(config_data)
        configs['model']['name'] = configs['model']['name'].lower()
        configs['tune'] = {'enable': False}
        configs['device'] = args.device
        if args.dataset is not None:
            configs['data']['name'] = args.dataset
        if args.seed is not None:
            configs['train']['seed'] = args.seed

        # text embeddings
        usrprf_embeds_path = "./data/amazon/usr_emb_np.pkl"
        itmprf_embeds_path = "./data/amazon/itm_emb_np.pkl"
        with open(usrprf_embeds_path, 'rb') as f:
            configs['usrprf_embeds'] = pickle.load(f)
        with open(itmprf_embeds_path, 'rb') as f:
            configs['itmprf_embeds'] = pickle.load(f)

        return configs
configs = parse_configure()
