import torch
import random
import numpy as np
from relgraph import generate_relation_triplets

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def initialize(target, msg, d_e, d_r, B):
    # 初始化实体嵌入矩阵：大小为实体数 * 实体嵌入维度，填充为0，并移至GPU
    init_emb_ent = torch.zeros((target.num_ent, d_e)).cuda()
    # 初始化关系嵌入矩阵：大小为关系数的两倍 * 关系嵌入维度，填充为0，并移至GPU
    # 关系数乘以2是为了同时考虑关系的逆关系
    init_emb_rel = torch.zeros((2 * target.num_rel, d_r)).cuda()
    
    # init_emb_ent=torch.nn.Parameter(torch.Tensor(target.num_ent, d_e))
    # init_emb_rel=torch.nn.Parameter(torch.Tensor(2 * target.num_rel, d_r))
    # 计算ReLU激活函数的增益因子，用于初始化
    gain = torch.nn.init.calculate_gain('relu')
    
    # 使用Xavier正态分布初始化方法初始化实体嵌入矩阵
    torch.nn.init.xavier_normal_(init_emb_ent, gain=gain)
    # torch.nn.init.xavier_normal_(init_emb_ent)
    # 使用Xavier正态分布初始化方法初始化关系嵌入矩阵
    torch.nn.init.xavier_normal_(init_emb_rel, gain=gain)
    # torch.nn.init.xavier_normal_(init_emb_rel)
    # 生成关系三元组，这个函数可能基于msg来生成用于训练的额外的关系三元组
    relation_triplets = generate_relation_triplets(msg, target.num_ent, target.num_rel, B)
    # 将生成的关系三元组列表转换为张量并移至GPU
    relation_triplets = torch.tensor(relation_triplets).cuda()
    
    # 返回初始化的实体嵌入矩阵、关系嵌入矩阵和关系三元组
    return init_emb_ent, init_emb_rel, relation_triplets
