import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
# torch.manual_seed(0)


class InGramEntityLayer(nn.Module):
    def __init__(self, dim_in_ent, dim_out_ent, dim_rel, bias = True, num_head = 8):
        super(InGramEntityLayer, self).__init__()
        # 初始化各种维度和参数
        self.dim_out_ent = dim_out_ent  # 实体输出维度
        self.dim_hid_ent = dim_out_ent // num_head  # 每个头的隐藏维度
        assert dim_out_ent == self.dim_hid_ent * num_head  # 确保输出维度可以均匀分配到各头
        self.num_head = num_head  # 多头注意力的头数

        # 定义注意力投影层
        self.attn_proj = nn.Linear(2 * dim_in_ent + dim_rel, dim_out_ent, bias=bias)
        # 注意力向量参数
        self.attn_vec = nn.Parameter(torch.zeros((1, num_head, self.dim_hid_ent)))
        # 聚合投影层
        self.aggr_proj = nn.Linear(dim_in_ent + dim_rel, dim_out_ent, bias=bias)

        self.dim_rel = dim_rel  # 关系的维度
        # self.act = nn.LeakyReLU(negative_slope=0.2)  # 激活函数
        self.act =torch.tanh
        self.bias = bias  # 是否使用偏置
        self.param_init()  # 参数初始化

    
    def param_init(self):
        # 参数初始化方法，使用Xavier初始化
        nn.init.xavier_normal_(self.attn_proj.weight)
        nn.init.xavier_normal_(self.attn_vec)
        nn.init.xavier_normal_(self.aggr_proj.weight)
        if self.bias:
            nn.init.zeros_(self.attn_proj.bias)
            nn.init.zeros_(self.aggr_proj.bias)
    
    def forward(self, emb_ent, emb_rel, triplets): 
        # 前向传播定义
        num_ent = len(emb_ent)  # 实体的数量
        num_rel = len(emb_rel)  # 关系的数量
        head_idxs = triplets[..., 0]  # 头实体索引
        rel_idxs = triplets[..., 1]  # 关系索引
        tail_idxs = triplets[..., 2]  # 尾实体索引
        # 计算每个尾实体出现的频次
        ent_freq = torch.zeros((num_ent,)).cuda().index_add(dim=0, index=tail_idxs, \
                                                            source=torch.ones_like(tail_idxs, dtype=torch.float).cuda()).unsqueeze(dim=1)
        if torch.any(ent_freq == 0):
            # print("Warning: ent_freq contains zeros.")
            ent_freq = ent_freq + 1e-16  # 添加一个小值以避免除零错误
        # print('ent_freq',ent_freq)
        # 计算实体的自环关系
        self_rel = torch.zeros((num_ent, self.dim_rel)).cuda().index_add(dim=0, index=tail_idxs, source=emb_rel[rel_idxs]) / ent_freq
        # print('self_rel',self_rel)
        # add self-loops
        # 添加自环
        emb_rels = torch.cat([emb_rel[rel_idxs], self_rel], dim = 0)
        head_idxs = torch.cat([head_idxs, torch.arange(num_ent).cuda()], dim = 0)
        tail_idxs = torch.cat([tail_idxs, torch.arange(num_ent).cuda()], dim = 0)
        # print('emb_rels',emb_rels)
        # print('head_idxs',head_idxs)
        # print('tail_idxs',tail_idxs)
        # 计算注意力值的输入矩阵
        concat_mat_att = torch.cat([emb_ent[tail_idxs], emb_ent[head_idxs], \
                                    emb_rels], dim = -1)
        # print('concat_mat_att',concat_mat_att)
        attn_val_raw = (self.act(self.attn_proj(concat_mat_att).view(-1, self.num_head, self.dim_hid_ent)) * 
                       self.attn_vec).sum(dim = -1, keepdim = True)
        # print('attn_val_raw',attn_val_raw)
        # 散列索引准备用于scatter操作
        scatter_idx = tail_idxs.unsqueeze(dim = -1).repeat(1, self.num_head).unsqueeze(dim = -1)
        # print('scatter_idx',scatter_idx)
        # 计算最大注意力值，用于归一化
        attn_val_max = torch.zeros((num_ent, self.num_head, 1)).cuda().scatter_reduce(dim = 0, \
                                                                    index = scatter_idx, \
                                                                    src = attn_val_raw, reduce = 'amax', \
                                                                    include_self = False)
        # print('attn_val_max',attn_val_max)
        attn_val = torch.exp(attn_val_raw - attn_val_max[tail_idxs])
        # print('attn_val',attn_val)
        # 计算注意力和
        attn_sums = torch.zeros((num_ent, self.num_head, 1)).cuda().index_add(dim = 0, index = tail_idxs, source = attn_val)
        # print('attn_sums',attn_sums)
        # 计算归一化的注意力权重
        beta = attn_val / (attn_sums[tail_idxs]+1e-16)
        # print('beta',beta)
        # 计算聚合值
        concat_mat = torch.cat([emb_ent[head_idxs], emb_rels], dim = -1)
        # print('concat_mat',concat_mat)
        aggr_val = beta * self.aggr_proj(concat_mat).view(-1, self.num_head, self.dim_hid_ent)
        # print('aggr_val',aggr_val)
        # 聚合输出     
        output = torch.zeros((num_ent, self.num_head, self.dim_hid_ent)).cuda().index_add(dim = 0, index = tail_idxs, source = aggr_val)
        # print('output',output)
        output = output.flatten(1,-1)  # Add batch normalization
        # print('output',output)
        # print(output.shape)
        # output=self.bn1(output)
        return output # 将多头输出拼接成单个向量

class InGramRelationLayer(nn.Module):
    def __init__(self, dim_in_rel, dim_out_rel, num_bin, bias = True, num_head = 8):
        super(InGramRelationLayer, self).__init__()
         # 初始化各种维度和参数
        self.dim_out_rel = dim_out_rel # 输出维度
        self.dim_hid_rel = dim_out_rel // num_head # 每个头的隐藏维度
        assert dim_out_rel == self.dim_hid_rel * num_head # 确保输出维度可以均匀分配到各头
        # 定义注意力投影层
        self.attn_proj = nn.Linear(2*dim_in_rel, dim_out_rel, bias = bias)
        # 注意力bins参数
        self.attn_bin = nn.Parameter(torch.zeros(num_bin, num_head, 1))
        # 注意力向量参数
        self.attn_vec = nn.Parameter(torch.zeros(1, num_head, self.dim_hid_rel))
        # 聚合投影层
        self.aggr_proj = nn.Linear(dim_in_rel, dim_out_rel, bias = bias)
        self.num_head = num_head # 多头注意力的头数
        #激活函数
        # self.act = nn.LeakyReLU(negative_slope = 0.2) # 激活函数
        self.act =torch.tanh
        self.num_bin = num_bin # bins的数量
        self.bias = bias  # 是否使用偏置

        self.param_init()  # 参数初始化
    
    def param_init(self):
        # 参数初始化方法，使用Xavier初始化
        nn.init.xavier_normal_(self.attn_proj.weight)
        nn.init.xavier_normal_(self.attn_vec)
        nn.init.xavier_normal_(self.aggr_proj.weight)
        if self.bias:
            nn.init.zeros_(self.attn_proj.bias)
            nn.init.zeros_(self.aggr_proj.bias)
    
    def forward(self, emb_rel, relation_triplets):
        # 前向传播定义
        num_rel = len(emb_rel) # 获取关系实体的数量
        # 从三元组中获取头尾实体的索引
        head_idxs = relation_triplets[..., 0]
        tail_idxs = relation_triplets[..., 1]
        # 拼接头尾实体的嵌入
        concat_mat = torch.cat([emb_rel[head_idxs], emb_rel[tail_idxs]], dim = -1)
        # print('concat_mat',concat_mat)
        # 计算注意力值
        attn_val_raw = (self.act(self.attn_proj(concat_mat).view(-1, self.num_head, self.dim_hid_rel)) * \
                        self.attn_vec).sum(dim = -1, keepdim = True) 
        # print('attn_val_raw',attn_val_raw)
        # 散列索引准备用于scatter操作
        scatter_idx = head_idxs.unsqueeze(dim = -1).repeat(1, self.num_head).unsqueeze(dim = -1)
        # print('scatter_idx',scatter_idx)
        # 计算最大注意力值，用于归一化
        attn_val_max = torch.zeros((num_rel, self.num_head, 1)).cuda().scatter_reduce(dim = 0, \
                                                                    index = scatter_idx, \
                                                                    src = attn_val_raw, reduce = 'amax', \
                                                                    include_self = False)
        # print('attn_val_max',attn_val_max)
        attn_val = torch.exp(attn_val_raw - attn_val_max[head_idxs])
        # print('attn_val',attn_val)
        # 计算注意力和
        attn_sums = torch.zeros((num_rel, self.num_head, 1)).cuda().index_add(dim = 0, index = head_idxs, source = attn_val)
        # print('attn_sums',attn_sums)
        # 计算归一化的注意力权重
        beta = attn_val / (attn_sums[head_idxs]+1e-16)
        # print('beta',beta)
        # 计算输出
        output = torch.zeros((num_rel, self.num_head, self.dim_hid_rel)).cuda().index_add(dim = 0, \
                                                                                            index = head_idxs, 
                                                                                            source = beta * self.aggr_proj(emb_rel[tail_idxs]).view(-1, self.num_head, self.dim_hid_rel))
        # print('output',output)
        # 将多头输出拼接成单个向量
        output=output.flatten(1,-1)
        # print('output',output)
        # print(output.shape)
        # output = self.bn1(output)  # Add batch normalization
        return output

class InGram(nn.Module):
    def __init__(self, dim_ent, hid_dim_ratio_ent, dim_rel, hid_dim_ratio_rel, num_bin,num_ent,num_rel, num_layer_ent=2, num_layer_rel=2, \
                 num_head = 8, bias = True ):
        super(InGram, self).__init__()
        self.gamma = nn.Parameter(
            torch.Tensor([10.0]),
            requires_grad=False
        )
        self.init_emb_ent=torch.nn.Parameter(torch.Tensor(num_ent, dim_ent))
        self.init_emb_rel=torch.nn.Parameter(torch.Tensor(2 * num_rel, dim_rel))
         # 初始化用于存放实体层和关系层的列表
        layers_ent = []
        layers_rel = []
        # 根据输入维度和指定的比例计算实体层和关系层的维度
        layer_dim_ent = hid_dim_ratio_ent * dim_ent
        layer_dim_rel = hid_dim_ratio_rel * dim_rel
        # 根据指定维度和参数创建实体层
        for _ in range(num_layer_ent):
            layers_ent.append(InGramEntityLayer(dim_ent, dim_ent, dim_rel, \
                                                bias = bias, num_head = num_head))
        # 根据指定维度和参数创建关系层
        for _ in range(num_layer_rel):
            layers_rel.append(InGramRelationLayer(dim_rel, dim_rel, num_bin, \
                                                  bias = bias, num_head = num_head))
        # 初始化实体和关系的残差连接的线性变换层
        res_proj_ent = []
        for _ in range(num_layer_ent):
            res_proj_ent.append(nn.Linear(dim_ent, dim_ent, bias = bias))
        
        res_proj_rel = []
        for _ in range(num_layer_rel):
            res_proj_rel.append(nn.Linear(dim_rel, dim_rel, bias = bias))
        
        # 将所有层和投影模块存储在ModuleLists中，以便正确跟踪参数
        self.res_proj_ent = nn.ModuleList(res_proj_ent)
        self.res_proj_rel = nn.ModuleList(res_proj_rel)
        self.bias = bias
        self.ent_proj1 = nn.Linear(dim_ent, layer_dim_ent, bias = bias)
        self.ent_proj2 = nn.Linear(layer_dim_ent, dim_ent, bias = bias)
        self.layers_ent = nn.ModuleList(layers_ent)
        self.layers_rel = nn.ModuleList(layers_rel)

        self.rel_proj1 = nn.Linear(dim_rel, layer_dim_rel, bias = bias)
        self.rel_proj2 = nn.Linear(layer_dim_rel, dim_rel, bias = bias)
        # 用于评分函数中的关系嵌入的额外投影
        self.rel_proj = nn.Linear(dim_rel, dim_ent, bias = bias)
        # 存储实体和关系层的数量
        self.num_layer_ent = num_layer_ent
        self.num_layer_rel = num_layer_rel
        # 每层之后应用的激活函数
        # self.act = nn.ReLU()
        self.act =torch.tanh
        # 初始化所有权重和偏置
        self.param_init()

        self.drop = torch.nn.Dropout(0.5)


    
    def param_init(self):
        # 使用Xavier正态初始化进行权重初始化，以获得更好的收敛性
        nn.init.xavier_normal_(self.ent_proj1.weight)
        nn.init.xavier_normal_(self.ent_proj2.weight)
        nn.init.xavier_normal_(self.rel_proj1.weight)
        nn.init.xavier_normal_(self.rel_proj2.weight)
        nn.init.xavier_normal_(self.rel_proj.weight)
        # nn.init.xavier_normal_(self.init_emb_ent.data, gain = nn.init.calculate_gain('relu'))
        # nn.init.xavier_normal_(self.init_emb_rel.data, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.init_emb_ent.data)
        nn.init.xavier_normal_(self.init_emb_rel.data)
        # 对残差投影层的权重应用Xavier初始化
        for layer_idx in range(self.num_layer_ent):
            nn.init.xavier_normal_(self.res_proj_ent[layer_idx].weight, gain = nn.init.calculate_gain('relu'))
        for layer_idx in range(self.num_layer_rel):
            nn.init.xavier_normal_(self.res_proj_rel[layer_idx].weight, gain = nn.init.calculate_gain('relu'))
        # 如果启用偏置，则将所有偏置初始化为零
        if self.bias:
            nn.init.zeros_(self.ent_proj1.bias)
            nn.init.zeros_(self.ent_proj2.bias)
            nn.init.zeros_(self.rel_proj1.bias)
            nn.init.zeros_(self.rel_proj2.bias)
            nn.init.zeros_(self.rel_proj.bias)
            for layer_idx in range(self.num_layer_ent):
                nn.init.zeros_(self.res_proj_ent[layer_idx].bias)
            for layer_idx in range(self.num_layer_rel):
                nn.init.zeros_(self.res_proj_rel[layer_idx].bias)
            

    def forward(self,emb_ent,emb_rel, triplets, relation_triplets):
        # 通过初始投影层传递输入嵌入
        # layer_emb_ent = self.ent_proj1(self.init_emb_ent)
        # layer_emb_rel = self.rel_proj1(self.init_emb_rel)
        layer_emb_ent = self.init_emb_ent
        layer_emb_rel = self.init_emb_rel

        # layer_emb_ent = emb_ent
        # layer_emb_rel = emb_rel
        # layer_emb_ent = self.ent_proj1(emb_ent)
        # layer_emb_rel = self.rel_proj1(emb_rel)
        # 处理关系嵌入通过关系层，包括跳跃连接和激活
        for layer_idx, layer in enumerate(self.layers_rel):
            layer_emb_rel = layer(layer_emb_rel, relation_triplets) + \
                            self.res_proj_rel[layer_idx](layer_emb_rel)
            layer_emb_rel = self.act(layer_emb_rel)
        # layer_emb_rel = self.drop(layer_emb_rel)
        # 处理实体嵌入通过实体层，包括跳跃连接和激活
        for layer_idx, layer in enumerate(self.layers_ent):
            layer_emb_ent = layer(layer_emb_ent, layer_emb_rel, triplets) + \
                            self.res_proj_ent[layer_idx](layer_emb_ent)
            layer_emb_ent = self.act(layer_emb_ent)
        # layer_emb_ent = self.drop(layer_emb_ent)

        # 返回最终变换的实体和关系嵌入
        # layer_emb_ent=self.ent_proj2(layer_emb_ent)
        # layer_emb_rel=self.rel_proj2(layer_emb_rel)
        return layer_emb_ent, layer_emb_rel
    
    def score(self, emb_ent, emb_rel, triplets):
        # 根据三元组中的索引提取嵌入用于评分
        head_idxs = triplets[..., 0]
        rel_idxs = triplets[..., 1]
        tail_idxs = triplets[..., 2]
        head_embs = emb_ent[head_idxs]
        tail_embs = emb_ent[tail_idxs]
        rel_embs = self.rel_proj(emb_rel[rel_idxs])
        # TransE
        # rel_embs = emb_rel[rel_idxs]
        # score = (head_embs + rel_embs) - tail_embs
        # score = self.gamma.item() - torch.norm(score, p=1, dim=-1)
        # Dismult
        score = (head_embs * rel_embs * tail_embs).sum(dim = -1)
        # ComplEx
        # re_head, im_head = torch.chunk(head_embs, 2, dim=-1)
        # re_relation, im_relation = torch.chunk(rel_embs, 2, dim=-1)
        # re_tail, im_tail = torch.chunk(tail_embs, 2, dim=-1)
        # re_score = re_head * re_relation - im_head * im_relation
        # im_score = re_head * im_relation + im_head * re_relation
        # score = re_score * re_tail + im_score * im_tail
        # score = score.sum(dim = -1)
        # RotatE
        # pi = 3.14159265358979323846
        # re_head, im_head = torch.chunk(head_embs, 2, dim=-1)
        # re_tail, im_tail = torch.chunk(tail_embs, 2, dim=-1)
        # #Make phases of relations uniformly distributed in [-pi, pi]
        # phase_relation = rel_embs/(self.embedding_range.item()/pi)
        # re_relation = torch.cos(phase_relation)
        # im_relation = torch.sin(phase_relation)
        # re_score = re_head * re_relation - im_head * im_relation
        # im_score = re_head * im_relation + im_head * re_relation
        # re_score = re_score - re_tail
        # im_score = im_score - im_tail
        # score = torch.stack([re_score, im_score], dim = 0)
        # score = score.norm(dim = 0)
        # score = self.gamma.item() - score.sum(dim = 1)

        return score



