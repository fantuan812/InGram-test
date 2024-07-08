from scipy.sparse import csr_matrix
from tqdm import tqdm
from utils import *
import numpy as np
import math
import igraph
import random
import torch
# torch.manual_seed(0)
# random.seed(0)
# np.random.seed(0)
def create_relation_graph(triplet, num_ent, num_rel):
	# triplet 是一个包含三元组（实体头、关系、实体尾）的N x 3数组
    # num_ent 是实体的总数量
    # num_rel 是关系的总数量
    
    # 从三元组数据中分别提取头实体和关系，以及关系和尾实体的组合
	ind_h = triplet[:, :2]  # 头实体和关系索引
	ind_t = triplet[:, 1:]  # 关系和尾实体索引
	
	# 创建头实体关系矩阵 E_h
    # 这里使用稀疏矩阵来有效存储数据，避免大规模矩阵的内存开销
    # 矩阵形状为 (num_ent, 2 * num_rel)，考虑到可能存在反向关系
	# print("预期的形状：", (num_ent, 2 * num_rel))
	# print("ind_h[:, 0] 的最大索引值：", np.max(ind_h[:, 0]))
	# print("ind_h[:, 1] 的最大索引值：", np.max(ind_h[:, 1]))
	E_h = csr_matrix((np.ones(len(ind_h)), (ind_h[:, 0], ind_h[:, 1])), shape=(num_ent, 2 * num_rel))
	# print("E_h 的类型：", type(E_h))
	# print("E_h 的形状：", E_h.shape)
	# print("E_h 的非零元素数：", E_h.nnz)
	# print("E_h 的非零元素：", E_h)
	# 创建尾实体关系矩阵 E_t
    # 注意：在创建E_t时，行和列索引的顺序与E_h相反，表示关系指向尾实体
	E_t = csr_matrix((np.ones(len(ind_t)), (ind_t[:, 1], ind_t[:, 0])), shape=(num_ent, 2 * num_rel))
	# 计算头实体关系矩阵 E_h 的每行的和，用于后续的度矩阵计算
	diag_vals_h = E_h.sum(axis=1).A1
	# 对行和进行平方的倒数计算，确保不对0进行除法
	diag_vals_h[diag_vals_h!=0] = 1/(diag_vals_h[diag_vals_h!=0]**2)
	# 计算尾实体关系矩阵 E_t 的每行的和
	diag_vals_t = E_t.sum(axis=1).A1
	diag_vals_t[diag_vals_t!=0] = 1/(diag_vals_t[diag_vals_t!=0]**2)

	# 创建头实体和尾实体的度矩阵的逆矩阵D_h_inv和D_t_inv
	D_h_inv = csr_matrix((diag_vals_h, (np.arange(num_ent), np.arange(num_ent))), shape=(num_ent, num_ent))
	D_t_inv = csr_matrix((diag_vals_t, (np.arange(num_ent), np.arange(num_ent))), shape=(num_ent, num_ent))

	# 计算邻接矩阵A_h和A_t
    # 使用矩阵乘法计算邻接矩阵
	A_h = E_h.transpose() @ D_h_inv @ E_h # 表示头实体到尾实体的关系强度
	A_t = E_t.transpose() @ D_t_inv @ E_t
	# 返回两个邻接矩阵的总和，这个总和矩阵可以用于后续的图分析或作为图神经网络的输入
	return A_h + A_t

def get_relation_triplets(G_rel, B):
	# G_rel是igraph图对象，B是分桶数量，用于将权重转换为类别标签
	rel_triplets = []
	# 遍历图中的所有边，提取边的信息
	for tup in G_rel.get_edgelist():
		h,t = tup
		# 获取边的ID
		tupid = G_rel.get_eid(h,t)
		# 获取边的权重
		w = G_rel.es[tupid]["weight"]
		# 保存边的信息为三元组（头节点，尾节点，权重）
		rel_triplets.append((int(h), int(t), float(w)))
	# 将列表转换为numpy数组
	rel_triplets = np.array(rel_triplets)
	# 获取三元组的数量
	nnz = len(rel_triplets)
	# 按照权重的负值进行排序，这样可以得到从大到小的索引顺序
	temp = (-rel_triplets[:,2]).argsort()
	# 创建一个空数组用于存储权重的排名
	weight_ranks = np.empty_like(temp)
	# 给权重排序，权重大的排名靠前
	weight_ranks[temp] = np.arange(nnz) + 1
	# 将权重转换为排名，再转换为分类标签
	relation_triplets = []
	for idx,triplet in enumerate(rel_triplets):
		h,t,w = triplet
		# 计算每个三元组的排名，并根据总数B转换为类别
        # math.ceil用于向上取整，保证至少为1
        # 减1是因为类别通常从0开始计数
		rk = int(math.ceil(weight_ranks[idx]/nnz*B))-1
		# 保存处理后的三元组
		relation_triplets.append([int(h), int(t), rk])
		# 确保计算的类别在有效范围内
		assert rk >= 0
		assert rk < B
	# 将处理后的三元组列表转换为numpy数组并返回
	return np.array(relation_triplets)

def generate_relation_triplets(triplet, num_ent, num_rel, B):
	# 生成关系图的邻接矩阵
	A = create_relation_graph(triplet, num_ent, num_rel)
	# 使用邻接矩阵创建加权图
	G_rel = igraph.Graph.Weighted_Adjacency(A)
	# 提取并处理三元组
	relation_triplets = get_relation_triplets(G_rel, B)
	return relation_triplets