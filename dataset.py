from utils import *
import numpy as np
import random
import igraph
import copy
import time
import os
from random import sample


class TrainData():
	def __init__(self, path):
		self.path = path # 数据文件的路径
		self.rel_info = {}  # 存储每对实体之间的关系
		self.pair_info = {} # 存储每个关系对应的实体对
		self.spanning = [] # 最小生成树中的边
		self.remaining = [] # 不在最小生成树中的边
		# 初始化实体和关系的索引，并读取三元组
		self.ent2id = None
		self.rel2id = None
		self.triplets = self.read_triplet(path + 'train.txt')
		# 计算三元组、实体和关系的数量
		self.num_triplets = len(self.triplets) # 三元组的总数
		self.num_ent, self.num_rel = len(self.ent2id), len(self.rel2id) # 实体和关系的数量
		# self.sr2o = ddict(set)
	# 从文件中读取三元组数据
	def read_triplet(self, path):
		id2ent1,id2rel1,id2ent, id2rel,triplets= [], [], [], [],[]
		# print(self.path.replace('train.txt','')+ "entities.dict")
		self.ent2id,self.rel2id=self.loads(self.path)
		# print(self.ent2id)
		# num_lines_to_read = 80000	
		with open(path, 'r') as f:
			# lines = f.readlines()
			# total_lines = len(lines)
			# # 确保要读取的行数不超过文件的总行数
			# num_lines_to_read = min(num_lines_to_read, total_lines)
			# # 使用 `random.sample` 方法随机选择行的索引
			# random_indices = random.sample(range(total_lines), num_lines_to_read)
			# # 根据这些索引从原始行列表中提取随机行
			# random_lines = [lines[i] for i in random_indices]
			for line in f.readlines():
				h, r, t = line.strip().split('\t')
				# id2ent.append(h)
				# id2ent.append(t)
				# id2rel.append(r)
				triplets.append((h, r, t))
		# 移除重复的实体和关系，创建索引映射
		# id2ent = remove_duplicate(id2ent)
		# id2rel = remove_duplicate(id2rel)
		# self.ent2id = {ent: idx for idx, ent in enumerate(id2ent)}
		# self.rel2id = {rel: idx for idx, rel in enumerate(id2rel)}
		# 将三元组中的实体和关系名称转换为索引
		triplets = [(self.ent2id[h], self.rel2id[r], self.ent2id[t]) for h, r, t in triplets]
		# with open(path, 'r') as f:
		# 	for line in f.readlines():
		# 		h, r, t = line.strip().split('\t')
		# 		h, r, t = self.ent2id[h], self.rel2id[r], self.ent2id[t]
				# self.sr2o[(h, r)].add(t)
				# self.sr2o[(t, r+len(self.rel2id))].add(h)
		# 建立关系信息和实体对信息
		for (h,r,t) in triplets:
			# 存储关系到关系信息字典
			if (h,t) in self.rel_info:
				self.rel_info[(h,t)].append(r)
			else:
				self.rel_info[(h,t)] = [r]
			# 存储实体对到实体对信息字典
			if r in self.pair_info:
				self.pair_info[r].append((h,t))
			else:
				self.pair_info[r] = [(h,t)]
		# 使用igraph创建图，并计算最小生成树
		G = igraph.Graph.TupleList(np.array(triplets)[:, 0::2])
		G_ent = igraph.Graph.TupleList(np.array(triplets)[:, 0::2], directed = True)
		# 计算最小生成树
		spanning = G_ent.spanning_tree()
		# 删除最小生成树的边，用于生成图
		G_ent.delete_edges(spanning.get_edgelist())
		# 存储最小生成树的边
		for e in spanning.es:
			e1,e2 = e.tuple
			e1 = spanning.vs[e1]["name"]
			e2 = spanning.vs[e2]["name"]
			self.spanning.append((e1,e2))
		
		spanning_set = set(self.spanning)


		
		print("-----Train Data Statistics-----")
		print(f"{len(self.ent2id)} entities, {len(self.rel2id)} relations")
		print(f"{len(triplets)} triplets")
		self.triplet2idx = {triplet:idx for idx, triplet in enumerate(triplets)}
		# 添加逆三元组，扩展训练数据
		self.triplets_with_inv = np.array([(t, r + len(id2rel), h) for h,r,t in triplets] + triplets)
		return triplets
	# 分割数据为训练和验证用
	def split_transductive(self, p):
		msg, sup = [], []

		# rels_encountered = np.zeros(self.num_rel)
		# remaining_triplet_indexes = np.ones(self.num_triplets)

		# for h,t in self.spanning:
		# 	r = random.choice(self.rel_info[(h,t)])
		# 	msg.append((h, r, t))
		# 	remaining_triplet_indexes[self.triplet2idx[(h,r,t)]] = 0
		# 	rels_encountered[r] = 1


		# for r in (1-rels_encountered).nonzero()[0].tolist():
		# 	h,t = random.choice(self.pair_info[int(r)])
		# 	msg.append((h, r, t))
		# 	remaining_triplet_indexes[self.triplet2idx[(h,r,t)]] = 0

		# start = time.time()
		# sup = [self.triplets[idx] for idx, tf in enumerate(remaining_triplet_indexes) if tf]
		msg=self.triplets
		msg = np.array(msg)
		# random.shuffle(sup)
		# sup = np.array(sup)
		# add_num = max(int(self.num_triplets * p) - len(msg), 0)
		# msg = np.concatenate([msg, sup[:add_num]])
		# sup = sup[add_num:]

		msg_inv = np.fliplr(msg).copy()
		msg_inv[:,1] += self.num_rel
		msg = np.concatenate([msg, msg_inv])

		return msg, sup
	def loads(self,path):
		ent2id=dict()
		rel2id=dict()
		with open(path.replace('train.txt','')+ "entities.dict", 'r') as f:
			for line in f.readlines():
				lines = line.strip().split('\t')
				if(len(lines)<2):
					continue
				elif(len(lines)==2):
					id,entity = lines[0],lines[1]
					label = entity
				else:
					continue
				ent2id[entity] = int(id)
		with open(path.replace('train.txt','')+"relations.dict", 'r') as f:
			for line in f.readlines():
				lines = line.strip().split('\t')
				if(len(lines)<2):
					continue
				elif(len(lines)==2):
					id,relation = lines[0],lines[1]
					label = relation
				else:
					continue
				rel2id[relation] = int(id)
		return ent2id,rel2id
class TestNewData():
	def __init__(self, path, data_type = "valid"):
		# 初始化函数，设置数据路径和数据类型（默认为"valid"）
		self.path = path
		self.data_type = data_type
		self.ent2id = None
		self.rel2id = None
		# 读取三元组数据，并初始化相关属性
		self.msg_triplets, self.sup_triplets, self.filter_dict = self.read_triplet()
		self.num_ent, self.num_rel = len(self.ent2id), len(self.rel2id)  # 计算实体关系数量
		

	def read_triplet(self):
		id2ent1,id2rel1,id2ent, id2rel, msg_triplets, sup_triplets = [], [], [], [],[],[]
		total_triplets = []
		# 从文件中读取消息三元组
		# with open(self.path + "msg.txt", 'r') as f:
		# with open(self.path + "entities.dict", 'r') as f:
		# 	for line in f.readlines():
		# 		id, entities = line.strip().split('\t')
		# 		id2ent.append(entities)
		# with open(self.path +  "relations.dict", 'r') as f:
		# 	for line in f.readlines():
		# 		id, relation = line.strip().split('\t')
		# 		id2rel.append(relation)	
		self.ent2id,self.rel2id=self.loads(self.path)
		with open(self.path + self.data_type + ".txt", 'r') as f:
			for line in f.readlines():
				h, r, t = line.strip().split('\t')
				# 添加实体和关系到列表，确保没有重复
				# id2ent.append(h)
				# id2ent.append(t)
				# id2rel.append(r)
				msg_triplets.append((h, r, t))
				total_triplets.append((h, r, t))
		# for data_type in ['valid', 'test']:
		# 	if data_type == self.data_type:
		# 		continue
		# 	with open(self.path + data_type + ".txt", 'r') as f:
		# 		for line in f.readlines():
		# 			h, r, t = line.strip().split('\t')
		# 			# id2ent.append(h)
		# 			# id2ent.append(t)
		# 			# id2rel.append(r)
		# 			# assert (self.ent2id[h], self.rel2id[r], self.ent2id[t]) not in msg_triplets, \
		# 			# 	(self.ent2id[h], self.rel2id[r], self.ent2id[t]) 
		# 			total_triplets.append((h,r,t))
		# 去重操作，确保实体和关系不重复
		# id2ent = remove_duplicate(id2ent)
		# id2rel = remove_duplicate(id2rel)
		# # 生成实体和关系的映射字典
		# self.ent2id = {ent: idx for idx, ent in enumerate(id2ent)}
		# self.rel2id = {rel: idx for idx, rel in enumerate(id2rel)}
		# 将三元组中的实体和关系转换为对应的ID
		num_rel = len(id2rel)
		msg_triplets = [(self.ent2id[h], self.rel2id[r], self.ent2id[t]) for h, r, t in msg_triplets]
		# 生成逆三元组，用于数据处理
		msg_inv_triplets = [(t, r+num_rel, h) for h,r,t in msg_triplets]
		# 根据数据类型读取相应的三元组文件
		with open(self.path + self.data_type + ".txt", 'r') as f:
			for line in f.readlines():
				h, r, t = line.strip().split('\t')
				sup_triplets.append((self.ent2id[h], self.rel2id[r], self.ent2id[t]))
				# 断言检查，确保没有重复的三元组
				# assert (self.ent2id[h], self.rel2id[r], self.ent2id[t]) not in msg_triplets, \
				# 	(self.ent2id[h], self.rel2id[r], self.ent2id[t]) 
				total_triplets.append((h,r,t))
		# 额外检查其他数据类型中的三元组	
		# for data_type in ['valid', 'test']:
		# 	if data_type == self.data_type:
		# 		continue
		# 	with open(self.path + data_type + ".txt", 'r') as f:
		# 		for line in f.readlines():
		# 			h, r, t = line.strip().split('\t')
					
		# 			# assert (self.ent2id[h], self.rel2id[r], self.ent2id[t]) not in msg_triplets, \
		# 			# 	(self.ent2id[h], self.rel2id[r], self.ent2id[t]) 
		# 			total_triplets.append((h,r,t))	

		# 创建过滤字典，用于快速查找相关三元组
		filter_dict = {}
		for triplet in total_triplets:
			h,r,t = triplet
			if ('_', self.rel2id[r], self.ent2id[t]) not in filter_dict:
				filter_dict[('_', self.rel2id[r], self.ent2id[t])] = [self.ent2id[h]]
			else:
				filter_dict[('_', self.rel2id[r], self.ent2id[t])].append(self.ent2id[h])

			if (self.ent2id[h], '_', self.ent2id[t]) not in filter_dict:
				filter_dict[(self.ent2id[h], '_', self.ent2id[t])] = [self.rel2id[r]]
			else:
				filter_dict[(self.ent2id[h], '_', self.ent2id[t])].append(self.rel2id[r])
				
			if (self.ent2id[h], self.rel2id[r], '_') not in filter_dict:
				filter_dict[(self.ent2id[h], self.rel2id[r], '_')] = [self.ent2id[t]]
			else:
				filter_dict[(self.ent2id[h], self.rel2id[r], '_')].append(self.ent2id[t])
		# 打印数据统计信息
		print(f"-----{self.data_type.capitalize()} Data Statistics-----")
		print(f"Message set has {len(msg_triplets)} triplets")
		print(f"Supervision set has {len(sup_triplets)} triplets")
		print(f"{len(self.ent2id)} entities, " + \
			  f"{len(self.rel2id)} relations, "+ \
			  f"{len(total_triplets)} triplets")
		# 将消息三元组和逆三元组合并，返回处理后的数据
		msg_triplets = msg_triplets + msg_inv_triplets

		return np.array(msg_triplets), np.array(sup_triplets), filter_dict
	
	def loads(self,path):
		ent2id=dict()
		rel2id=dict()
		with open(path.replace('train.txt','')+ "entities.dict", 'r') as f:
			for line in f.readlines():
				lines = line.strip().split('\t')
				if(len(lines)<2):
					continue
				elif(len(lines)==2):
					id,entity = lines[0],lines[1]
					label = entity
				else:
					continue
				ent2id[entity] = int(id)
		with open(path.replace('train.txt','')+"relations.dict", 'r') as f:
			for line in f.readlines():
				lines = line.strip().split('\t')
				if(len(lines)<2):
					continue
				elif(len(lines)==2):
					id,relation = lines[0],lines[1]
					label = relation
				else:
					continue
				rel2id[relation] = int(id)
		return ent2id,rel2id


