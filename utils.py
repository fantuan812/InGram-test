import numpy as np

def remove_duplicate(x):
	return list(dict.fromkeys(x))

def generate_neg(triplets, num_ent, num_neg = 1):
	import torch
	# 将输入的三元组沿新的维度复制 num_neg 次
	neg_triplets = triplets.unsqueeze(dim=1).repeat(1,num_neg,1)
	# 生成一个随机矩阵，决定是扰动头部还是尾部实体
	rand_result = torch.rand((len(triplets),num_neg)).cuda()
	perturb_head = rand_result < 0.5  # 头部扰动的布尔索引
	perturb_tail = rand_result >= 0.5  # 尾部扰动的布尔索引
	# 为每个扰动位置生成一个随机索引，这个索引表示将被替换的实体
	rand_idxs = torch.randint(low=0, high = num_ent-1, size = (len(triplets),num_neg)).cuda()
	# 确保生成的随机索引不与原始的实体索引相同，防止生成有效的三元组作为负例
    # 对于头部扰动的情况，如果随机索引大于等于原头部实体索引，则索引+1
	rand_idxs[perturb_head] += rand_idxs[perturb_head] >= neg_triplets[:,:,0][perturb_head]
	# 对于尾部扰动的情况，同样的逻辑
	rand_idxs[perturb_tail] += rand_idxs[perturb_tail] >= neg_triplets[:,:,2][perturb_tail]
	# 使用生成的随机索引替换头部或尾部实体
	neg_triplets[:,:,0][perturb_head] = rand_idxs[perturb_head]
	neg_triplets[:,:,2][perturb_tail] = rand_idxs[perturb_tail]
	# 将 neg_triplets 由 shape [batch_size, num_neg, 3] 转换为 [batch_size * num_neg, 3]
	neg_triplets = torch.cat(torch.split(neg_triplets, 1, dim = 1), dim = 0).squeeze(dim = 1)

	return neg_triplets

def get_rank(triplet, scores, filters, target = 0):
	thres = scores[triplet[0,target]].item()
	scores[filters] = thres - 1
	rank = (scores > thres).sum() + (scores == thres).sum()//2 + 1
	return rank.item()

def get_metrics(rank):
	rank = np.array(rank, dtype = np.int_)
	mr = np.mean(rank)
	mrr = np.mean(1 / rank)
	hit10 = np.sum(rank < 11) / len(rank)
	hit3 = np.sum(rank < 4) / len(rank)
	hit1 = np.sum(rank < 2) / len(rank)
	return mr, mrr, hit10, hit3, hit1