from relgraph import generate_relation_triplets
from dataset import TrainData, TestNewData
from tqdm import tqdm
import random
from models import InGram
import torch
import numpy as np
from utils import generate_neg
import os
from evaluation import evaluates
from initialize import initialize
from my_parser import parse

import os
# 设置多线程和随机种子以保证结果的可复现性
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# OMP_NUM_THREADS = 8
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
np.random.seed(seed) # Numpy module.
random.seed(seed) # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False # 禁用benchmark，保证可复现
torch.backends.cudnn.deterministic = True
# 设置PyTorch的一些环境参数，以优化运算和调试
torch.autograd.set_detect_anomaly(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.use_deterministic_algorithms(True)
# torch.set_num_threads(8)


# 解析命令行参数
args = parse()
for i in range(args.client_num):
	torch.cuda.empty_cache() # 清空CUDA缓存
	assert args.data_name in os.listdir(args.data_path), f"{args.data_name} Not Found"
	path = args.data_path + args.data_name + "/" + str(i)+"/"
	train = TrainData(path) # 加载训练数据
	num_ent=train.num_ent
	num_rel=train.num_rel
	valid = TestNewData(path, data_type = "valid") # 加载验证数据

	# 如果不禁用写入，设置检查点保存路径
	if not args.no_write:
		os.makedirs(f"./ckpt/{args.exp}/{args.data_name}_{str(i)}", exist_ok=True)
	# 格式化文件名以包含训练参数
	file_format = f"lr_{args.learning_rate}_dim_{args.dimension_entity}_{args.dimension_relation}" + \
				f"_bin_{args.num_bin}_total_{args.num_epoch}_every_{args.validation_epoch}" + \
				f"_neg_{args.num_neg}_layer_{args.num_layer_ent}_{args.num_layer_rel}" + \
				f"_hid_{args.hidden_dimension_ratio_entity}_{args.hidden_dimension_ratio_relation}" + \
				f"_head_{args.num_head}_margin_{args.margin}"

	# 提取并设置模型参数
	d_e = args.dimension_entity
	d_r = args.dimension_relation
	hdr_e = args.hidden_dimension_ratio_entity
	hdr_r = args.hidden_dimension_ratio_relation
	B = args.num_bin
	epochs = args.num_epoch
	valid_epochs = args.validation_epoch
	num_neg = args.num_neg
	bestmrr = 0
	# 初始化模型并移动到GPU
	my_model = InGram(dim_ent = d_e, hid_dim_ratio_ent = hdr_e, dim_rel = d_r, hid_dim_ratio_rel = hdr_r, \
					num_bin = B, num_ent=num_ent,num_rel=num_rel, num_layer_ent = args.num_layer_ent, num_layer_rel = args.num_layer_rel, \
					num_head = args.num_head)
	my_model = my_model.cuda()
	loss_fn = torch.nn.MarginRankingLoss(margin = args.margin, reduction = 'mean')
	# for name, param in my_model.state_dict().items() :
	# 	# if "layers_rel" in name :
	# 	# 	if "weight" in name:
	# 	# 		print(name)
	# 	# 		print(param)
	# 	print(name)
	# 设置优化器
	optimizer = torch.optim.Adam(my_model.parameters(), lr = args.learning_rate)
	# pbar = tqdm(range(epochs))

	total_loss = 0

	# 训练循环
	# for epoch in pbar:
	epoch = 0
	temp=0
	maxepoch=10000
	while(epoch<maxepoch):
		
		optimizer.zero_grad()
		msg, sup = train.split_transductive(1) # 随机分割数据 msg训练数据 

		# 初始化实体和关系的嵌入向量
		init_emb_ent, init_emb_rel, relation_triplets = initialize(train,msg, d_e, d_r, B)
		msg = torch.tensor(msg).cuda()
		sup = torch.tensor(sup).cuda()

		# 前向传播
		emb_ent, emb_rel = my_model(msg, relation_triplets)
		pos_scores = my_model.score(emb_ent, emb_rel, msg)
		neg_scores = my_model.score(emb_ent, emb_rel, generate_neg(msg, train.num_ent, num_neg = num_neg)) #  generate_neg生成负例三元组

		# 计算损失并反向传播
		loss = loss_fn(pos_scores.repeat(num_neg), neg_scores, torch.ones_like(neg_scores))

		loss.backward()
		torch.nn.utils.clip_grad_norm_(my_model.parameters(), 0.1, error_if_nonfinite = False)
		optimizer.step()
		total_loss += loss.item()
		# pbar.set_description(f"loss {loss.item()}")	 # 更新进度条
		print('loss:',loss.item(),'epoch:',epoch)
		# 每隔一定周期进行验证
		if ((epoch + 1) % valid_epochs) == 0:
			temp+=1
			print("Validation")
			my_model.eval()
			val_init_emb_ent, val_init_emb_rel, val_relation_triplets = initialize(valid, valid.msg_triplets, \
																					d_e, d_r, B)

			mrr = evaluates(my_model, valid, epoch, val_init_emb_ent, val_init_emb_rel, val_relation_triplets)

			# 如果允许写入，保存模型状态
			if not args.no_write:
				torch.save({'model_state_dict': my_model.state_dict(), \
							'optimizer_state_dict': optimizer.state_dict(), \
							'inf_emb_ent': val_init_emb_ent, \
							'inf_emb_rel': val_init_emb_rel}, \
					f"ckpt/{args.exp}/{args.data_name}_{str(i)}/{file_format}_{epoch+1}.ckpt")
			if bestmrr < mrr:
				temp=0
				bestmrr = mrr
				print('bestepoch:',epoch)
				if not args.no_write:
					torch.save({'model_state_dict': my_model.state_dict(), \
								'optimizer_state_dict': optimizer.state_dict(), \
								'inf_emb_ent': val_init_emb_ent, \
								'inf_emb_rel': val_init_emb_rel}, \
						f"ckpt/{args.exp}/{args.data_name}_{str(i)}/{file_format}_best.ckpt")
			my_model.train() # 恢复训练模式
		epoch+=1
		# if temp > 10:
		# 	break