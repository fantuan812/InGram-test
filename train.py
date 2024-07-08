from relgraph import generate_relation_triplets
from dataset import TrainData, TestNewData
from tqdm import tqdm
import random
from models import InGram
import torch
import numpy as np
from utils import generate_neg
import os
from evaluation import evaluates,evaluateg
from initialize import initialize
from my_parser import parse
from torch.utils.tensorboard import SummaryWriter
import os
import time
from test1 import test
# 设置多线程和随机种子以保证结果的可复现性  你好
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
torch.cuda.empty_cache() # 清空CUDA缓存

# 解析命令行参数
args = parse()

assert args.data_name in os.listdir(args.data_path), f"{args.data_name} Not Found"
path = args.data_path + args.data_name + "/"
train = TrainData(path) # 加载训练数据

num_ent=train.num_ent
num_rel=train.num_rel
valid = TestNewData(path, data_type = "valid") # 加载验证数据

# 如果不禁用写入，设置检查点保存路径
if not args.no_write:
	os.makedirs(f"./ckpt/{args.exp}/{args.data_name}", exist_ok=True)
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
				num_bin = B,num_ent=num_ent,num_rel=num_rel, num_layer_ent = args.num_layer_ent, num_layer_rel = args.num_layer_rel, \
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



# init_emb_ent, init_emb_rel, relation_triplets = initialize(train,msg, d_e, d_r, B)
# emb_ent, emb_rel, relation_triplets = initialize(train,msg, d_e, d_r, B)
name='FB15k 128'
total_loss = 0
# writer = SummaryWriter(log_dir='./logs/' + time.strftime('%y-%m-%d_%H.%M', time.localtime()))
# writer = {
#     'valid':SummaryWriter(log_dir='./logs/' + time.strftime('%y-%m-%d_%H.%M', time.localtime())), 
#     'test': SummaryWriter(log_dir='./logs/' + time.strftime('%y-%m-%d_%H.%M', time.localtime())+'test')
# }
writer = {
    'valid':SummaryWriter(log_dir='./logs/' + name), 
    # 'test': SummaryWriter(log_dir='./logs/' + name+'test')
}
# 训练循环
# for epoch in pbar:
# print(my_model.state_dict())
epoch = 0
temp=0
maxepoch=10000
while(epoch<maxepoch):
	optimizer.zero_grad()
	
	# print('init_emb_ent',init_emb_ent)
	# print('init_emb_rel',init_emb_rel)
	# print('relation_triplets',relation_triplets)
	# print(len(relation_triplets))
	
	# sup = torch.tensor(sup).cuda()
	# print("emb_ent:",emb_ent)
	# print("emb_rel:",emb_rel)
	# 前向传播
	msg, sup = train.split_transductive(1) # 随机分割数据 msg训练数据 
	# 初始化实体和关系的嵌入向量
	init_emb_ent, init_emb_rel, relation_triplets = initialize(train,msg, d_e, d_r, B)
	msg = torch.tensor(msg).cuda()
	emb_ent, emb_rel = my_model(init_emb_ent, init_emb_rel, msg, relation_triplets)
	# print('initemb_ent',emb_ent)
	# emb_ent, emb_rel = my_model(emb_ent, emb_rel, msg, relation_triplets)
	pos_scores = my_model.score(emb_ent, emb_rel, msg)
	neg_scores = my_model.score(emb_ent, emb_rel, generate_neg(msg, train.num_ent, num_neg = num_neg)) #  generate_neg生成负例三元组

	# 计算损失并反向传播
	loss = loss_fn(pos_scores.repeat(num_neg), neg_scores, torch.ones_like(neg_scores))

	loss.backward()

	torch.nn.utils.clip_grad_norm_(my_model.parameters(), 0.1, error_if_nonfinite = False)
	optimizer.step()
	total_loss += loss.item()
	writer['valid'].add_scalar("loss", loss.detach(), epoch)
	
	
	# pbar.set_description(f"loss {loss.item()}")	 # 更新进度条
	print('epoch',epoch,'loss',loss.item())
	
	# 每隔一定周期进行验证
	if ((epoch + 1) % valid_epochs) == 0:
		print("Validationepoch:",epoch)
		my_model.eval()
		val_init_emb_ent, val_init_emb_rel, val_relation_triplets = initialize(valid, valid.msg_triplets, \
																				d_e, d_r, B)
		
		mr, mrr, hit10, hit3, hit1=evaluateg(my_model, valid, epoch, val_init_emb_ent, val_init_emb_rel, val_relation_triplets)
		writer['valid'].add_scalar("mr", mr, epoch)
		writer['valid'].add_scalar("mrr", mrr, epoch)
		writer['valid'].add_scalar("hit10", hit10, epoch)
		writer['valid'].add_scalar("hit3", hit3, epoch)
		writer['valid'].add_scalar("hit1", hit1, epoch)
		writer['valid'].close
		# 如果允许写入，保存模型状态
		if not args.no_write:
			torch.save({'model_state_dict': my_model.state_dict(), \
						'optimizer_state_dict': optimizer.state_dict(), \
						'inf_emb_ent': val_init_emb_ent, \
						'inf_emb_rel': val_init_emb_rel}, \
				f"ckpt/{args.exp}/{args.data_name}/{file_format}_{epoch+1}.ckpt")
		# mrr1=test(epoch)
		# writer['test'].add_scalar("mrr", mrr1, epoch)
		# writer['test'].close
		if bestmrr<mrr:
			temp=0
			bestmrr=mrr
			print('bestepoch:',epoch)
			if not args.no_write:
				torch.save({'model_state_dict': my_model.state_dict(), \
							'optimizer_state_dict': optimizer.state_dict(), \
							'inf_emb_ent': val_init_emb_ent, \
							'inf_emb_rel': val_init_emb_rel}, \
					f"ckpt/{args.exp}/{args.data_name}/{file_format}_best.ckpt")
		my_model.train() # 恢复训练模式
		temp+=1
	epoch+=1
	# if temp > 10:
	# 	break
	
	# writer.close()

	
	# emb_ent=emb_ent.detach()
	# emb_rel=emb_rel.detach()
	# print('emb_ent',emb_ent)
