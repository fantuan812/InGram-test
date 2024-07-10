import torch
from model import InGram
from relgraph import generate_relation_triplets
from dataset import TrainData, TestNewData
from tqdm import tqdm
import random
from models import InGram
import torch
import numpy as np
from utils import generate_neg,get_metrics
import os
from evaluation import evaluates,evaluatemulti
from initialize import initialize
from my_parser import parse

import os
# 设置多线程和随机种子以保证结果的可复现性
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
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

class Server:
    def __init__(self,args):
        # self.args = args
        # self.data_num=model_params
        # path = self.args.data_path + self.args.data_name + "/"+model_params+"/"
        # self.train = TrainData(path)
        # self.model = InGram(
        #     dim_ent=args.dimension_entity,
        #     hid_dim_ratio_ent=args.hidden_dimension_ratio_entity,
        #     dim_rel=args.dimension_relation,
        #     hid_dim_ratio_rel=args.hidden_dimension_ratio_relation,
        #     num_bin=args.num_bin,
        #     target=self.train ,
        #     num_layer_ent=args.num_layer_ent,
        #     num_layer_rel=args.num_layer_rel,
        #     num_head=args.num_head
        # ).cuda()
        # # self.relation_layer_weights = {name: param.clone() for name, param in self.model.state_dict().items() if "layers_rel" in name and ("aggr_proj" in name or "attn_vec" in name or "attn_proj" in name)}
        # self.relation_layer_weights = {name: param.clone() for name, param in self.model.state_dict().items() if ("layers_rel" in name or "layers_ent" in name )  and "aggr_proj" in name }
        print()
        # print('self.relation_layer_weights',self.relation_layer_weights)
    def collect_and_aggregate_weights(self, clients):
        # 从所有客户端收集权重并聚合
        collected_weights = [client.get_relation_weights() for client in clients]
        # print(collected_weights)
        # self.relation_layer_weights = {name: torch.mean(torch.stack([cw[name] for cw in collected_weights]), dim=0) for name in collected_weights[0].keys()}
        ent_size = 0
        for client in clients:
            ent_size += client.num_ent
        rel_size = 0
        for client in clients:
            rel_size += client.num_rel
        self.relation_layer_weights = {}
        for name in collected_weights[0].keys():
            if "layers_rel" in name:
                accumulated_weights = torch.sum(torch.stack([
                    torch.mul(client.relation_weights[name], client.num_rel) for client in clients
                ]), dim=0)
                averaged_weights = torch.div(accumulated_weights, rel_size).clone()
                self.relation_layer_weights[name] = averaged_weights
            elif "layers_ent" in name :
                # 加权平均
                # accumulated_weights = torch.sum(torch.stack([
                #     torch.mul(client.relation_weights[name], client.num_ent) for client in clients
                # ]), dim=0)
                # averaged_weights = torch.div(accumulated_weights, ent_size).clone()
                # self.relation_layer_weights[name] = averaged_weights
                # 简单平均
                averaged_weights=torch.mean(torch.stack([cw[name] for cw in collected_weights]), dim=0)
                self.relation_layer_weights[name] = averaged_weights
    def distribute_weights(self, clients):
        # 分发权重给所有客户端
        for client in clients:
            client.update_relation_weights(self.relation_layer_weights)

class Client:
    def __init__(self, model_params):
        self.args = parse()
        self.data_num=model_params
        # for i in range(3):
        #     if '_'+str(i) in self.args.data_name:
        #         self.path=self.args.data_name.replace('_'+str(i),'')
        assert self.args.data_name in os.listdir(self.args.data_path), f"{self.args.data_name} Not Found"
        path = self.args.data_path + self.args.data_name + "/"+model_params+"/"
        self.train = TrainData(path) # 加载训练数据
        self.valid = TestNewData(path, data_type = "valid") # 加载验证数据
        self.mrr=0
        self.num_ent=self.train.num_ent
        self.num_rel=self.train.num_rel
        # 提取并设置模型参数
        self.d_e = self.args.dimension_entity
        self.d_r = self.args.dimension_relation
        self.hdr_e = self.args.hidden_dimension_ratio_entity
        self.hdr_r = self.args.hidden_dimension_ratio_relation
        self.B = self.args.num_bin
        self.epochs = self.args.num_epoch
        # if(int(model_params)==2):
        #     self.valid_epochs = self.args.validation_epoch*2
        # else:
        #     self.valid_epochs = self.args.validation_epoch
        self.valid_epochs = self.args.validation_epoch
        self.num_neg = self.args.num_neg
        self.file_format = f"lr_{self.args.learning_rate}_dim_{self.args.dimension_entity}_{self.args.dimension_relation}" + \
                    f"_bin_{self.args.num_bin}_total_{self.args.num_epoch}_every_{self.args.validation_epoch}" + \
                    f"_neg_{self.args.num_neg}_layer_{self.args.num_layer_ent}_{self.args.num_layer_rel}" + \
                    f"_hid_{self.args.hidden_dimension_ratio_entity}_{self.args.hidden_dimension_ratio_relation}" + \
                    f"_head_{self.args.num_head}_margin_{self.args.margin}"
        self.model = InGram(dim_ent = self.d_e, hid_dim_ratio_ent = self.hdr_e, dim_rel = self.d_r, hid_dim_ratio_rel = self.hdr_r, \
                        num_bin = self.B,num_ent=self.num_ent,num_rel=self.num_rel, num_layer_ent = self.args.num_layer_ent, num_layer_rel = self.args.num_layer_rel, \
                        num_head = self.args.num_head)
        # self.relation_weights = {name: param.clone() for name, param in self.model.state_dict().items() if "layers_ent" in name if "aggr_proj" in name}
        self.relation_weights = {name: param.clone() for name, param in self.model.state_dict().items() if ("layers_rel" in name or "layers_ent" in name )  and "aggr_proj" in name  }
    def train_and_update_weights(self):
        # self.relation_weights = {name: param.clone() for name, param in self.model.state_dict().items() if "layers_ent" in name if "weight" in name}
        # self.relation_weights = {name: param.clone() for name, param in self.model.state_dict().items() if "layers_rel" in name and "aggr_proj" in name }
        # print("self.relation_weights",self.relation_weights)
        # 解析命令行参数
        # 初始化模型并移动到GPU
        self.model = self.model.cuda()
        loss_fn = torch.nn.MarginRankingLoss(margin = self.args.margin, reduction = 'mean')

        # 设置优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.args.learning_rate)
        # print("self.data_num:",self.data_num)
        # print(self.data_num)
        # print(self.data_num == 2)
        # if(int(self.data_num) == 2):
        #     pbar= tqdm(range(self.valid_epochs*2))
        # else:
        #     pbar = tqdm(range(self.valid_epochs))
        pbar = tqdm(range(self.valid_epochs))
        print(pbar)
        total_loss = 0
        msg, sup = self.train.split_transductive(1) # 随机分割数据 msg训练数据 
        # 初始化实体和关系的嵌入向量
        init_emb_ent, init_emb_rel, relation_triplets = initialize(self.train,msg, self.d_e, self.d_r, self.B)
        msg = torch.tensor(msg).cuda()
        sup = torch.tensor(sup).cuda()

        # 训练循环
        for epoch in pbar:
            self.optimizer.zero_grad()
            

            # 前向传播
            emb_ent, emb_rel = self.model(msg, relation_triplets)
            pos_scores = self.model.score(emb_ent, emb_rel, msg)
            # neg_triplets = generate_neg(msg, self.train.num_ent, num_neg=self.num_neg).cuda()
            # neg_scores = self.model.score(emb_ent, emb_rel, neg_triplets)
            neg_scores = self.model.score(emb_ent, emb_rel, generate_neg(msg, self.train.num_ent, num_neg = self.num_neg)) #  generate_neg生成负例三元组

            # 计算损失并反向传播
            loss = loss_fn(pos_scores.repeat(self.num_neg), neg_scores, torch.ones_like(neg_scores))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1, error_if_nonfinite = False)
            self.optimizer.step()
            total_loss += loss.item()
            pbar.set_description(f"loss {loss.item()}")	 # 更新进度条

            # 每隔一定周期进行验证
            # if ((epoch + 1) % self.valid_epochs) == 0:         
        # self.relation_weights = {name: param.clone() for name, param in self.model.state_dict().items() if "layers_ent" in name if "aggr_proj" in name}
        self.relation_weights = {name: param.clone() for name, param in self.model.state_dict().items() if ("layers_rel" in name or "layers_ent" in name ) and "aggr_proj" in name  }
        # print("weights",self.relation_weights)
    def evaluation(self):
        print("Validation")
        self.model.eval()
        # 如果不禁用写入，设置检查点保存路径
        if not self.args.no_write:
            os.makedirs(f"./ckpt/{self.args.exp}/{self.args.data_name}/{self.data_num}", exist_ok=True)
        
        val_init_emb_ent, val_init_emb_rel, val_relation_triplets = initialize(self.valid, self.valid.msg_triplets, \
                                                                                self.d_e, self.d_r, self.B)
        print(self.args.data_name)
        mrr=evaluates(self.model, self.valid, 200,val_init_emb_ent, val_init_emb_rel, val_relation_triplets)
        self.save(mrr,val_init_emb_ent,val_init_emb_rel)
        self.model.train()
    def evaluationmulti(self):
        print("Validation")
        self.model.eval()
        # 如果不禁用写入，设置检查点保存路径
        if not self.args.no_write:
            os.makedirs(f"./ckpt/{self.args.exp}/{self.args.data_name}/{self.data_num}", exist_ok=True)
        val_init_emb_ent, val_init_emb_rel, val_relation_triplets = initialize(self.valid, self.valid.msg_triplets, \
                                                                                self.d_e, self.d_r, self.B)
        print(self.args.data_name)
        rank,mrr=evaluatemulti(self.model, self.valid, 200,val_init_emb_ent, val_init_emb_rel, val_relation_triplets)
        self.save(mrr,val_init_emb_ent,val_init_emb_rel)
        self.model.train()
        return rank
    def save(self,mrr,val_init_emb_ent,val_init_emb_rel):
        if mrr>self.mrr:
            print("bestepoch")
            self.mrr=mrr
            # 如果允许写入，保存模型状态
            if not self.args.no_write:
                torch.save({'model_state_dict': self.model.state_dict(), \
                            'optimizer_state_dict': self.optimizer.state_dict(), \
                            'inf_emb_ent': val_init_emb_ent, \
                            'inf_emb_rel': val_init_emb_rel}, \
                    f"ckpt/{self.args.exp}/{self.args.data_name}/{self.data_num}/{self.file_format}.ckpt")
    def savebest(self):
        
        val_init_emb_ent, val_init_emb_rel, val_relation_triplets = initialize(self.valid, self.valid.msg_triplets, \
                                                                                self.d_e, self.d_r, self.B)
        # 如果允许写入，保存模型状态
        if not self.args.no_write:
            torch.save({'model_state_dict': self.model.state_dict(), \
                        'optimizer_state_dict': self.optimizer.state_dict(), \
                        'inf_emb_ent': val_init_emb_ent, \
                        'inf_emb_rel': val_init_emb_rel}, \
                f"ckpt/{self.args.exp}/{self.args.data_name}/{self.data_num}/{self.file_format}_best.ckpt")
    def get_relation_weights(self):
        # 提供关系层权重
        return self.relation_weights

    def update_relation_weights(self, updated_weights):
        # 更新关系层权重
        
        state_dict = self.model.state_dict()
        # 检查参数名称是否匹配
        # print("Model keys:", self.model.state_dict().keys())
        # print("Updated keys:", updated_weights.keys())

        # # 检查权重维度是否匹配
        # for name, param in updated_weights.items():
        #     model_param = self.model.state_dict().get(name, None)
        #     if model_param is not None:
        #         if model_param.shape != param.shape:
        #             print(f"Mismatch in dimensions for {name}: model {model_param.shape}, updated {param.shape}")
        #         else:
        #             print(f"Dimension match for {name} is correct.")
        #     else:
        #         print(f"{name} not found in model parameters.")
        for name, param in updated_weights.items():
            if name in state_dict:
                state_dict[name].copy_(param)
        self.model.load_state_dict(state_dict)


args = parse()
server = Server(args)
num =args.client_num
clients = [Client(str(i)) for i in range(num)]
# 客户端训练并更新本地权重
# for client in clients:
#         client.train_and_update_weights() 
bestmrr=0
temp=0
# for i in range(50):
epoch=0
maxepoch=100
while(epoch<maxepoch):
    epoch +=1
    ranks=[]
    

    for client in clients:
        client.train_and_update_weights()
    # if(i>=99 and(i % 100 ) ==0):
    # for client in clients:
    #     client.evaluation()
    # 服务器收集、聚合并分发权重
    server.collect_and_aggregate_weights(clients)
    server.distribute_weights(clients)
    if epoch%1==0:
        for client in clients:
            rank=client.evaluationmulti()
            ranks+=rank
        
        mr, mrr, hit10, hit3, hit1 = get_metrics(ranks)
        
        if mrr > bestmrr :
            bestmrr=mrr
            temp = 0
            print("epoch:",epoch)
            print("--------BestAvgLP--------")
            print(f"MR: {mr:.1f}")
            print(f"MRR: {mrr:.3f}")
            print(f"Hits@10: {hit10:.3f}")
            print(f"Hits@1: {hit1:.3f}")
            for client in clients:
                client.savebest()
        else:
            temp += 1
        # if temp > 10:
        #     break
    
    