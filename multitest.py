from relgraph import generate_relation_triplets
from dataset import TestNewData,TrainData
from tqdm import tqdm
import random
from models import InGram
import torch
import numpy as np
from utils import get_rank, get_metrics
from my_parser import parse
from evaluation import evaluatemulti
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
torch.cuda.empty_cache()

args = parse(test=True)
ranks=[]
num =args.client_num
for i in range(num):
    assert args.data_name in os.listdir(args.data_path), f"{args.data_name} Not Found"
    path = args.data_path + args.data_name  + "/" + str(i)+"/"
    train = TrainData(path) # 加载训练数据
    num_ent=train.num_ent
    num_rel=train.num_rel
    test = TestNewData(path, data_type = "test")
    file_format = f"lr_{args.learning_rate}_dim_{args.dimension_entity}_{args.dimension_relation}" + \
                    f"_bin_{args.num_bin}_total_{args.num_epoch}_every_{args.validation_epoch}" + \
                    f"_neg_{args.num_neg}_layer_{args.num_layer_ent}_{args.num_layer_rel}" + \
                    f"_hid_{args.hidden_dimension_ratio_entity}_{args.hidden_dimension_ratio_relation}" + \
                    f"_head_{args.num_head}_margin_{args.margin}"
    if not args.best:
        file_format = f"lr_{args.learning_rate}_dim_{args.dimension_entity}_{args.dimension_relation}" + \
                    f"_bin_{args.num_bin}_total_{args.num_epoch}_every_{args.validation_epoch}" + \
                    f"_neg_{args.num_neg}_layer_{args.num_layer_ent}_{args.num_layer_rel}" + \
                    f"_hid_{args.hidden_dimension_ratio_entity}_{args.hidden_dimension_ratio_relation}" + \
                    f"_head_{args.num_head}_margin_{args.margin}"

    d_e = args.dimension_entity
    d_r = args.dimension_relation
    hdr_e = args.hidden_dimension_ratio_entity
    hdr_r = args.hidden_dimension_ratio_relation
    B = args.num_bin
    num_neg = args.num_neg

    my_model = InGram(dim_ent = d_e, hid_dim_ratio_ent = hdr_e, dim_rel = d_r, hid_dim_ratio_rel = hdr_r,\
                    num_bin = B, num_ent=num_ent,num_rel=num_rel,num_layer_ent = args.num_layer_ent, num_layer_rel = args.num_layer_rel, \
                    num_head = args.num_head)
    my_model = my_model.cuda()

    if not args.single:
        if not args.best:
            my_model.load_state_dict(torch.load(f"ckpt/{args.exp}/{args.data_name}/{str(i) }/{file_format}.ckpt")["model_state_dict"])
        else : 
            my_model.load_state_dict(torch.load(f"ckpt/{args.exp}/{args.data_name}/{str(i) }/{file_format}_best.ckpt")["model_state_dict"])
    else :
        if not args.best:
            print("single")
            my_model.load_state_dict(torch.load(f"ckpt/{args.exp}/{args.data_name+'_'+str(i) }/{file_format}_{args.target_epoch}.ckpt")["model_state_dict"])
        else :
            print('best')
            my_model.load_state_dict(torch.load(f"ckpt/{args.exp}/{args.data_name+'_'+str(i) }/{file_format}_best.ckpt")["model_state_dict"])


    print("Test")
    my_model.eval()
    test_msg = test.msg_triplets
    test_sup = test.sup_triplets
    test_relation_triplets = generate_relation_triplets(test_msg, test.num_ent, test.num_rel, B)
    if not args.single:
        if not args.best:
            test_init_emb_ent = torch.load(f"ckpt/{args.exp}/{args.data_name}/{str(i) }/{file_format}.ckpt")["inf_emb_ent"]
            test_init_emb_rel = torch.load(f"ckpt/{args.exp}/{args.data_name}/{str(i) }/{file_format}.ckpt")["inf_emb_rel"]
        else:
            test_init_emb_ent = torch.load(f"ckpt/{args.exp}/{args.data_name}/{str(i) }/{file_format}_best.ckpt")["inf_emb_ent"]
            test_init_emb_rel = torch.load(f"ckpt/{args.exp}/{args.data_name}/{str(i) }/{file_format}_best.ckpt")["inf_emb_rel"]
    else:
        if not args.best:
            test_init_emb_ent = torch.load(f"ckpt/{args.exp}/{args.data_name+'_'+str(i) }/{file_format}_{args.target_epoch}.ckpt")["inf_emb_ent"]
            test_init_emb_rel = torch.load(f"ckpt/{args.exp}/{args.data_name+'_'+str(i) }/{file_format}_{args.target_epoch}.ckpt")["inf_emb_rel"]
        else :
            test_init_emb_ent = torch.load(f"ckpt/{args.exp}/{args.data_name+'_'+str(i) }/{file_format}_best.ckpt")["inf_emb_ent"]
            test_init_emb_rel = torch.load(f"ckpt/{args.exp}/{args.data_name+'_'+str(i) }/{file_format}_best.ckpt")["inf_emb_rel"]

    test_sup = torch.tensor(test_sup).cuda()
    test_msg = torch.tensor(test_msg).cuda()

    test_relation_triplets = torch.tensor(test_relation_triplets).cuda()

    rank,mrr = evaluatemulti(my_model, test, args.target_epoch-1, test_init_emb_ent, test_init_emb_rel, test_relation_triplets)
    ranks += rank
print("--------AllLP--------")
mr, mrr, hit10, hit3, hit1 = get_metrics(ranks)
print(f"MR: {mr:.1f}")
print(f"MRR: {mrr:.3f}")
print(f"Hits@10: {hit10:.3f}")
print(f"Hits@1: {hit1:.3f}")