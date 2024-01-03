#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy, sys
import time
import numpy as np
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
import random
import torch.utils.model_zoo as model_zoo
from pathlib import Path
import pickle

# __file__에서 상위 폴더의 lib에 대한 절대 경로를 가져옴
lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    #sys.path 리스트 맨 앞에 추가
    sys.path.insert(0, str(lib_dir))
mod_dir = (Path(__file__).parent / ".." / "lib" / "models").resolve()
if str(mod_dir) not in sys.path:
    sys.path.insert(0, str(mod_dir))

from resnet import resnet18
from options import args_parser
from update import LocalUpdate, save_protos, LocalTest, test_inference_new_het_lt
from models import CNNMnist, CNNFemnist
from utils import get_dataset, average_weights, exp_details, proto_aggregation, agg_func, average_weights_per, average_weights_sem

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    # 'resnet18' : 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}



def FedProto_taskheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list):
    summary_writer = SummaryWriter('../tensorboard/'+ args.dataset +'_fedproto_' + str(args.ways) + 'w' + str(args.shots) + 's' + str(args.stdev) + 'e_' + str(args.num_users) + 'u_' + str(args.rounds) + 'r')
    # 글로벌 프로토 정의
    global_protos = []

    #클라이언트 인덱스 리스트
    idxs_users = np.arange(args.num_users)
    protos_client_has = {}
    for idx in idxs_users:
        protos_client_has[idx] = classes_list[idx]
    reverse_dict = {}
    for key, values in protos_client_has.items():
        sorted_values = tuple(sorted(values))
        reverse_dict.setdefault(sorted_values, []).append(key)
    same_clients = [values for values in reverse_dict.values() if len(values) > 1]

    print(f'same_clients: {same_clients}')

    train_loss, train_accuracy = [], []

    for round in tqdm(range(args.rounds)):
        local_weights, local_losses, local_protos = [], [], {}
        print(f'\n | Global Training Round : {round + 1} |\n')

        proto_loss = 0
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            #protos는 클라이언트별로 각 클래스의 프로토를 딕셔너리로 저장 {클래스: 프로토}
            #protos.items()는 딕셔너리의 키와 값을 튜플로 반환
            #protos.keys()는 딕셔너리의 키를 반환
            #protos.values()는 딕셔너리의 값을 반환
            #protos[클래스]는 딕셔너리의 클래스에 해당하는 프로토를 반환
            w, loss, acc, protos = local_model.update_weights_het(args, idx, global_protos, model=copy.deepcopy(local_model_list[idx]), global_round=round)
            # if idx == 3:
            #     # 0번째 클라이언트는 [9, 7, 1, 4, 0]의 클래스를 갖고 있음
            #     # 프로토의 키는 클래스, 값은 프로토 -> protos[9]는 9번 클래스의 프로토 (shot 개수)를 반환
            #     # n-way, k-shot에서 n은 각 클라이언트가 갖는 클래스의 개수, k는 각 클라이언트가 갖는 클래스의 샘플 (proto) 개수
            #     print(f'proto item len: {len(protos.items())}, proto item: {protos.keys()}, proto item value len 0 : {len(protos[0])}, \
            #           proto item value len 5 : {protos[5][0].shape}, \
            #             proto item value len 7 : {len(protos[7])}')
            #     # print(f'proto: {protos}')
            # if idx == 4:
            #     print(f'proto item len: {len(protos.items())}, proto item: {protos.keys()}, proto item value len 7 : {len(protos[7])}, \
            #           proto item value len 9 : {len(protos[9])}')

            #클래스 별 shot 개의 프로토를 평균내어 agg_protos에 저장
            agg_protos = agg_func(protos)
            #클라이언트 별 로컬 에포크 끝난 이후 마지막 가중치 저장
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss['total']))
            local_protos[idx] = agg_protos
            summary_writer.add_scalar('Train/Loss/user' + str(idx + 1), loss['total'], round)
            summary_writer.add_scalar('Train/Loss1/user' + str(idx + 1), loss['1'], round)
            summary_writer.add_scalar('Train/Loss2/user' + str(idx + 1), loss['2'], round)
            summary_writer.add_scalar('Train/Acc/user' + str(idx + 1), acc, round)
            proto_loss += loss['2']

        # update global weights
        local_weights_list = local_weights
        # print(f'local_weights_list: {len(local_weights_list)}')

        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights_list[idx], strict=True)
            local_model_list[idx] = local_model



        # update global weights
        global_protos = proto_aggregation(local_protos)
        #global_proto는 각 클래스에 대한 평균 프로토들 즉, 10개의 프로토를 갖고 있음
        # print(f'global_protos: {global_protos}')
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
    
    acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_lt, global_protos)

    # with open('fedproto_wo_grad_20-5-100-50-4.pickle', 'wb') as f:
    #     pickle.dump(acc_list_g, f)
    print('For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_l), np.std(acc_list_l)))
    print('For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_g),np.std(acc_list_g)))
    print('For all users (with protos), mean of proto loss is {:.5f}, std of proto loss is {:.5f}'.format(np.mean(loss_list), np.std(loss_list)))

    # save protos
    if args.dataset == 'mnist':
        save_protos(args, local_model_list, test_dataset, user_groups_lt)

def FedProto_modelheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list):
    summary_writer = SummaryWriter('../tensorboard/'+ args.dataset +'_fedproto_mh_' + str(args.ways) + 'w' + str(args.shots) + 's' + str(args.stdev) + 'e_' + str(args.num_users) + 'u_' + str(args.rounds) + 'r')

    global_protos = []
    idxs_users = np.arange(args.num_users)

    train_loss, train_accuracy = [], []

    for round in tqdm(range(args.rounds)):
        local_weights, local_losses, local_protos = [], [], {}
        print(f'\n | Global Training Round : {round + 1} |\n')

        proto_loss = 0
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w, loss, acc, protos = local_model.update_weights_het(args, idx, global_protos, model=copy.deepcopy(local_model_list[idx]), global_round=round)
            agg_protos = agg_func(protos)

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss['total']))

            local_protos[idx] = agg_protos
            summary_writer.add_scalar('Train/Loss/user' + str(idx + 1), loss['total'], round)
            summary_writer.add_scalar('Train/Loss1/user' + str(idx + 1), loss['1'], round)
            summary_writer.add_scalar('Train/Loss2/user' + str(idx + 1), loss['2'], round)
            summary_writer.add_scalar('Train/Acc/user' + str(idx + 1), acc, round)
            proto_loss += loss['2']

        # update global weights
        local_weights_list = local_weights

        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights_list[idx], strict=True)
            local_model_list[idx] = local_model

        # update global protos
        global_protos = proto_aggregation(local_protos)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

    acc_list_l, acc_list_g = test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_lt, global_protos)
    print('For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_g),np.std(acc_list_g)))
    print('For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_l), np.std(acc_list_l)))

if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()
    exp_details(args)

    # set random seeds
    args.device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda:2':
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # load dataset and user groups
    n_list = np.random.randint(max(2, args.ways - args.stdev), min(args.num_classes, args.ways + args.stdev + 1), args.num_users)
    if args.dataset == 'mnist':
        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev - 1, args.num_users)
    elif args.dataset == 'cifar10':
        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev + 1, args.num_users)
    elif args.dataset =='cifar100':
        k_list = np.random.randint(args.shots, args.shots + 1, args.num_users)
    elif args.dataset == 'femnist':
        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev + 1, args.num_users)

    train_dataset, test_dataset, user_groups, user_groups_lt, classes_list, classes_list_gt = get_dataset(args, n_list, k_list)



    # 출력 채널을 및 클라이언트 설정을 (이기종) 다르게하여 모델 생성
    local_model_list = []
    for i in range(args.num_users):
        if args.dataset == 'mnist':
            if args.mode == 'model_heter':
                if i<7:
                    args.out_channels = 18
                elif i>=7 and i<14:
                    args.out_channels = 20
                else:
                    args.out_channels = 22
            else:
                args.out_channels = 20

            local_model = CNNMnist(args=args)

        elif args.dataset == 'femnist':
            if args.mode == 'model_heter':
                if i<7:
                    args.out_channels = 18
                elif i>=7 and i<14:
                    args.out_channels = 20
                else:
                    args.out_channels = 22
            else:
                args.out_channels = 20
            local_model = CNNFemnist(args=args)

        elif args.dataset == 'cifar100' or args.dataset == 'cifar10':
            if args.mode == 'model_heter':
                if i<10:
                    args.stride = [1,4]
                else:
                    args.stride = [2,2]
            else:
                args.stride = [2, 2]
            resnet = resnet18(args, pretrained=False, num_classes=args.num_classes)
            initial_weight = model_zoo.load_url(model_urls['resnet18'])
            local_model = resnet
            initial_weight_1 = local_model.state_dict()
            for key in initial_weight.keys():
                if key[0:3] == 'fc.' or key[0:5]=='conv1' or key[0:3]=='bn1':
                    initial_weight[key] = initial_weight_1[key]

            local_model.load_state_dict(initial_weight)

        local_model.to(args.device)
        #훈련 및 리스트에 추가
        local_model.train()
        local_model_list.append(local_model)

    if args.mode == 'task_heter':
        FedProto_taskheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list)
    else:
        FedProto_modelheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list)


# python federated_main.py --mode task_heter --dataset mnist --num_classes 10 --num_users 20 --ways 3 --shots 100 --stdev 2 --rounds 100 --train_shots_max 110 --ld 1

# For all users (with protos), mean of test acc is 0.98301, std of test acc is 0.00299
# For all users (w/o protos), mean of test acc is 0.98326, std of test acc is 0.00242
# For all users (with protos), mean of proto loss is 0.02729, std of test acc is 0.00846