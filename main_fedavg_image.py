import torch
from datasets import DCC_Data
from nodes import FedAvgServerNode, FedAvgClientNode
from args import args_parser
from utils_train import *
import os
import wandb

if __name__ == '__main__':

    args = args_parser()

    # set random seed
    setup_seed(args.random_seed)

    # print_message(args)
    # assign GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    torch.cuda.set_device('cuda:'+args.device)

    setting_name = args.exp_name + '_FedAvg' + '_node_num'+ str(args.node_num)+'_select'+str(args.select_ratio) + '_dir' + str(args.dirichlet_alpha) + '_E' + str(args.client_epoch) + '_feature' + args.data_feature_partition + '_Randomseed' + str(args.random_seed)

    # initialize visualization tool wandb
    wandb.init(
        config = args,
        project = 'FedDCC result',
        name = setting_name , notes = args.exp_name
    )

    # initialize dataset partition
    # the data partition is the same as DCC but the 'load_data' function is different
    data = DCC_Data(args)

    server_trainset, client_trainset, server_linear_feature_columns, server_dnn_feature_columns, client_linear_feature_columns, client_dnn_feature_columns = dataset_loader(args, data)


    # Initialize server and clients' nodes
    server_node = FedAvgServerNode(-1, data.train_loader, server_trainset, args, server_linear_feature_columns, server_dnn_feature_columns)
    # server_node.model.load_state_dict(torch.load('/home/lzx/Device_cloud_collaboration/my_code/outputs/resnet8_56_gkt_init.pth'))

    # initialize the client nodes
    client_node = {}
    for i in range(args.node_num): 
        client_node[i] = FedAvgClientNode(i, data.train_loader[i], client_trainset, args, client_linear_feature_columns, client_dnn_feature_columns) 
        client_node[i].model.load_state_dict(server_node.model.state_dict())

    # distributed training
    loss_list = []
    for i in range(len(client_node)):
        loss_list.append(0.0)

    server_acc_recorder = RunningAverage()
    client_acc_recorder = RunningAverage()
    for rounds in range(args.T):
        print(setting_name)
        print('===============The {:d}-th round==============='.format(rounds + 1))
        lr_scheduler(rounds, client_node, server_node, args)
        # match_list is for the selected clients in this round
        if args.whether_partial_select == 'partial':
            match_list = generate_matchlist(client_node, args.select_ratio)
        else:
            match_list = [j for j in range(len(client_node))]

        # clients local training
        every_client_acc_recorder = RunningAverage()
        # here load the asyn updates' function for asyn_epoch
        if args.whether_asyn_update == 'asyn_epoch' or args.whether_asyn_update == 'both' :
            asyn_epoch(client_node)
        for i in range(len(client_node)):
            client_node[i].train()
            test_acc = client_node[i].test()

        # update clients' test acc
            print('client ', i,  ', test acc is', test_acc)
            every_client_acc_recorder.update(test_acc)
        try:
            print('average clients acc ', every_client_acc_recorder.value())
        except:
            pass
        try:
            wandb.log({'client_testacc': every_client_acc_recorder.value()}, step = rounds)
        except:
            pass
        if rounds >= args.T - 10:
            client_acc_recorder.update(every_client_acc_recorder.value())

        # central server receive models from clients
        server_node.receive(client_node, match_list = match_list)
        # server average the models
        server_node.average(match_list)
        test_acc = server_node.test()
        print('server ', 'test acc is', test_acc)
        try:
            wandb.log({'server_testacc': test_acc}, step = rounds)
        except:
            pass

        # record the last 10 rounds acc for server
        if rounds >= args.T - 10:
            server_acc_recorder.update(test_acc)

        # clients receive the global model from the server
        # here load the asyn updates' function for asyn_epoch
        if args.whether_asyn_update == 'asyn_version' or args.whether_asyn_update == 'both' :
            update_version_list = asyn_version(client_node)
        else:
            update_version_list = [1 for _ in range(len(client_node))]
        for i in range(len(client_node)):
            if update_version_list[i] == 1:
                client_node[i].receive(server_node)

    try:
        wandb.log({'server_final_testacc': server_acc_recorder.value(),
        'clients_final_global_testacc': client_acc_recorder.value()})
    except:
        pass