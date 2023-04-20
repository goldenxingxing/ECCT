import torch
from datasets import DCC_Data
from args import args_parser
import wandb
from utils_train import *
from nodes import DCCClientNode, DCCServerNode

 
if __name__ == '__main__':

    args = args_parser()

    #TODO
    # set random seed
    setup_seed(args.random_seed)

    # args.whether_asyn_update = 'asyn_epoch'
    # args.whether_partial_select = 'partial'
    # args.server_kd_method = 'filtr_kd' #'wo_kd'
    # args.client_kd_method = 'wo_kd'

    args.exp_name = 'alipay_clientsparse'
    args.dataset = 'alipay'
    args.node_num = 2000
    args.framework = 'dcc'
    args.device = 'cuda:1' #'cpu'
    args.output_feature = 'all' #'all'
    # args.T = 4

    if 'cuda' in args.device:
        torch.cuda.set_device(args.device)

    # TODO change the setting names
    setting_name = args.exp_name + '_FedDCC_twostage' + args.server_kd_method + args.client_kd_method \
    + '_select'+str(args.select_ratio) + '_E' + str(args.client_epoch) + 'servE'+ str(args.server_epoch) \
    + '_'  + args.whether_asyn_update + '_Randomseed' + str(args.random_seed)

    # TODO use recorder here
    # initialize visualization tool wandb
    wandb.init(
        config = args,
        project = 'FedDCC result',
        name = setting_name , notes = args.exp_name
    )

    # initialize dataset partition
    data = DCC_Data(args)
    server_trainset, client_trainset, server_linear_feature_columns, server_dnn_feature_columns, client_linear_feature_columns, client_dnn_feature_columns = dataset_loader(args, data)

    # initialize server node
    server_node = DCCServerNode(-1, data.train_loader, server_trainset, args, server_linear_feature_columns, server_dnn_feature_columns)
    # TODO
    # server_node.model.load_state_dict(torch.load('/home/lzx/Device_cloud_collaboration/my_code/outputs/resnet56_server_init.pth'))

    # initialize the client nodes
    client_node = {}
    for i in range(args.node_num): 
        client_node[i] = DCCClientNode(i, data.train_loader[i], client_trainset, args, client_linear_feature_columns, client_dnn_feature_columns) 
        client_node[i].model.load_state_dict(client_node[0].model.state_dict())

    # distributed training
    final_metric_recorder_client = {'clientacc':0.0, 'clientauc':0.0, 'clientmse':0.0, 'clientf1score':0.0}
    final_metric_recorder_server = {'serveracc':0.0, 'serverauc':0.0, 'servermse':0.0, 'serverf1score':0.0}
    best_metric_recorder_client = {'clientacc':0.0, 'clientauc':0.0, 'clientmse':3.0, 'clientf1score':0.0}
    best_metric_recorder_server = {'serveracc':0.0, 'serverauc':0.0, 'servermse':3.0, 'serverf1score':0.0}
    metric = ['acc', 'auc', 'mse', 'f1score']
    for rounds in range(args.T):
        print(setting_name)
        print('===============The {:d}-th round==============='.format(rounds + 1))
        lr_scheduler(rounds, client_node, server_node, args)

        # two stage training, first using CE loss only, second using CE + KD losses
        if rounds <= int(args.T/2):
            server_node.kd_method = 'wo_kd'
            for i in range(len(client_node)):
                client_node[i].kd_method = 'wo_kd'
        else:
            server_node.kd_method = args.server_kd_method
            for i in range(len(client_node)):
                client_node[i].kd_method = args.client_kd_method

        # match_list is for the selected clients in this round
        if args.whether_partial_select == 'partial' and rounds != 0:
            match_list = generate_matchlist(client_node, args.select_ratio)
        else:
            match_list = [j for j in range(len(client_node))]
        
        # clients local training
        for i in range(len(client_node)):
            # including local CE loss and KD loss using server's logits
            if rounds == 0:
                client_node[i].infer_feature()
                # print('client ', i,  ' first round infer completed')
            else:
                if args.whether_asyn_update == 'asyn_epoch' or args.whether_asyn_update == 'both' :
                    asyn_epoch(client_node)
                client_node[i].train()

        # test on clients
        if rounds == 0:
            metric_recorder_client = {'clientacc':0.0, 'clientauc':0.0, 'clientmse':0.0, 'clientf1score':0.0}
        else:
            metric_recorder_client = DCC_test_globalMetric('client', server_node, client_node, match_list, args)
        print('client: ', metric_recorder_client)
        try:
            wandb.log(metric_recorder_client, step = rounds)
        except:
            pass

        # server training
        server_node.receive(client_node, match_list = match_list)
        server_node.train(match_list)

        # test on server
        metric_recorder_server = DCC_test_globalMetric('server', server_node, client_node, match_list, args)
        print('server: ', metric_recorder_server)
        try:
            wandb.log(metric_recorder_server, step = rounds)
        except:
            pass
        
        # clients receive the global model from the server
        # here load the asyn updates' function for asyn_epoch
        if (args.whether_asyn_update == 'asyn_version' or args.whether_asyn_update == 'both') and rounds !=  0:
            update_version_list = asyn_version(client_node)
        else:
            update_version_list = [1 for _ in range(len(client_node))]
        for i in range(len(client_node)):
            if update_version_list[i] == 1:
                client_node[i].receive(server_node)
    
        # update the best_metric_recorder
        for metric_name in best_metric_recorder_client.keys():
            if 'mse' in metric_name:
                if best_metric_recorder_client[metric_name] > metric_recorder_client[metric_name]:
                    best_metric_recorder_client[metric_name] = metric_recorder_client[metric_name]
            else:
                if best_metric_recorder_client[metric_name] < metric_recorder_client[metric_name]:
                    best_metric_recorder_client[metric_name] = metric_recorder_client[metric_name]
        for metric_name in best_metric_recorder_server.keys():
            if 'mse' in metric_name:
                if best_metric_recorder_server[metric_name] > metric_recorder_server[metric_name]:
                    best_metric_recorder_server[metric_name] = metric_recorder_server[metric_name]
            else:
                if best_metric_recorder_server[metric_name] < metric_recorder_server[metric_name]:
                    best_metric_recorder_server[metric_name] = metric_recorder_server[metric_name]


        # in the last 10 rounds, update the final recorder
        if rounds >= args.T - 10:
            for metric_name in final_metric_recorder_client.keys():
                final_metric_recorder_client[metric_name] += 0.1 * metric_recorder_client[metric_name]
            for metric_name in final_metric_recorder_server.keys():
                final_metric_recorder_server[metric_name] += 0.1 * metric_recorder_server[metric_name]

    best_client_cache = dict()
    for metric_name in best_metric_recorder_client.keys():
        best_client_cache['best'+metric_name] = best_metric_recorder_client[metric_name]
    best_server_cache = dict()
    for metric_name in best_metric_recorder_server.keys():
        best_server_cache['best'+metric_name] = best_metric_recorder_server[metric_name]
    final_client_cache = dict()
    for metric_name in final_metric_recorder_client.keys():
        final_client_cache['final'+metric_name] = final_metric_recorder_client[metric_name]
    final_server_cache = dict()
    for metric_name in final_metric_recorder_server.keys():
        final_server_cache['final'+metric_name] = final_metric_recorder_server[metric_name]

    try:
        wandb.log(best_client_cache)
        wandb.log(best_server_cache)
        wandb.log(final_client_cache)
        wandb.log(final_server_cache)
    except:
        pass

    # save the results to local resp
    torch.save([best_metric_recorder_client, best_metric_recorder_server,
    final_metric_recorder_client, final_metric_recorder_server], '/home/lzx/Device_cloud_collaboration/ctr/output/'+setting_name+'.pth')

