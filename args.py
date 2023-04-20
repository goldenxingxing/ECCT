import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # noniid setting
    parser.add_argument('--iid', type=int, default=0,  # 选择iid/non-iid #todo 0
                        help='set 1 for iid')

    # Total
    parser.add_argument('--task', type=str, default='ctr',
                        help="training task, ctr or cvr, for alipay")
    parser.add_argument('--device', type=str, default='0',
                        help="device: {cuda, cpu}")
    parser.add_argument('--framework', type=str, default='fed',
                        help="dcc or fed")
    parser.add_argument('--node_num', type=int, default=20, # 200
                        help="Number of nodes")
    parser.add_argument('--T', type=int, default=100,
                        help="Number of rounds: T")
    parser.add_argument('--client_epoch', type=int, default=1, # 3, 1
                        help="Number of local epochs ")
    parser.add_argument('--server_epoch', type=int, default=5,
                        help="Number of server epoch ")
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help="Type of algorithms:{mnist, cifar10,cifar100, fmnist}") 
    parser.add_argument('--whether_training_on_client', default=1, type=int)
    parser.add_argument('--whether_asyn_update', default='none', type=str,
                        help='asyn updates for clients: {none (syn update), asyn_epoch (asyn local epochs), asyn_version (asyn model/embedding version), both}')
    parser.add_argument('--whether_partial_select', default='full', type=str,
                        help='whether partial select client nodes, full or partial')
    parser.add_argument('--select_ratio', default=0.5, type=float,
                        help='select ratio of participated clients')
    parser.add_argument('--exp_name', default='_', type=str,
                        help='exp name, w.r.t the setting name')
    parser.add_argument('--client_feature_size', default=10, type=int,
                        help='client feature size 32 for full')

    
    # Model
    parser.add_argument('--global_model', type=str, default='mlp',
                        help='Type of global model: {mlp, LeNet5,CNN, ResNet18}')
    parser.add_argument('--local_model', type=str, default='CNN',
                        help='Type of local model: {CNN, ResNet8, AlexNet}')
    parser.add_argument('--client_model_type', type=str, default='homo',
                        help='homo model or heter model setups for clients models')

    # Data
    parser.add_argument('--batchsize', type=int, default=128,
                        help="batchsize")
    parser.add_argument('--server_valid_ratio', type=float, default=0.3,
                    help="the ratio of validate set in the central server")  
    parser.add_argument('--client_valid_ratio', type=float, default=0.3,
                    help="the ratio of validate set in the clients")                   
    parser.add_argument('--dirichlet_alpha', type=float, default=0.5, 
                    help="dirichlet_alpha")

    # Features
    parser.add_argument('--client_features', type=str, default='sparse',
                        help="client_features for RS, sparse/dense/all")
    parser.add_argument('--server_features', type=str, default='dense',
                        help="server_features for RS, sparse/dense/all/none")
    parser.add_argument('--data_feature_partition', type=str, default='full',
                    help="data feature partition for clients, partial or full, used for FedAvg or GKT")
    parser.add_argument('--input_feature', type=str, default='all', 
                    help="input feature of a model, all, sparse, dense")
    parser.add_argument('--output_feature', type=str, default='all', 
                    help="output feature of the last layer, all, sparse, dense")

    # Optim
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help="optimizer: {sgd, adam}")
    parser.add_argument('--lr', type=float, default=0.01,  # 0.08
                        help='learning rate')
    parser.add_argument('--stop_decay', type=int, default=150,
                        help='round when learning rate stop decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--random_seed', type=int, default=10,
                        help="random seed for dir distribution")
    parser.add_argument('--temperature', type=float, default=1.0,
                        help="the temperature for KD loss")
    parser.add_argument('--server_kd_method', type=str, default='filtr_kd',
                        help="the method for knowledge distillation: wo_kd, kd, filt_kd, calib_kd")
    parser.add_argument('--client_kd_method', type=str, default='wo_kd',
                        help="the method for knowledge distillation: wo_kd, kd, filt_kd, calib_kd")
    parser.add_argument('--alpha', type=float, default=1.0, 
                        help="hyperparam for KD loss and CE loss") 
    args = parser.parse_args()
    return args
