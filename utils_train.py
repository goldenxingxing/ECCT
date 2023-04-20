import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import models
import random
from torch.backends import cudnn
import torchmetrics

class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def value(self):
        return self.total / float(self.steps)

class KL_Loss(nn.Module):
    def __init__(self, temperature=1):
        super(KL_Loss, self).__init__()
        self.T = temperature

    def forward(self, output_batch, teacher_outputs):
        output_batch = F.log_softmax(output_batch / self.T, dim=1)
        teacher_outputs = F.softmax(teacher_outputs / self.T, dim=1) + 10 ** (-7)
        loss = self.T * self.T * nn.KLDivLoss(reduction='batchmean')(output_batch, teacher_outputs)
        return loss


class Binary_CE(nn.Module):
    def __init__(self):
        super(Binary_CE, self).__init__()

    def forward(self, output, target):
        loss = F.binary_cross_entropy(output.squeeze(), target.squeeze().float(), reduction='sum')
        return loss


class CE_Loss(nn.Module):
    def __init__(self, temperature=1):
        super(CE_Loss, self).__init__()
        self.T = temperature

    def forward(self, output_batch, teacher_outputs):
        output_batch = F.log_softmax(output_batch / self.T, dim=1)
        teacher_outputs = F.softmax(teacher_outputs / self.T, dim=1)
        loss = -self.T * self.T * torch.sum(torch.mul(output_batch, teacher_outputs)) / teacher_outputs.size(0)
        return loss


def dataset_loader(args, data):
    if args.dataset in ['alipay', 'avazu']:
        if args.framework == 'fed' and args.input_feature == 'all':
            server_trainset = data.train_set_all
            server_linear_feature_columns = data.linear_feature_columns_all
            server_dnn_feature_columns = data.dnn_feature_columns_all
            client_trainset = data.train_set_all
            client_linear_feature_columns = data.linear_feature_columns_all
            client_dnn_feature_columns = data.dnn_feature_columns_all
        elif args.client_features == 'sparse':
            server_trainset = data.train_set_dense
            server_linear_feature_columns = data.linear_feature_columns_dense
            server_dnn_feature_columns = data.dnn_feature_columns_dense
            # client has the sparse features
            client_trainset = data.train_set_sparse
            client_linear_feature_columns = data.linear_feature_columns_sparse
            client_dnn_feature_columns = data.dnn_feature_columns_sparse
        elif args.client_features == 'dense':
            server_trainset = data.train_set_sparse
            server_linear_feature_columns = data.linear_feature_columns_sparse
            server_dnn_feature_columns = data.dnn_feature_columns_sparse
            # client has the dense features
            client_trainset = data.train_set_dense
            client_linear_feature_columns = data.linear_feature_columns_dense
            client_dnn_feature_columns = data.dnn_feature_columns_dense
    elif args.dataset == 'cifar10':
        server_trainset = data.train_set
        server_linear_feature_columns = None
        server_dnn_feature_columns = None
        client_trainset = data.train_set
        client_linear_feature_columns = None
        client_dnn_feature_columns = None
    return server_trainset, client_trainset, server_linear_feature_columns, server_dnn_feature_columns, client_linear_feature_columns, client_dnn_feature_columns

def init_DCCClient_model(linear_feature_columns, dnn_feature_columns, args):
    #TODO change the input of this funct when data is cifar10
    if args.dataset in ['alipay', 'avazu']:
        model = models.AutoIntClient(args, linear_feature_columns, dnn_feature_columns, device=args.device)
    elif args.dataset == 'cifar10':
        model = models.resnet8_56(10)
    return model

def init_DCCClient_model_small(linear_feature_columns, dnn_feature_columns, args):
    if args.dataset == 'cifar10':
        model = models.resnet5_56(10)
    return model

def init_DCCServer_model(linear_feature_columns, dnn_feature_columns, args):
    if args.dataset in ['alipay', 'avazu']:
        model = models.AutoIntServer(args, linear_feature_columns, dnn_feature_columns, device=args.device)
    elif args.dataset == 'cifar10':
        model = models.resnet56_server(10)
    return model

def init_GKTClient_model(linear_feature_columns, dnn_feature_columns, args):
    if args.dataset in ['alipay', 'avazu']:
        model = models.AutoIntClient_gkt(args, linear_feature_columns, dnn_feature_columns, device=args.device)
    elif args.dataset == 'cifar10':
        model = models.resnet8_56_gkt(10)
    return model

def init_GKTClient_model_small(linear_feature_columns, dnn_feature_columns, args):
    if args.dataset == 'cifar10':
        model = models.resnet5_56_gkt(10)
    return model

def init_GKTServer_model(linear_feature_columns, dnn_feature_columns, args):
    if args.dataset in ['alipay', 'avazu']:
        model = models.AutoIntServer_gkt(args, linear_feature_columns, dnn_feature_columns, device=args.device)
    elif args.dataset == 'cifar10':
        model = models.resnet56_server_gkt(10)
    return model



def load_clientdata(X_batch, args):
    # print(X_batch.shape)
    X_batch_client = X_batch[:, :, :, :args.client_feature_size]
    # print(X_batch_client.shape)
    return X_batch_client

def load_serverdata(X_batch, args):
    X_batch_server = X_batch[:, :, :, args.client_feature_size:]
    return X_batch_server

def load_clientdata_gkt(X_batch, args):
    # print(X_batch.shape)
    if args.data_feature_partition == 'partial':
        X_batch_client = X_batch[:, :, :, :args.client_feature_size]
    elif args.data_feature_partition == 'full':
        X_batch_client = X_batch
    # print(X_batch_client.shape)
    return X_batch_client



def init_optimizer(model, args):
    optimizer = []
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-3) #5e-4
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    return optimizer

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True

def lr_scheduler(rounds, node_list, server_node , args):
    if rounds != 0:
        args.lr *= 0.99 #0.99
        for i in range(len(node_list)):
            node_list[i].args.lr = args.lr
            node_list[i].optimizer.param_groups[0]['lr'] = args.lr
        
        server_node.args.lr = args.lr
        server_node.optimizer.param_groups[0]['lr'] = args.lr

def lr_scheduler_forlocal(epochs, node, args):
    if epochs != 0:
        args.lr *= 0.99
        node.args.lr = args.lr
        node.optimizer.param_groups[0]['lr'] = args.lr


def RS_metrics(metric_name, y, y_pred):
    # y, y_pred = y, y_pred.astype("float64")
    if metric_name == 'acc':
        return torchmetrics.functional.accuracy(torch.from_numpy(np.where(y_pred > 0.5, 1, 0)), y)
        # (y, np.where(y_pred > 0.5, 1, 0))
    elif metric_name == 'auc':
        return torchmetrics.functional.auroc(y_pred, y, task="binary")
    elif metric_name == 'mse':
        return torchmetrics.functional.mean_squared_error(y_pred, y)
    elif metric_name == 'f1score':
        return torchmetrics.functional.f1_score(y_pred, y)
    else:
        raise ValueError

# Current Function: test the overall accuracy over all clients
def DCC_test_globalMetric(output_model, server_node, client_node, match_list, args):
    '''
    output_model: the test output model, server or client, if server, it uses the clients' embeddings;
    match_list: the list of clients, involving in this evaluation
    '''
    pred_buffer = []
    label_buffer = []
    for idx in match_list:
        server_node.model.to(args.device).eval() 
        client_node[idx].model.to(args.device).eval() 

        if args.dataset == 'cifar10':
            num_data = sum([len(server_node.validate_set[idx].dataset) for idx in range(len(client_node))])
            label_buffer.append(num_data)
            correct = 0.0
        with torch.no_grad():
            if output_model == 'client':
                local_data_dict_test = client_node[idx].validate_set
            elif output_model == 'server':
                local_data_dict_test = server_node.validate_set[idx]
            for batch_idx, (samples, labels) in enumerate(local_data_dict_test):
                if output_model == 'client':
                    samples_client, labels_client = samples, labels
                    server_extracted_features, labels_server = client_node[idx].server_extracted_feature_dict_test[batch_idx], client_node[idx].server_labels_dict_test[batch_idx]
                    labels = labels_server
                    server_extracted_features, samples_client, labels = server_extracted_features.to(args.device), samples_client.to(args.device), labels.to(args.device)
                    log_probs, _ = client_node[idx].model(samples_client, server_extracted_features)
                elif output_model == 'server':
                    samples_server, labels_server = samples, labels
                    client_extracted_features, labels_client = server_node.client_extracted_feature_dict_test[idx][batch_idx], server_node.client_labels_dict_test[idx][batch_idx]
                    labels = labels_server
                    samples_server, client_extracted_features, labels = samples_server.to(args.device), client_extracted_features.to(args.device), labels.to(args.device)
                    log_probs, _ = server_node.model(samples_server, client_extracted_features)
                if args.dataset == 'cifar10':
                    pred = log_probs.argmax(dim=1)
                    correct += pred.eq(labels.view_as(pred)).sum().item()
                elif args.dataset == 'alipay':
                    pred_buffer.append(log_probs.cpu().data.numpy())
                    label_buffer.append(labels.cpu().data.numpy())
        if args.dataset == 'cifar10':
            pred_buffer.append(correct)
    if args.dataset == 'cifar10':
        avg_client_globalAcc = sum(pred_buffer)/len(label_buffer)
        return avg_client_globalAcc
    elif args.dataset == 'alipay':
        metric = ['acc', 'auc', 'mse', 'f1score']
        metric_recorder = dict()
        pred_buffer = np.concatenate(pred_buffer).astype("float64")
        pred_buffer = np.concatenate(pred_buffer).astype("float64")
        label_buffer = np.concatenate(label_buffer).astype("int")
        label_buffer = np.concatenate(label_buffer).astype("int")
        for metric_name in metric:
            metric_recorder[output_model+metric_name] = RS_metrics(metric_name, torch.from_numpy(label_buffer), torch.from_numpy(pred_buffer))
        
        return metric_recorder


def GKT_test_globalMetric(output_model, algorithm, server_node, client_node, match_list, args):
    '''
    output_model: the test output model, server or client, if server, it uses the clients' embeddings;
    match_list: the list of clients, involving in this evaluation
    '''
    pred_buffer = []
    label_buffer = []
    for idx in match_list:
        server_node.model.eval() 
        client_node[idx].model.eval() 

        if args.dataset == 'cifar10':
            num_data = sum([len(server_node.validate_set[idx].dataset) for idx in range(len(client_node))])
            label_buffer.append(num_data)
            correct = 0.0
        with torch.no_grad():
            local_data_dict_test = server_node.validate_set[idx]
            for batch_idx, (samples, labels) in enumerate(local_data_dict_test):
                samples = samples.to(client_node[idx].model.device)
                labels = labels.to(client_node[idx].model.device)
                log_probs, client_extracted_features = client_node[idx].model(samples)

                if output_model == 'client':
                    log_probs = log_probs
                elif output_model == 'server':
                    if algorithm == 'fedgkt':
                        log_probs = server_node.model(client_extracted_features)
                    elif algorithm == 'fedavg':
                        log_probs, _ = server_node.model(samples)

                if args.dataset == 'cifar10':
                    pred = log_probs.argmax(dim=1)
                    correct += pred.eq(labels.view_as(pred)).sum().item()
                elif args.dataset == 'alipay':
                    pred_buffer.append(log_probs.cpu().data.numpy())
                    label_buffer.append(labels.cpu().data.numpy())

        if args.dataset == 'cifar10':
            pred_buffer.append(correct)

    if args.dataset == 'cifar10':
        avg_client_globalAcc = sum(pred_buffer)/len(label_buffer)
        return avg_client_globalAcc
    elif args.dataset == 'alipay':
        metric = ['acc', 'auc', 'mse', 'f1score']
        metric_recorder = dict()
        pred_buffer = np.concatenate(pred_buffer).astype("float64")
        pred_buffer = np.concatenate(pred_buffer).astype("float64")
        label_buffer = np.concatenate(label_buffer).astype("int")
        label_buffer = np.concatenate(label_buffer).astype("int")

        for metric_name in metric:
            metric_recorder[output_model+metric_name] = RS_metrics(metric_name, torch.from_numpy(label_buffer), torch.from_numpy(pred_buffer))
        
        return metric_recorder


def ServerOnly_test_globalMetric(server_node, match_list):
    '''
    output_model: the test output model, server or client, if server, it uses the clients' embeddings;
    match_list: the list of clients, involving in this evaluation
    '''
    pred_buffer = []
    label_buffer = []
    for idx in match_list:
        server_node.model.eval() 

        with torch.no_grad():
            local_data_dict_test = server_node.validate_set[idx]
            for batch_idx, (samples, labels) in enumerate(local_data_dict_test):

                samples = samples.to(server_node.model.device)
                labels = labels.to(server_node.model.device)
                log_probs = server_node.model(samples)

                pred_buffer.append(log_probs.cpu().data.numpy())
                label_buffer.append(labels.cpu().data.numpy())

    metric = ['acc', 'auc', 'mse', 'f1score']
    metric_recorder = dict()
    pred_buffer = np.concatenate(pred_buffer).astype("float64")
    pred_buffer = np.concatenate(pred_buffer).astype("float64")
    label_buffer = np.concatenate(label_buffer).astype("int")
    label_buffer = np.concatenate(label_buffer).astype("int")

    for metric_name in metric:
        metric_recorder['server'+metric_name] = RS_metrics(metric_name, torch.from_numpy(label_buffer), torch.from_numpy(pred_buffer))
    
    return metric_recorder

def asyn_epoch(client_node):
    # epoch_list = [0, 1, 2, 3, 4, 5]
    # epoch_list = [0, 1, 2]
    epoch_list = [0, 2, 4, 6]
    client_epoch_list = np.random.choice(epoch_list, len(client_node), replace = True)
    for i in range(len(client_node)):
        client_node[i].client_epoch = int(client_epoch_list[i])
    return

def asyn_version(client_node):
    update_or_not = [0, 1]
    update_version_list = np.random.choice(update_or_not, len(client_node), replace = True)
    return update_version_list

def generate_matchlist(client_node, ratio = 0.5):
    candidate_list = [i for i in range(len(client_node))]
    select_num = int(ratio * len(client_node))
    match_list = np.random.choice(candidate_list, select_num, replace = False).tolist()
    return match_list