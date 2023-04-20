from calendar import c
import copy
from math import ceil
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import torch.nn.functional as F
# from Data import DatasetSplit
from datasets import DatasetSplit
from utils_train import *



def weights_zero(model):
    for p in model.parameters():
        if p.data is not None:
            p.data.detach_()
            p.data.zero_()

class Node(object):
    def __init__(self, num_id, local_data, train_set, args):
        self.num_id = num_id
        self.args = args
        self.node_num = self.args.node_num
        self.client_epoch = args.client_epoch
        self.server_epoch = args.server_epoch
        if num_id == -1:
            valid_ratio = args.server_valid_ratio
        else:
            valid_ratio = args.client_valid_ratio

        if num_id != -1:
            self.local_data, self.validate_set = self.split_node_dataset(local_data, train_set, valid_ratio)
        else:
            # if the node is the central server, then split the its clients' data
            self.local_data = dict()
            self.validate_set = dict()
            for client_idx in local_data.keys():
                self.local_data[client_idx], self.validate_set[client_idx] =  self.split_node_dataset(local_data[client_idx], train_set, valid_ratio)
        
    def train_val_split(self, idxs, train_set, valid_ratio): # local data index, trainset

        # np.random.shuffle(idxs)

        validate_size = valid_ratio * len(idxs)
        # print(len(idxs))

        idxs_test = idxs[:ceil(validate_size)]
        idxs_train = idxs[ceil(validate_size):]

        train_loader = DataLoader(DatasetSplit(train_set, idxs_train),
                                  batch_size=self.args.batchsize, num_workers=0, shuffle=False)

        test_loader = DataLoader(DatasetSplit(train_set, idxs_test),
                                 batch_size=self.args.batchsize,  num_workers=0, shuffle=False)
        
        return train_loader, test_loader
    
    def split_node_dataset(self, local_data, train_set, valid_ratio):
        if self.args.iid == 1:  # iid
            # self.local_data = local_data
            local_data, validate_set = self.train_val_split(local_data.indices, train_set, valid_ratio)
        else:  # noniid
            local_data, validate_set = self.train_val_split(local_data, train_set, valid_ratio)
        
        return local_data, validate_set


class DCCServerNode(Node):
    def __init__(self, num_id, local_data, train_set, args, linear_feature_columns, dnn_feature_columns):
        super().__init__(num_id, local_data, train_set, args)

        # initialize server model and optimizer
        self.model = init_DCCServer_model(linear_feature_columns, dnn_feature_columns, self.args).to(self.args.device)
        self.optimizer = init_optimizer(self.model, args)

        # initialize server loss functions
        if self.args.dataset == 'cifar10':
            self.criterion_CE = nn.CrossEntropyLoss()
            self.criterion_KD = KL_Loss(self.args.temperature)
        elif args.dataset in ['alipay', 'avazu']:
            self.criterion_CE = Binary_CE()
            self.criterion_KD = Binary_CE()
        self.best_acc = 0.0

        # initialize server dict about the knowledge
        # key: client_index; value: extracted_feature_dict
        self.client_extracted_feature_dict = dict()

        # key: client_index; value: logits_dict
        self.client_logits_dict = dict()

        # key: client_index; value: labels_dict
        self.client_labels_dict = dict()

        # for test
        self.client_extracted_feature_dict_test = dict()
        self.client_labels_dict_test = dict()
        self.server_extracted_feature_dict_test = dict()
        self.server_labels_dict_test = dict()
        self.server_extracted_feature_dict = dict()
        self.server_logits_dict = dict()
        self.server_labels_dict = dict()

        for i in range(self.node_num):
            self.server_extracted_feature_dict_test[i] = dict()
            self.server_labels_dict_test[i] = dict()
            self.server_extracted_feature_dict[i] = dict()
            self.server_logits_dict[i] = dict()
            self.server_labels_dict[i] = dict()

        self.kd_method = self.args.server_kd_method


    def train(self, match_list):

        # train and update
        self.model.train().to(self.args.device)
        for _ in range(self.server_epoch):
            for client_index in match_list:

                local_data_dict = self.local_data[client_index]
                extracted_feature_dict = self.client_extracted_feature_dict[client_index]
                logits_dict = self.client_logits_dict[client_index]
                labels_dict = self.client_labels_dict[client_index]

                # for batch_index in extracted_feature_dict.keys():
                for batch_idx, (samples, labels) in enumerate(local_data_dict):
                    if self.args.dataset == 'cifar10':
                        samples = load_serverdata(samples, self.args)
                    samples, labels = samples.to(self.args.device), labels.to(self.args.device)

                    batch_feature_map_x = extracted_feature_dict[batch_idx].to(self.args.device)
                    output_batch, _ = self.model(samples, batch_feature_map_x)

                    # according to different kd methods, training
                    if self.kd_method == 'kd':
                        batch_logits = logits_dict[batch_idx].to(self.args.device)
                        loss_kd = self.criterion_KD(output_batch, batch_logits).to(self.args.device)
                        loss_true = self.criterion_CE(output_batch, labels).to(self.args.device)
                        loss = loss_kd + self.args.alpha * loss_true

                    elif self.kd_method == 'wo_kd':
                        loss_true = self.criterion_CE(output_batch, labels).to(self.args.device)
                        loss = loss_true
                    elif self.kd_method == 'filtr_kd':

                        loss_true = self.criterion_CE(output_batch, labels).to(self.args.device)

                        batch_logits = logits_dict[batch_idx].to(self.args.device)

                        # filtering logits
                        batch_logits = torch.where(torch.round(batch_logits)==labels, batch_logits, labels.float())
                        loss_kd = self.criterion_KD(output_batch, batch_logits).to(self.args.device)
                        
                        loss = loss_kd + self.args.alpha * loss_true    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
        # after training, update the trainset logits and embeddings
        self.model.to(self.args.device).eval()

        for client_index in self.client_extracted_feature_dict.keys():

            local_data_dict = self.local_data[client_index]
            extracted_feature_dict = self.client_extracted_feature_dict[client_index]
            # for batch_index in extracted_feature_dict.keys():
            for batch_idx, (samples, labels) in enumerate(local_data_dict):
                if self.args.dataset == 'cifar10':
                    samples = load_serverdata(samples, self.args)
                samples, labels = samples.to(self.args.device), labels.to(self.args.device)

                log_probs, extracted_features = self.model(samples, extracted_feature_dict[batch_idx].to(self.args.device))
                self.server_extracted_feature_dict[client_index][batch_idx] = extracted_features.detach()
                self.server_logits_dict[client_index][batch_idx] = log_probs.detach()
                self.server_labels_dict[client_index][batch_idx] = labels.detach()
        
        # update the testset embeddings for clients' local inference
        for client_index in self.client_extracted_feature_dict_test.keys():

            local_data_dict_test = self.validate_set[client_index]
            extracted_feature_dict_test = self.client_extracted_feature_dict_test[client_index]

            for batch_idx, (samples, labels) in enumerate(local_data_dict_test):
                if self.args.dataset == 'cifar10':
                    samples = load_serverdata(samples, self.args)
                test_samples, test_labels = samples.to(self.args.device), labels.to(self.args.device)

                _, extracted_features_test = self.model(test_samples, extracted_feature_dict_test[batch_idx])
                self.server_extracted_feature_dict_test[client_index][batch_idx] = extracted_features_test.detach()
                self.server_labels_dict_test[client_index][batch_idx] = test_labels.detach()
        return

    def receive(self, client_node, match_list):
        # get updated knowledge from server
        for client_idx in match_list:
            self.client_extracted_feature_dict[client_idx] = copy.deepcopy(client_node[client_idx].client_extracted_feature_dict)
            self.client_logits_dict[client_idx] = copy.deepcopy(client_node[client_idx].client_logits_dict)
            self.client_labels_dict[client_idx] = copy.deepcopy(client_node[client_idx].client_labels_dict)
            self.client_extracted_feature_dict_test[client_idx] = copy.deepcopy(client_node[client_idx].client_extracted_feature_dict_test)
            self.client_labels_dict_test[client_idx] = copy.deepcopy(client_node[client_idx].client_labels_dict_test)
        return

    def test(self):
        self.model.cuda().eval()  

        correct = 0.0
        with torch.no_grad():
            for client_index in self.client_extracted_feature_dict_test.keys():

                local_data_dict_test = self.validate_set[client_index]
                extracted_feature_dict_test = self.client_extracted_feature_dict_test[client_index]

                for batch_idx, (images, labels) in enumerate(local_data_dict_test):
                    images = load_serverdata(images, self.args)
                    test_images, test_labels = images.cuda(), labels.cuda()

                    log_probs, _ = self.model(test_images, extracted_feature_dict_test[batch_idx])
                    pred = log_probs.argmax(dim=1)
                    correct += pred.eq(test_labels.view_as(pred)).sum().item()
            
            num_data = sum([len(self.validate_set[idx].dataset) for idx in self.client_extracted_feature_dict_test.keys()])
            test_acc = correct / num_data * 100

        return test_acc

class DCCClientNode(Node):
    def __init__(self, num_id, local_data, train_set, args, linear_feature_columns, dnn_feature_columns):
        super().__init__(num_id, local_data, train_set, args)
        self.model = init_DCCClient_model(linear_feature_columns, dnn_feature_columns, args).to(self.args.device)
        self.optimizer = init_optimizer(self.model, args)

        # initialize client loss functions
        if self.args.dataset == 'cifar10':
            self.criterion_CE = nn.CrossEntropyLoss()
            self.criterion_KD = KL_Loss(self.args.temperature)
        elif args.dataset in ['alipay', 'avazu']:
            self.criterion_CE = Binary_CE()
            self.criterion_KD = Binary_CE()
        self.best_acc = 0.0

        # initialize server dict about the knowledge
        self.client_extracted_feature_dict = dict()
        self.client_logits_dict = dict()
        self.client_labels_dict = dict()

        self.server_extracted_feature_dict = dict()
        self.server_logits_dict = dict()

        # for test
        self.client_extracted_feature_dict_test = dict()
        self.client_labels_dict_test = dict()

        self.server_extracted_feature_dict_test = dict()
        self.server_labels_dict_test = dict()

        self.kd_method = self.args.client_kd_method

    def train(self):

        self.model.to(self.args.device).train()
        # train and update
        epoch_loss = []
        for _ in range(self.client_epoch):
            batch_loss = []
            for batch_idx, (samples, labels) in enumerate(self.local_data):
                if self.args.dataset == 'cifar10':
                    samples = load_clientdata(samples, self.args)
                samples, labels = samples.to(self.args.device), labels.to(self.args.device)

                log_probs, _ = self.model(samples, self.server_extracted_feature_dict[batch_idx].to(self.args.device))
                loss_true = self.criterion_CE(log_probs, labels)

                if self.kd_method == 'kd':
                    if len(self.server_logits_dict) != 0:
                        large_model_logits = self.server_logits_dict[batch_idx].to(self.args.device)
                        loss_kd = self.criterion_KD(log_probs, large_model_logits)
                        loss = loss_true + self.args.alpha * loss_kd
                    else:
                        loss = loss_true
                elif self.kd_method == 'wo_kd':
                        loss = loss_true
                elif self.kd_method == 'filtr_kd':
                    if len(self.server_logits_dict) != 0:
                        large_model_logits = self.server_logits_dict[batch_idx].to(self.args.device)

                        # filtering logits
                        large_model_logits = torch.where(torch.round(large_model_logits)==labels, large_model_logits, labels.float())
                        loss_kd = self.criterion_KD(log_probs, large_model_logits)
                        loss = loss_true + self.args.alpha * loss_kd
                    else:
                        loss = loss_true

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # after training, update the logits and embeddings
        self.model.to(self.args.device).eval()

        for batch_idx, (samples, labels) in enumerate(self.local_data):
            if self.args.dataset == 'cifar10':
                samples = load_clientdata(samples, self.args)
            samples, labels = samples.to(self.args.device), labels.to(self.args.device)

            log_probs, extracted_features = self.model(samples, self.server_extracted_feature_dict[batch_idx].to(self.args.device))
            self.client_extracted_feature_dict[batch_idx] = extracted_features.detach()
            self.client_logits_dict[batch_idx] = log_probs.detach()
            self.client_labels_dict[batch_idx] = labels.detach()

        for batch_idx, (samples, labels) in enumerate(self.validate_set):
            if self.args.dataset == 'cifar10':
                samples = load_clientdata(samples, self.args)
            test_samples, test_labels = samples.to(self.args.device), labels.to(self.args.device)

            _, extracted_features_test = self.model(test_samples, self.server_extracted_feature_dict_test[batch_idx].to(self.args.device))
            self.client_extracted_feature_dict_test[batch_idx] = extracted_features_test.detach()
            self.client_labels_dict_test[batch_idx] = test_labels.detach()
        return

    def infer_feature(self):

        # after training, update the logits and embeddings
        self.model.to(self.args.device).train()

        for batch_idx, (samples, labels) in enumerate(self.local_data):
            if self.args.dataset == 'cifar10':
                samples = load_clientdata(samples, self.args)
            samples, labels = samples.to(self.args.device), labels.to(self.args.device)

            extracted_features = self.model.forward_infer_feature(samples)
            self.client_extracted_feature_dict[batch_idx] = extracted_features.detach()
            self.client_labels_dict[batch_idx] = labels.detach()

        for batch_idx, (samples, labels) in enumerate(self.validate_set):
            if self.args.dataset == 'cifar10':
                samples = load_clientdata(samples, self.args)
            test_samples, test_labels = samples.to(self.args.device), labels.to(self.args.device)

            extracted_features_test = self.model.forward_infer_feature(test_samples)
            self.client_extracted_feature_dict_test[batch_idx] = extracted_features_test.detach()
            self.client_labels_dict_test[batch_idx] = test_labels.detach()

        return

    def receive(self,server_node):
        # get updated knowledge from server
        self.server_extracted_feature_dict = copy.deepcopy(server_node.server_extracted_feature_dict[self.num_id])
        self.server_logits_dict = copy.deepcopy(server_node.server_logits_dict[self.num_id])
        self.server_extracted_feature_dict_test = copy.deepcopy(server_node.server_extracted_feature_dict_test[self.num_id])
        self.server_labels_dict_test = copy.deepcopy(server_node.server_labels_dict_test[self.num_id])
        
        return

    def test(self):
        self.model.cuda().eval()  

        correct = 0.0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.validate_set):
                images = load_clientdata(images, self.args)
                images, labels = images.cuda(), labels.cuda()

                log_probs, _ = self.model(images, self.server_extracted_feature_dict_test[batch_idx].cuda())
                pred = log_probs.argmax(dim=1)
                correct += pred.eq(labels.view_as(pred)).sum().item()
            test_acc = correct / len(self.validate_set.dataset) * 100

        return test_acc

class GKTServerNode(Node):
    def __init__(self, num_id, local_data, train_set, args, linear_feature_columns, dnn_feature_columns):
        super().__init__(num_id, local_data, train_set, args)

        # initialize server model and optimizer
        self.model = init_GKTServer_model(linear_feature_columns, dnn_feature_columns, self.args).to(self.args.device)
        self.optimizer = init_optimizer(self.model, args)

        # initialize server loss functions
        if self.args.dataset == 'cifar10':
            self.criterion_CE = nn.CrossEntropyLoss()
            self.criterion_KD = KL_Loss(self.args.temperature)
        elif args.dataset in ['alipay', 'avazu']:
            self.criterion_CE = Binary_CE()
            self.criterion_KD = Binary_CE()
        self.best_acc = 0.0

        # initialize server dict about the knowledge
        # key: client_index; value: extracted_feature_dict
        self.client_extracted_feature_dict = dict()

        # key: client_index; value: logits_dict
        self.client_logits_dict = dict()

        # key: client_index; value: labels_dict
        self.client_labels_dict = dict()

        # key: client_index; value: labels_dict
        self.server_logits_dict = dict()

        for i in range(self.node_num):
            self.server_logits_dict[i] = dict()

        # for test
        self.client_extracted_feature_dict_test = dict()
        self.client_labels_dict_test = dict()

        self.kd_method = self.args.server_kd_method


    def train(self, match_list):

        # train and update
        self.model.to(self.args.device).train()
        for _ in range(self.server_epoch):
            for client_index in match_list:

                extracted_feature_dict = self.client_extracted_feature_dict[client_index]
                logits_dict = self.client_logits_dict[client_index]
                labels_dict = self.client_labels_dict[client_index]

                # initialize the dict for each client
                self.server_logits_dict[client_index] = dict()

                # for batch_index in extracted_feature_dict.keys():
                for batch_idx in extracted_feature_dict.keys():

                    batch_feature_map_x = extracted_feature_dict[batch_idx].to(self.args.device)
                    labels = labels_dict[batch_idx].to(self.args.device)
                    batch_logits = logits_dict[batch_idx].to(self.args.device)

                    # TODO here is how the server model works
                    output_batch = self.model(batch_feature_map_x)

                    # according to different kd methods, training
                    if self.kd_method == 'kd':

                        loss_kd = self.criterion_KD(output_batch, batch_logits).to(self.args.device)
                        loss_true = self.criterion_CE(output_batch, labels).to(self.args.device)
                        loss = loss_kd + self.args.alpha * loss_true

                    elif self.kd_method == 'wo_kd':
                        loss_true = self.criterion_CE(output_batch, labels).to(self.args.device)
                        loss = loss_true
                    elif self.kd_method == 'calib_kd':
                        # calibrating logits
                        batch_logits_pred = batch_logits.argmax(dim=1)
                        for i in range(len(labels)):
                            cache = batch_logits[i][labels[i]]
                            batch_logits[i][labels[i]] = batch_logits[i][batch_logits_pred[i]]
                            batch_logits[i][batch_logits_pred[i]] = cache

                        loss_kd = self.criterion_KD(output_batch, batch_logits).to(self.args.device)
                        loss_true = self.criterion_CE(output_batch, labels).to(self.args.device)
                        loss = loss_kd + self.args.alpha * loss_true    

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

        # after training, update the trainset logits
        self.model.to(self.args.device).eval()

        for client_index in self.client_extracted_feature_dict.keys():

            extracted_feature_dict = self.client_extracted_feature_dict[client_index]

            # for batch_index in extracted_feature_dict.keys():
            for batch_idx in extracted_feature_dict.keys():
                batch_feature_map_x = extracted_feature_dict[batch_idx].to(self.args.device)
                
                log_probs = self.model(batch_feature_map_x)
                self.server_logits_dict[client_index][batch_idx] = log_probs.detach()

        return

    def receive(self, client_node, match_list):
        # get updated knowledge from server
        for client_idx in match_list:

            self.client_extracted_feature_dict[client_idx] = copy.deepcopy(client_node[client_idx].client_extracted_feature_dict)
            self.client_logits_dict[client_idx] = copy.deepcopy(client_node[client_idx].client_logits_dict)
            self.client_labels_dict[client_idx] = copy.deepcopy(client_node[client_idx].client_labels_dict)
            self.client_extracted_feature_dict_test[client_idx] = copy.deepcopy(client_node[client_idx].client_extracted_feature_dict_test)
            self.client_labels_dict_test[client_idx] = copy.deepcopy(client_node[client_idx].client_labels_dict_test)

        return

    def test(self):
        self.model.cuda().eval()  

        correct = 0.0
        with torch.no_grad():
            for client_index in self.client_extracted_feature_dict_test.keys():

                extracted_feature_dict_test = self.client_extracted_feature_dict_test[client_index]
                labels_dict = self.client_labels_dict_test[client_index]

                # for batch_index in extracted_feature_dict.keys():
                for batch_idx in extracted_feature_dict_test.keys():
                    batch_feature_map_x = extracted_feature_dict_test[batch_idx].cuda()
                    labels = labels_dict[batch_idx].cuda()

                    log_probs= self.model(batch_feature_map_x)
                    pred = log_probs.argmax(dim=1)
                    correct += pred.eq(labels.view_as(pred)).sum().item()
            
            num_data = sum([len(self.validate_set[idx].dataset) for idx in self.client_extracted_feature_dict_test.keys()])
            test_acc = correct / num_data * 100

        return test_acc

class GKTClientNode(Node):
    def __init__(self, num_id, local_data, train_set, args, linear_feature_columns, dnn_feature_columns):
        super().__init__(num_id, local_data, train_set, args)

        self.model = init_GKTClient_model(linear_feature_columns, dnn_feature_columns, self.args).to(self.args.device)
        self.optimizer = init_optimizer(self.model, args)

        # initialize client loss functions
        if self.args.dataset == 'cifar10':
            self.criterion_CE = nn.CrossEntropyLoss()
            self.criterion_KD = KL_Loss(self.args.temperature)
        elif args.dataset in ['alipay', 'avazu']:
            self.criterion_CE = Binary_CE()
            self.criterion_KD = Binary_CE()
        self.best_acc = 0.0

        # initialize server dict about the knowledge
        self.client_extracted_feature_dict = dict()
        self.client_logits_dict = dict()
        self.client_labels_dict = dict()

        self.server_logits_dict = dict()

        # for test
        self.client_extracted_feature_dict_test = dict()
        self.client_labels_dict_test = dict()

        self.kd_method = self.args.client_kd_method

    def train(self):

        self.model.to(self.args.device).train()
        # train and update
        epoch_loss = []
        for _ in range(self.client_epoch):
            batch_loss = []
            for batch_idx, (samples, labels) in enumerate(self.local_data):
                if self.args.dataset == 'cifar10':
                    samples = load_clientdata_gkt(samples, self.args)
                samples, labels = samples.to(self.args.device), labels.to(self.args.device)

                log_probs, _ = self.model(samples)
                loss_true = self.criterion_CE(log_probs, labels)

                if self.kd_method == 'kd':
                    if len(self.server_logits_dict) != 0:
                        large_model_logits = self.server_logits_dict[batch_idx].to(self.args.device)
                        loss_kd = self.criterion_KD(log_probs, large_model_logits)
                        loss = loss_true + self.args.alpha * loss_kd
                    else:
                        loss = loss_true
                elif self.kd_method == 'wo_kd':
                        loss = loss_true
                elif self.kd_method == 'calib_kd':
                    if len(self.server_logits_dict) != 0:
                        large_model_logits = self.server_logits_dict[batch_idx].to(self.args.device)

                        # calibrating logits
                        batch_logits_pred = large_model_logits.argmax(dim=1)
                        for i in range(len(labels)):
                            cache = large_model_logits[i][labels[i]]
                            large_model_logits[i][labels[i]] = large_model_logits[i][batch_logits_pred[i]]
                            large_model_logits[i][batch_logits_pred[i]] = cache

                        loss_kd = self.criterion_KD(log_probs, large_model_logits)
                        loss = loss_true + self.args.alpha * loss_kd
                    else:
                        loss = loss_true

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # after training, update the logits and embeddings
        self.model.to(self.args.device).eval()

        for batch_idx, (samples, labels) in enumerate(self.local_data):
            if self.args.dataset == 'cifar10':
                samples = load_clientdata_gkt(samples, self.args)
            samples, labels = samples.to(self.args.device), labels.to(self.args.device)
            log_probs, extracted_features = self.model(samples)
            self.client_extracted_feature_dict[batch_idx] = extracted_features.detach()
            self.client_logits_dict[batch_idx] = log_probs.detach()
            self.client_labels_dict[batch_idx] = labels.detach()

        for batch_idx, (samples, labels) in enumerate(self.validate_set):
            if self.args.dataset == 'cifar10':
                samples = load_clientdata_gkt(samples, self.args)
            test_samples, test_labels = samples.to(self.args.device), labels.to(self.args.device)
            _, extracted_features_test = self.model(test_samples)
            self.client_extracted_feature_dict_test[batch_idx] = extracted_features_test.detach()
            self.client_labels_dict_test[batch_idx] = test_labels.detach()
        return

    def receive(self,server_node):
        # get updated knowledge from server
        self.server_logits_dict = copy.deepcopy(server_node.server_logits_dict[self.num_id])
        return

    def test(self):
        self.model.cuda().eval()  

        correct = 0.0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.validate_set):
                images = load_clientdata_gkt(images, self.args)
                images, labels = images.cuda(), labels.cuda()

                log_probs, _ = self.model(images)
                pred = log_probs.argmax(dim=1)
                correct += pred.eq(labels.view_as(pred)).sum().item()
            test_acc = correct / len(self.validate_set.dataset) * 100

        return test_acc

class FedAvgServerNode(Node):
    def __init__(self, num_id, local_data, train_set, args, linear_feature_columns, dnn_feature_columns):
        super().__init__(num_id, local_data, train_set, args)

        # initialize server model
        # self.model = init_GKTClient_model(linear_feature_columns, dnn_feature_columns, self.args).to(self.args.device)
        self.model = init_GKTClient_model(linear_feature_columns, dnn_feature_columns, self.args)
        self.optimizer = init_optimizer(self.model, args)

        # initialize clients models
        self.client_models = dict()

    def average(self, match_list):
        self.Dict = self.model.state_dict()

        weights_zero(self.model)

        for key in self.Dict.keys():
            for i in match_list:
                self.Dict[key] = self.Dict[key] + self.client_models[i][key]
            self.Dict[key] = self.Dict[key] / float(len(match_list))
        self.model.load_state_dict(copy.deepcopy(self.Dict))
        return

    def receive(self, client_node, match_list):
        # get models from clients
        for client_idx in match_list:
            self.client_models[client_idx] = copy.deepcopy(client_node[client_idx].model.state_dict())
        return

    def test(self):
        self.model.cuda().eval()  

        correct = 0.0
        with torch.no_grad():
            for client_index in range(len(self.validate_set)):
                for batch_idx, (images, labels) in enumerate(self.validate_set[client_index]):

                    images = load_clientdata_gkt(images, self.args)
                    images, labels = images.cuda(), labels.cuda()

                    log_probs, _ = self.model(images)
                    pred = log_probs.argmax(dim=1)
                    correct += pred.eq(labels.view_as(pred)).sum().item()
            
            num_data = sum([len(self.validate_set[idx].dataset) for idx in range(len(self.validate_set))])
            test_acc = correct / num_data * 100
        return test_acc

class FedAvgClientNode(Node):
    def __init__(self, num_id, local_data, train_set, args, linear_feature_columns, dnn_feature_columns):
        super().__init__(num_id, local_data, train_set, args)
        self.model = init_GKTClient_model(linear_feature_columns, dnn_feature_columns, self.args)
        self.optimizer = init_optimizer(self.model, args)

        # initialize client loss functions
        if self.args.dataset == 'cifar10':
            self.criterion_CE = nn.CrossEntropyLoss()
        elif args.dataset in ['alipay', 'avazu']:
            self.criterion_CE = Binary_CE()
        self.best_acc = 0.0

    def train(self):
        self.model.train().to(self.args.device)
        # self.model.to(self.args.device).train()
        # train and update
        epoch_loss = []
        for _ in range(self.client_epoch):
            batch_loss = []
            for batch_idx, (samples, labels) in enumerate(self.local_data):
                samples, labels = samples.to(self.model.device), labels.to(self.model.device)

                log_probs, _ = self.model(samples)
                loss_true = self.criterion_CE(log_probs, labels)
                loss = loss_true

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return

    def receive(self, server_node):
        # get updated knowledge from server
        self.model.load_state_dict(copy.deepcopy(server_node.model.state_dict()))
        return

    def test(self):
        self.model.cuda().eval()  

        correct = 0.0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.validate_set):
                images = load_clientdata_gkt(images, self.args)
                images, labels = images.cuda(), labels.cuda()

                log_probs, _ = self.model(images)
                pred = log_probs.argmax(dim=1)
                correct += pred.eq(labels.view_as(pred)).sum().item()
            test_acc = correct / len(self.validate_set.dataset) * 100
        return test_acc