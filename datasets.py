import numpy as np
import torch
import torchvision
import torch.utils.data as Data
from torch.utils.data import Dataset
from torchvision import transforms
import copy
from deepctr_utils import *
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,LabelEncoder


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class CustomSubset(torch.utils.data.Subset):
    '''A custom subset class'''
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        dataset.targets = torch.tensor(dataset.targets)
        self.targets = dataset.targets[indices] 
        self.classes = dataset.classes
        self.indices = indices

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]      
        return x, y 

    def __len__(self):
        return len(self.indices)


class DCC_Data(object):
    def __init__(self, args):
        self.args = args
        node_num = args.node_num

        if args.dataset == 'alipay':
            
            # load the data
            file_path = './alipay_member_data.csv'
            data = pd.read_csv(file_path)

            # use 1/4 of the full dataset
            data = data[:500000]

            sparse_features = data.iloc[::,7:18].columns.values.tolist() + data.iloc[::,1:3].columns.values.tolist()

            # TODO change the dense feature, here is 3 features
            dense_features = data.iloc[::,22:].columns.values.tolist()

            scaler = MinMaxScaler(feature_range=(0,1))
            encoder = LabelEncoder()
            for feat in sparse_features:
                data[feat] = encoder.fit_transform(data[feat])
            data[dense_features] = scaler.fit_transform(data[dense_features])

            # preprocess the data
            if args.task == 'ctr':
                target = ['ctr_label']
            elif args.task == 'cvr':
                target = ['cvr_label']
            else:
                raise ValueError
            # TODO change the target
            # target = ['cvr_label']
            data.sort_values('user_id', inplace=True)

            ### Here, we only use sparse features to create the dataset ###
            fixlen_feature_columns_sparse = [SparseFeat(feat, data[feat].nunique()) for feat in sparse_features]
            dnn_feature_columns_sparse = []
            linear_feature_columns_sparse = fixlen_feature_columns_sparse
            feature_names_sparse = get_feature_names(linear_feature_columns_sparse + dnn_feature_columns_sparse)

            train = data
            train_model_input_sparse = {name: train[name] for name in feature_names_sparse}

            feature_index_sparse = build_input_features(
                        linear_feature_columns_sparse + dnn_feature_columns_sparse)

            if isinstance(train_model_input_sparse, dict):
                train_model_input_sparse = [train_model_input_sparse[feature] for feature in feature_index_sparse]

            for i in range(len(train_model_input_sparse)):
                if len(train_model_input_sparse[i].shape) == 1:
                    train_model_input_sparse[i] = np.expand_dims(train_model_input_sparse[i], axis=1)

            train_tensor_data_sparse = Data.TensorDataset(
                torch.from_numpy(
                    np.concatenate(train_model_input_sparse, axis=-1)),
                torch.from_numpy(train[target].values))

            self.train_set_sparse = train_tensor_data_sparse
            self.linear_feature_columns_sparse = linear_feature_columns_sparse
            self.dnn_feature_columns_sparse = dnn_feature_columns_sparse

            ### Here, we only use dense features to create the dataset ###
            fixlen_feature_columns_dense = [DenseFeat(feat,1,) for feat in dense_features]
            dnn_feature_columns_dense = fixlen_feature_columns_dense
            linear_feature_columns_dense = fixlen_feature_columns_dense
            feature_names_dense = get_feature_names(linear_feature_columns_dense + dnn_feature_columns_dense)

            train = data
            train_model_input_dense = {name: train[name] for name in feature_names_dense}

            feature_index_dense = build_input_features(
                        linear_feature_columns_dense + dnn_feature_columns_dense)

            if isinstance(train_model_input_dense, dict):
                train_model_input_dense = [train_model_input_dense[feature] for feature in feature_index_dense]

            for i in range(len(train_model_input_dense)):
                if len(train_model_input_dense[i].shape) == 1:
                    train_model_input_dense[i] = np.expand_dims(train_model_input_dense[i], axis=1)

            train_tensor_data_dense = Data.TensorDataset(
                torch.from_numpy(
                    np.concatenate(train_model_input_dense, axis=-1)),
                torch.from_numpy(train[target].values))

            self.train_set_dense = train_tensor_data_dense
            self.linear_feature_columns_dense = linear_feature_columns_dense
            self.dnn_feature_columns_dense = dnn_feature_columns_dense

            ### Here, we use both dense features and the sparse features ###
            fixlen_feature_columns_all = [SparseFeat(feat,data[feat].nunique()) 
                                    for feat in sparse_features] + [DenseFeat(feat,1,)
                                                                    for feat in dense_features]
            dnn_feature_columns_all = [DenseFeat(feat,1,) for feat in dense_features]
            linear_feature_columns_all = fixlen_feature_columns_all
            feature_names_all = get_feature_names(linear_feature_columns_all + dnn_feature_columns_all)

            train = data
            train_model_input_all = {name: train[name] for name in feature_names_all}

            feature_index_all = build_input_features(
                        linear_feature_columns_all + dnn_feature_columns_all)

            if isinstance(train_model_input_all, dict):
                train_model_input_all = [train_model_input_all[feature] for feature in feature_index_all]

            for i in range(len(train_model_input_all)):
                if len(train_model_input_all[i].shape) == 1:
                    train_model_input_all[i] = np.expand_dims(train_model_input_all[i], axis=1)

            train_tensor_data_all = Data.TensorDataset(
                torch.from_numpy(
                    np.concatenate(train_model_input_all, axis=-1)),
                torch.from_numpy(train[target].values))

            self.train_set_all = train_tensor_data_all
            self.linear_feature_columns_all = linear_feature_columns_all
            self.dnn_feature_columns_all = dnn_feature_columns_all

            ### Generate the train loader ###
            self.train_loader = build_trainloader_alipay_uniform(node_num)


        elif args.dataset == 'avazu':
            # load the data
            file_path = './avazu/200ksample.csv'
            data = pd.read_csv(file_path)
            data = data.drop(columns=['Unnamed: 0'])
            data.set_index('hour',inplace=True)
            sparse_features = data.iloc[::,2:15].columns.values.tolist()
            dense_features = data.iloc[::,15:].columns.values.tolist()

            # print(len(sparse_features), len(dense_features))

            scaler = MinMaxScaler(feature_range=(0,1))
            encoder = LabelEncoder()
            for feat in sparse_features:
                data[feat] = encoder.fit_transform(data[feat])
            data[dense_features] = scaler.fit_transform(data[dense_features])

            # preprocess the data
            target = ['click']
            data.sort_values('device_ip', inplace=True)

            ### Here, we only use sparse features to create the dataset ###
            fixlen_feature_columns_sparse = [SparseFeat(feat,data[feat].nunique()) for feat in sparse_features]
            dnn_feature_columns_sparse = []
            linear_feature_columns_sparse = fixlen_feature_columns_sparse
            feature_names_sparse = get_feature_names(linear_feature_columns_sparse + dnn_feature_columns_sparse)

            train = data
            train_model_input_sparse = {name: train[name] for name in feature_names_sparse}

            feature_index_sparse = build_input_features(
                        linear_feature_columns_sparse + dnn_feature_columns_sparse)

            if isinstance(train_model_input_sparse, dict):
                train_model_input_sparse = [train_model_input_sparse[feature] for feature in feature_index_sparse]

            for i in range(len(train_model_input_sparse)):
                if len(train_model_input_sparse[i].shape) == 1:
                    train_model_input_sparse[i] = np.expand_dims(train_model_input_sparse[i], axis=1)

            train_tensor_data_sparse = Data.TensorDataset(
                torch.from_numpy(
                    np.concatenate(train_model_input_sparse, axis=-1)),
                torch.from_numpy(train[target].values))

            self.train_set_sparse = train_tensor_data_sparse
            self.linear_feature_columns_sparse = linear_feature_columns_sparse
            self.dnn_feature_columns_sparse = dnn_feature_columns_sparse

            ### Here, we only use dense features to create the dataset ###
            fixlen_feature_columns_dense = [DenseFeat(feat,1,) for feat in dense_features]
            dnn_feature_columns_dense = fixlen_feature_columns_dense
            linear_feature_columns_dense = fixlen_feature_columns_dense
            feature_names_dense = get_feature_names(linear_feature_columns_dense + dnn_feature_columns_dense)

            train = data
            train_model_input_dense = {name: train[name] for name in feature_names_dense}

            feature_index_dense = build_input_features(
                        linear_feature_columns_dense + dnn_feature_columns_dense)

            if isinstance(train_model_input_dense, dict):
                train_model_input_dense = [train_model_input_dense[feature] for feature in feature_index_dense]

            for i in range(len(train_model_input_dense)):
                if len(train_model_input_dense[i].shape) == 1:
                    train_model_input_dense[i] = np.expand_dims(train_model_input_dense[i], axis=1)

            train_tensor_data_dense = Data.TensorDataset(
                torch.from_numpy(
                    np.concatenate(train_model_input_dense, axis=-1)),
                torch.from_numpy(train[target].values))

            self.train_set_dense = train_tensor_data_dense
            self.linear_feature_columns_dense = linear_feature_columns_dense
            self.dnn_feature_columns_dense = dnn_feature_columns_dense

            ### Here, we use both dense features and the sparse features ###
            fixlen_feature_columns_all = [SparseFeat(feat,data[feat].nunique()) 
                                    for feat in sparse_features] + [DenseFeat(feat,1,)
                                                                    for feat in dense_features]
            dnn_feature_columns_all = [DenseFeat(feat,1,) for feat in dense_features]
            linear_feature_columns_all = fixlen_feature_columns_all
            feature_names_all = get_feature_names(linear_feature_columns_all + dnn_feature_columns_all)

            train = data
            train_model_input_all = {name: train[name] for name in feature_names_all}

            feature_index_all = build_input_features(
                        linear_feature_columns_all + dnn_feature_columns_all)

            if isinstance(train_model_input_all, dict):
                train_model_input_all = [train_model_input_all[feature] for feature in feature_index_all]

            for i in range(len(train_model_input_all)):
                if len(train_model_input_all[i].shape) == 1:
                    train_model_input_all[i] = np.expand_dims(train_model_input_all[i], axis=1)

            train_tensor_data_all = Data.TensorDataset(
                torch.from_numpy(
                    np.concatenate(train_model_input_all, axis=-1)),
                torch.from_numpy(train[target].values))

            self.train_set_all = train_tensor_data_all
            self.linear_feature_columns_all = linear_feature_columns_all
            self.dnn_feature_columns_all = dnn_feature_columns_all

            ### Generate the train loader ###
            # self.train_loader = build_trainloader_avazu_uniform(node_num)
            self.train_loader = build_trainloader_avazu()
 
        elif args.dataset == 'cifar10':
            # Data enhancement
            tra_transformer = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            val_transformer = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )

            self.train_set = torchvision.datasets.CIFAR10(
                root="./cifar/", train=True, download=False, transform=tra_transformer
            )
            if args.iid == 0:  # noniid
                random_state = np.random.RandomState(int(args.random_seed))
                num_indices = len(self.train_set)
                groups, proportion = build_non_iid_by_dirichlet(random_state=random_state, dataset=self.train_set, non_iid_alpha=args.dirichlet_alpha, num_classes=10, num_indices=num_indices, n_workers=node_num)  
                self.train_loader = groups
                self.groups = groups
                self.proportion = proportion
            else:
                data_num = [int(50000/node_num) for _ in range(node_num)]
                splited_set = torch.utils.data.random_split(self.train_set, data_num)
                self.train_loader = splited_set

            self.test_set = torchvision.datasets.CIFAR10(
                root="./cifar/", train=False, download=False, transform=val_transformer
            )
            self.test_loader = torch.utils.data.random_split(self.test_set, [int(len(self.test_set))])



def build_trainloader_alipay_uniform(node_num):
    # sample_num = 2000000
    sample_num = 500000
    sample_per_node = int(sample_num/node_num)
    train_loader = {}
    for user in range(node_num):
        train_loader[user] = []
        for sample_idx in range(sample_per_node*user, sample_per_node * (user+1)):
            train_loader[user].append(sample_idx)

    return train_loader

def build_trainloader_avazu():
    client_start_idx = torch.load('./client_start_idx.pth')
    train_loader = {}
    for user in range(len(client_start_idx)-1):
        train_loader[user] = []
        for sample_idx in range(client_start_idx[user], client_start_idx[user+1]):
            train_loader[user].append(sample_idx)

    return train_loader

def build_trainloader_avazu_uniform(node_num):
    client_start_idx = torch.load('./client_start_idx.pth')
    sample_num = client_start_idx[-1]
    sample_per_node = int(sample_num/node_num)
    train_loader = {}
    for user in range(node_num):
        train_loader[user] = []
        for sample_idx in range(sample_per_node*user, sample_per_node * (user+1)):
            train_loader[user].append(sample_idx)

    return train_loader

def build_non_iid_by_dirichlet(
    random_state = np.random.RandomState(0), dataset = 0, non_iid_alpha = 10, num_classes = 10, num_indices = 60000, n_workers = 10
):
    
    #TODO
    indicesbyclass = {}
    for i in range(num_classes):
        indicesbyclass[i] = []
    
    for idx, target in enumerate(dataset.targets):
        indicesbyclass[int(target)].append(idx)
    
    for i in range(num_classes):
        random_state.shuffle(indicesbyclass[i])
    
    client_partition = random_state.dirichlet(np.repeat(non_iid_alpha, n_workers), num_classes).transpose()

    for i in range(len(client_partition)):
        for j in range(len(client_partition[i])):
            client_partition[i][j] = int(round(client_partition[i][j]*len(indicesbyclass[j])))
    
    client_partition_index = copy.deepcopy(client_partition)
    for i in range(len(client_partition)):
        for j in range(len(client_partition[i])):
            if i == 0:
                client_partition_index[i][j] = client_partition_index[i][j]
            elif i == len(client_partition) - 1:
                client_partition_index[i][j] = len(indicesbyclass[j])
            else:
                client_partition_index[i][j] = client_partition_index[i-1][j] + client_partition_index[i][j]
            
    dict_users = {}
    for i in range(n_workers):
        dict_users[i] = []
    
    for i in range(len(client_partition)):
        for j in range(len(client_partition[i])):
            if i == 0:
                dict_users[i].extend(indicesbyclass[j][:int(client_partition_index[i][j])])
            else:
                dict_users[i].extend(indicesbyclass[j][int(client_partition_index[i-1][j]) : int(client_partition_index[i][j])])
    
    for i in range(len(dict_users)):
        random_state.shuffle(dict_users[i])

    return dict_users, client_partition
