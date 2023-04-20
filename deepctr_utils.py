import copy
from collections import OrderedDict, namedtuple, defaultdict
from itertools import chain
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np


DEFAULT_GROUP_NAME = "default_group"

class SparseFeat(namedtuple('SparseFeat',
                            ['name', 'vocabulary_size', 'embedding_dim', 'use_hash', 'dtype', 'embedding_name',
                             'group_name'])):
    __slots__ = ()

    def __new__(cls, name, vocabulary_size, embedding_dim=4, use_hash=False, dtype="int32", embedding_name=None,
                group_name=DEFAULT_GROUP_NAME):
        if embedding_name is None:
            embedding_name = name
        if embedding_dim == "auto":
            embedding_dim = 6 * int(pow(vocabulary_size, 0.25))
        if use_hash:
            print(
                "Notice! Feature Hashing on the fly currently is not supported in torch version,you can use tensorflow version!")
        return super(SparseFeat, cls).__new__(cls, name, vocabulary_size, embedding_dim, use_hash, dtype,
                                              embedding_name, group_name)

    def __hash__(self):
        return self.name.__hash__()

class VarLenSparseFeat(namedtuple('VarLenSparseFeat',
                                  ['sparsefeat', 'maxlen', 'combiner', 'length_name'])):
    __slots__ = ()

    def __new__(cls, sparsefeat, maxlen, combiner="mean", length_name=None):
        return super(VarLenSparseFeat, cls).__new__(cls, sparsefeat, maxlen, combiner, length_name)

    @property
    def name(self):
        return self.sparsefeat.name

    @property
    def vocabulary_size(self):
        return self.sparsefeat.vocabulary_size

    @property
    def embedding_dim(self):
        return self.sparsefeat.embedding_dim

    @property
    def use_hash(self):
        return self.sparsefeat.use_hash

    @property
    def dtype(self):
        return self.sparsefeat.dtype

    @property
    def embedding_name(self):
        return self.sparsefeat.embedding_name

    @property
    def group_name(self):
        return self.sparsefeat.group_name

    def __hash__(self):
        return self.name.__hash__()

class DenseFeat(namedtuple('DenseFeat', ['name', 'dimension', 'dtype'])):
    __slots__ = ()

    def __new__(cls, name, dimension=1, dtype="float32"):
        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype)

    def __hash__(self):
        return self.name.__hash__()

def get_feature_names(feature_columns):
    features = build_input_features(feature_columns)
    return list(features.keys())


def build_input_features(feature_columns):
    # Return OrderedDict: {feature_name:(start, start+dimension)}

    features = OrderedDict()

    start = 0
    for feat in feature_columns:
        feat_name = feat.name
        if feat_name in features:
            continue
        if isinstance(feat, SparseFeat):
            features[feat_name] = (start, start + 1)
            start += 1
        elif isinstance(feat, DenseFeat):
            features[feat_name] = (start, start + feat.dimension)
            start += feat.dimension
        elif isinstance(feat, VarLenSparseFeat):
            features[feat_name] = (start, start + feat.maxlen)
            start += feat.maxlen
            if feat.length_name is not None and feat.length_name not in features:
                features[feat.length_name] = (start, start + 1)
                start += 1
        else:
            raise TypeError("Invalid feature column type,got", type(feat))
    return features



def get_varlen_pooling_list(embedding_dict, features, feature_index, varlen_sparse_feature_columns, device):
    varlen_sparse_embedding_list = []
    for feat in varlen_sparse_feature_columns:
        seq_emb = embedding_dict[feat.name]
        if feat.length_name is None:
            seq_mask = features[:, feature_index[feat.name][0]:feature_index[feat.name][1]].long() != 0

            emb = SequencePoolingLayer(mode=feat.combiner, supports_masking=True, device=device)(
                [seq_emb, seq_mask])
        else:
            seq_length = features[:, feature_index[feat.length_name][0]:feature_index[feat.length_name][1]].long()
            emb = SequencePoolingLayer(mode=feat.combiner, supports_masking=False, device=device)(
                [seq_emb, seq_length])
        varlen_sparse_embedding_list.append(emb)
    return varlen_sparse_embedding_list


def create_embedding_matrix(feature_columns, init_std=0.0001, linear=False, sparse=False, device='cpu'):
    # Return nn.ModuleDict: for sparse features, {embedding_name: nn.Embedding}
    # for varlen sparse features, {embedding_name: nn.EmbeddingBag}
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []

    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if len(feature_columns) else []

    embedding_dict = nn.ModuleDict(
        {feat.embedding_name: nn.Embedding(feat.vocabulary_size, feat.embedding_dim if not linear else 1, sparse=sparse)
         for feat in
         sparse_feature_columns + varlen_sparse_feature_columns}
    )

    # for feat in varlen_sparse_feature_columns:
    #     embedding_dict[feat.embedding_name] = nn.EmbeddingBag(
    #         feat.dimension, embedding_size, sparse=sparse, mode=feat.combiner)

    for tensor in embedding_dict.values():
        nn.init.normal_(tensor.weight, mean=0, std=float(init_std))

    return embedding_dict.to(device)


def embedding_lookup(X, sparse_embedding_dict, sparse_input_dict, sparse_feature_columns, return_feat_list=(),
                     mask_feat_list=(), to_list=False):
    """
        Args:
            X: input Tensor [batch_size x hidden_dim]
            sparse_embedding_dict: nn.ModuleDict, {embedding_name: nn.Embedding}
            sparse_input_dict: OrderedDict, {feature_name:(start, start+dimension)}
            sparse_feature_columns: list, sparse features
            return_feat_list: list, names of feature to be returned, defualt () -> return all features
            mask_feat_list, list, names of feature to be masked in hash transform
        Return:
            group_embedding_dict: defaultdict(list)
    """
    group_embedding_dict = defaultdict(list)
    for fc in sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if (len(return_feat_list) == 0 or feature_name in return_feat_list):
            # TODO: add hash function
            # if fc.use_hash:
            #     raise NotImplementedError("hash function is not implemented in this version!")
            lookup_idx = np.array(sparse_input_dict[feature_name])
            input_tensor = X[:, lookup_idx[0]:lookup_idx[1]].long()
            emb = sparse_embedding_dict[embedding_name](input_tensor)
            group_embedding_dict[fc.group_name].append(emb)
    if to_list:
        return list(chain.from_iterable(group_embedding_dict.values()))
    return group_embedding_dict


def varlen_embedding_lookup(X, embedding_dict, sequence_input_dict, varlen_sparse_feature_columns):
    varlen_embedding_vec_dict = {}
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if fc.use_hash:
            # lookup_idx = Hash(fc.vocabulary_size, mask_zero=True)(sequence_input_dict[feature_name])
            # TODO: add hash function
            lookup_idx = sequence_input_dict[feature_name]
        else:
            lookup_idx = sequence_input_dict[feature_name]
        varlen_embedding_vec_dict[feature_name] = embedding_dict[embedding_name](
            X[:, lookup_idx[0]:lookup_idx[1]].long())  # (lookup_idx)

    return varlen_embedding_vec_dict


def get_dense_input(X, features, feature_columns):
    dense_feature_columns = list(filter(lambda x: isinstance(
        x, DenseFeat), feature_columns)) if feature_columns else []
    dense_input_list = []
    for fc in dense_feature_columns:
        lookup_idx = np.array(features[fc.name])
        input_tensor = X[:, lookup_idx[0]:lookup_idx[1]].float()
        dense_input_list.append(input_tensor)
    return dense_input_list


def maxlen_lookup(X, sparse_input_dict, maxlen_column):
    if maxlen_column is None or len(maxlen_column)==0:
        raise ValueError('please add max length column for VarLenSparseFeat of DIN/DIEN input')
    lookup_idx = np.array(sparse_input_dict[maxlen_column[0]])
    return X[:, lookup_idx[0]:lookup_idx[1]].long()


class Linear(nn.Module):
    def __init__(self, feature_columns, feature_index, device='cpu', init_std=0.0001):
        super(Linear, self).__init__()
        self.feature_index = feature_index
        self.device = device
        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        self.dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        self.varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if len(feature_columns) else []

        self.embedding_dict = create_embedding_matrix(feature_columns, init_std, linear=True, sparse=False,
                                                      device=device)

        #         nn.ModuleDict(
        #             {feat.embedding_name: nn.Embedding(feat.dimension, 1, sparse=True) for feat in
        #              self.sparse_feature_columns}
        #         )
        # .to("cpu")
        for tensor in self.embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)

        if len(self.dense_feature_columns) > 0:
            # self.weight = nn.Parameter(torch.Tensor(sum(fc.dimension for fc in self.dense_feature_columns), 1).to(
            #     device))
            self.weight = nn.Parameter(torch.Tensor(sum(fc.dimension for fc in self.dense_feature_columns), 1).to(
                device))
            torch.nn.init.normal_(self.weight, mean=0, std=init_std)

    def forward(self, X, sparse_feat_refine_weight=None):

        sparse_embedding_list = [self.embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
            feat in self.sparse_feature_columns]

        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            self.dense_feature_columns]

        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index,
                                                      self.varlen_sparse_feature_columns)
        varlen_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                        self.varlen_sparse_feature_columns, self.device)

        sparse_embedding_list += varlen_embedding_list

        linear_logit = torch.zeros([X.shape[0], 1]).to(self.device)
        # linear_logit = torch.zeros([X.shape[0], 1]).to(self.device)
        if len(sparse_embedding_list) > 0:
            sparse_embedding_cat = torch.cat(sparse_embedding_list, dim=-1)
            if sparse_feat_refine_weight is not None:
                # w_{x,i}=m_{x,i} * w_i (in IFM and DIFM)
                sparse_embedding_cat = sparse_embedding_cat * sparse_feat_refine_weight.unsqueeze(1)
            sparse_feat_logit = torch.sum(sparse_embedding_cat, dim=-1, keepdim=False).to(self.device)
            linear_logit += sparse_feat_logit
        if len(dense_value_list) > 0:
            dense_value_logit = torch.cat(
                dense_value_list, dim=-1).matmul(self.weight)
            linear_logit += dense_value_logit

        return linear_logit


class PredictionLayer(nn.Module):
    """
      Arguments
         - **task**: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
         - **use_bias**: bool.Whether add bias term or not.
    """

    def __init__(self, task='binary', use_bias=True, **kwargs):
        if task not in ["binary", "multiclass", "regression"]:
            raise ValueError("task must be binary,multiclass or regression")

        super(PredictionLayer, self).__init__()
        self.use_bias = use_bias
        self.task = task
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros((1,)))

    def forward(self, X):
        output = X
        if self.use_bias:
            output += self.bias
        if self.task == "binary":
            output = torch.sigmoid(output)
        return output

def concat_fun(inputs, axis=-1):
    if len(inputs) == 1:
        return inputs[0]
    else:
        return torch.cat(inputs, dim=axis)


def combined_dnn_input(sparse_embedding_list, dense_value_list):
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_dnn_input = torch.flatten(
            torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
        dense_dnn_input = torch.flatten(
            torch.cat(dense_value_list, dim=-1), start_dim=1)
        return concat_fun([sparse_dnn_input, dense_dnn_input])
    elif len(sparse_embedding_list) > 0:
        return torch.flatten(torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
    elif len(dense_value_list) > 0:
        return torch.flatten(torch.cat(dense_value_list, dim=-1), start_dim=1)
    else:
        raise NotImplementedError


class InteractingLayer(nn.Module):
    """A Layer used in AutoInt that model the correlations between different feature fields by multi-head self-attention mechanism.
      Input shape
            - A 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
            - 3D tensor with shape:``(batch_size,field_size,embedding_size)``.
      Arguments
            - **in_features** : Positive integer, dimensionality of input features.
            - **head_num**: int.The head number in multi-head self-attention network.
            - **use_res**: bool.Whether or not use standard residual connections before output.
            - **seed**: A Python integer to use as random seed.
      References
            - [Song W, Shi C, Xiao Z, et al. AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks[J]. arXiv preprint arXiv:1810.11921, 2018.](https://arxiv.org/abs/1810.11921)
    """

    def __init__(self, embedding_size, head_num=2, use_res=True, scaling=False, seed=1024, device='cpu'):
        super(InteractingLayer, self).__init__()
        if head_num <= 0:
            raise ValueError('head_num must be a int > 0')
        if embedding_size % head_num != 0:
            raise ValueError('embedding_size is not an integer multiple of head_num!')
        self.att_embedding_size = embedding_size // head_num
        self.head_num = head_num
        self.use_res = use_res
        self.scaling = scaling
        self.seed = seed

        self.W_Query = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        self.W_key = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        self.W_Value = nn.Parameter(torch.Tensor(embedding_size, embedding_size))

        if self.use_res:
            self.W_Res = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        for tensor in self.parameters():
            nn.init.normal_(tensor, mean=0.0, std=0.05)

        self.to(device)

    def forward(self, inputs):

        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs.shape)))

        # None F D
        querys = torch.tensordot(inputs, self.W_Query, dims=([-1], [0]))
        keys = torch.tensordot(inputs, self.W_key, dims=([-1], [0]))
        values = torch.tensordot(inputs, self.W_Value, dims=([-1], [0]))

        # head_num None F D/head_num
        querys = torch.stack(torch.split(querys, self.att_embedding_size, dim=2))
        keys = torch.stack(torch.split(keys, self.att_embedding_size, dim=2))
        values = torch.stack(torch.split(values, self.att_embedding_size, dim=2))

        inner_product = torch.einsum('bnik,bnjk->bnij', querys, keys)  # head_num None F F
        if self.scaling:
            inner_product /= self.att_embedding_size ** 0.5
        self.normalized_att_scores = F.softmax(inner_product, dim=-1)  # head_num None F F
        result = torch.matmul(self.normalized_att_scores, values)  # head_num None F D/head_num

        result = torch.cat(torch.split(result, 1, ), dim=-1)
        result = torch.squeeze(result, dim=0)  # None F D
        if self.use_res:
            result += torch.tensordot(inputs, self.W_Res, dims=([-1], [0]))
        result = F.relu(result)

        return result

class BaseModel(nn.Module):
    def __init__(self, linear_feature_columns, dnn_feature_columns, l2_reg_linear=1e-5, l2_reg_embedding=1e-5,
                 init_std=0.0001, seed=1024, task='binary', device='cuda:0', gpus=None):

        super(BaseModel, self).__init__()
        torch.manual_seed(seed)
        self.dnn_feature_columns = dnn_feature_columns
        self.linear_feature_columns = linear_feature_columns

        self.reg_loss = torch.zeros((1,), device=device)
        self.aux_loss = torch.zeros((1,), device=device)
        self.device = device
        self.gpus = gpus
        if gpus and str(self.gpus[0]) not in self.device:
            raise ValueError(
                "`gpus[0]` should be the same gpu with `device`")

        self.feature_index = build_input_features(
            linear_feature_columns + dnn_feature_columns)
        self.dnn_feature_columns = dnn_feature_columns

        # TODO here linear<->dnn exchange
        # self.embedding_dict = create_embedding_matrix(dnn_feature_columns, init_std, sparse=False, device=device)
        self.embedding_dict = create_embedding_matrix(linear_feature_columns, init_std, sparse=False, device=device)
        #         nn.ModuleDict(
        #             {feat.embedding_name: nn.Embedding(feat.dimension, embedding_size, sparse=True) for feat in
        #              self.dnn_feature_columns}
        #         )

        self.linear_model = Linear(
            linear_feature_columns, self.feature_index, device=device)

        self.regularization_weight = []

        self.add_regularization_weight(self.embedding_dict.parameters(), l2=l2_reg_embedding)
        self.add_regularization_weight(self.linear_model.parameters(), l2=l2_reg_linear)

        self.out = PredictionLayer(task, )
        self.to(device)

        # parameters for callbacks
        self._is_graph_network = True  # used for ModelCheckpoint in tf2
        self._ckpt_saved_epoch = False  # used for EarlyStopping in tf1.14
        # self.history = History()

    def input_from_feature_columns(self, X, feature_columns, embedding_dict, support_dense=True):

        # X_protect = copy.deepcopy(X)

        sparse_feature_columns = list(filter(lambda y: isinstance(y, SparseFeat), feature_columns)) if len(feature_columns) else []
        dense_feature_columns = list(filter(lambda y: isinstance(y, DenseFeat), feature_columns)) if len(feature_columns) else []

        varlen_sparse_feature_columns = list(
            filter(lambda y: isinstance(y, VarLenSparseFeat), feature_columns)) if feature_columns else []

        if not support_dense and len(dense_feature_columns) > 0:
            raise ValueError(
                "DenseFeat is not supported in dnn_feature_columns")

        # X = copy.deepcopy(X_protect)

        # sparse_embedding_list = [embedding_dict[feat.embedding_name](X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for feat in sparse_feature_columns]
        sparse_embedding_list = [embedding_dict[feat.embedding_name](X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].int()) for feat in sparse_feature_columns]

        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index,
                                                      varlen_sparse_feature_columns)
        varlen_sparse_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                               varlen_sparse_feature_columns, self.device)

        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            dense_feature_columns]

        return sparse_embedding_list + varlen_sparse_embedding_list, dense_value_list

    def compute_input_dim(self, feature_columns, include_sparse=True, include_dense=True, feature_group=False):
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)), feature_columns)) if len(
            feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        dense_input_dim = sum(
            map(lambda x: x.dimension, dense_feature_columns))
        if feature_group:
            sparse_input_dim = len(sparse_feature_columns)
        else:
            sparse_input_dim = sum(feat.embedding_dim for feat in sparse_feature_columns)
        input_dim = 0
        if include_sparse:
            input_dim += sparse_input_dim
        if include_dense:
            input_dim += dense_input_dim
        return input_dim

    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        # For a Parameter, put it in a list to keep Compatible with get_regularization_loss()
        if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        # For generators, filters and ParameterLists, convert them to a list of tensors to avoid bugs.
        # e.g., we can't pickle generator objects when we save the model.
        else:
            weight_list = list(weight_list)
        self.regularization_weight.append((weight_list, l1, l2))

    def get_regularization_loss(self, ):
        total_reg_loss = torch.zeros((1,), device=self.device)
        for weight_list, l1, l2 in self.regularization_weight:
            for w in weight_list:
                if isinstance(w, tuple):
                    parameter = w[1]  # named_parameters
                else:
                    parameter = w
                if l1 > 0:
                    total_reg_loss += torch.sum(l1 * torch.abs(parameter))
                if l2 > 0:
                    try:
                        total_reg_loss += torch.sum(l2 * torch.square(parameter))
                    except AttributeError:
                        total_reg_loss += torch.sum(l2 * parameter * parameter)

        return total_reg_loss

    def add_auxiliary_loss(self, aux_loss, alpha):
        self.aux_loss = aux_loss * alpha

    def compile(self, optimizer,
                loss=None,
                metrics=None,
                ):
        """
        :param optimizer: String (name of optimizer) or optimizer instance. See [optimizers](https://pytorch.org/docs/stable/optim.html).
        :param loss: String (name of objective function) or objective function. See [losses](https://pytorch.org/docs/stable/nn.functional.html#loss-functions).
        :param metrics: List of metrics to be evaluated by the model during training and testing. Typically you will use `metrics=['accuracy']`.
        """
        self.metrics_names = ["loss"]
        self.optim = self._get_optim(optimizer)
        self.loss_func = self._get_loss_func(loss)
        self.metrics = self._get_metrics(metrics)

    def _get_optim(self, optimizer):
        if isinstance(optimizer, str):
            if optimizer == "sgd":
                optim = torch.optim.SGD(self.parameters(), lr=0.01)
            elif optimizer == "adam":
                optim = torch.optim.Adam(self.parameters())  # 0.001
            elif optimizer == "adagrad":
                optim = torch.optim.Adagrad(self.parameters())  # 0.01
            elif optimizer == "rmsprop":
                optim = torch.optim.RMSprop(self.parameters())
            else:
                raise NotImplementedError
        else:
            optim = optimizer
        return optim

    def _get_loss_func(self, loss):
        if isinstance(loss, str):
            if loss == "binary_crossentropy":
                loss_func = F.binary_cross_entropy
            elif loss == "mse":
                loss_func = F.mse_loss
            elif loss == "mae":
                loss_func = F.l1_loss
            else:
                raise NotImplementedError
        else:
            loss_func = loss
        return loss_func

    def _log_loss(self, y_true, y_pred, eps=1e-7, normalize=True, sample_weight=None, labels=None):
        # change eps to improve calculation accuracy
        return log_loss(y_true,
                        y_pred,
                        eps,
                        normalize,
                        sample_weight,
                        labels)

    @staticmethod
    def _accuracy_score(y_true, y_pred):
        return accuracy_score(y_true, np.where(y_pred > 0.5, 1, 0))

    def _get_metrics(self, metrics, set_eps=False):
        metrics_ = {}
        if metrics:
            for metric in metrics:
                if metric == "binary_crossentropy" or metric == "logloss":
                    if set_eps:
                        metrics_[metric] = self._log_loss
                    else:
                        metrics_[metric] = log_loss
                if metric == "auc":
                    metrics_[metric] = roc_auc_score
                if metric == "mse":
                    metrics_[metric] = mean_squared_error
                if metric == "accuracy" or metric == "acc":
                    metrics_[metric] = self._accuracy_score
                self.metrics_names.append(metric)
        return metrics_

    def _in_multi_worker_mode(self):
        # used for EarlyStopping in tf1.15
        return None

    @property
    def embedding_size(self, ):
        # feature_columns = self.dnn_feature_columns
        feature_columns = self.linear_feature_columns
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)), feature_columns)) if len(
            feature_columns) else []
        embedding_size_set = set([feat.embedding_dim for feat in sparse_feature_columns])
        if len(embedding_size_set) > 1:
            raise ValueError("embedding_dim of SparseFeat and VarlenSparseFeat must be same in this model!")
        if len(list(embedding_size_set)) > 0:
            return list(embedding_size_set)[0]
        else:
            return None

class DNN(nn.Module):
    """The Multi Layer Percetron

      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.

      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.

      Arguments
        - **inputs_dim**: input feature dimension.

        - **hidden_units**:list of positive integer, the layer number and units in each layer.

        - **activation**: Activation function to use.

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.

        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.

        - **use_bn**: bool. Whether use BatchNormalization before activation or not.

        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, inputs_dim, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False,
                 init_std=0.0001, dice_dim=3, seed=1024, device='cpu'):
        super(DNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")
        hidden_units = [inputs_dim] + list(hidden_units)

        self.linears = nn.ModuleList(
            [nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        if self.use_bn:
            self.bn = nn.ModuleList(
                [nn.BatchNorm1d(hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        self.activation_layers = nn.ModuleList(
            [activation_layer(activation, hidden_units[i + 1], dice_dim) for i in range(len(hidden_units) - 1)])

        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

        self.to(device)

    def forward(self, inputs):
        deep_input = inputs

        for i in range(len(self.linears)):

            fc = self.linears[i](deep_input)

            if self.use_bn:
                fc = self.bn[i](fc)

            fc = self.activation_layers[i](fc)

            fc = self.dropout(fc)
            deep_input = fc
        return deep_input

def activation_layer(act_name, hidden_size=None, dice_dim=2):
    """Construct activation layers

    Args:
        act_name: str or nn.Module, name of activation function
        hidden_size: int, used for Dice activation
        dice_dim: int, used for Dice activation
    Return:
        act_layer: activation layer
    """
    if isinstance(act_name, str):
        if act_name.lower() == 'sigmoid':
            act_layer = nn.Sigmoid()
        # elif act_name.lower() == 'linear':
        #     act_layer = Identity()
        elif act_name.lower() == 'relu':
            act_layer = nn.ReLU(inplace=True)
        # elif act_name.lower() == 'dice':
        #     assert dice_dim
        #     act_layer = Dice(hidden_size, dice_dim)
        elif act_name.lower() == 'prelu':
            act_layer = nn.PReLU()
    elif issubclass(act_name, nn.Module):
        act_layer = act_name()
    else:
        raise NotImplementedError

    return act_layer