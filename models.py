import torch
import torch.nn as nn
import torch.nn.functional as F
from deepctr_utils import *


######## Models for RS, i.e. Avazu and Alipay ########
class AutoInt(BaseModel):
    """Instantiates the AutoInt Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param att_layer_num: int.The InteractingLayer number to be used.
    :param att_head_num: int.The head number in multi-head  self-attention network.
    :param att_res: bool.Whether or not use standard residual connections before output.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param dnn_activation: Activation function to use in DNN
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_use_bn:  bool. Whether use BatchNormalization before activation or not in DNN
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cuda:0"`` or ``"cuda"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return: A PyTorch model instance.

    """

    def __init__(self, args, linear_feature_columns, dnn_feature_columns, att_layer_num=3,
                 att_head_num=2, att_res=True, dnn_hidden_units=(256, 128), dnn_activation='relu',
                 l2_reg_dnn=0, l2_reg_embedding=1e-5, dnn_use_bn=False, dnn_dropout=0, init_std=0.0001, seed=1024,
                 task='binary', device='cuda:0', gpus=None):

        super(AutoInt, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=0,
                                      l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                      device=device, gpus=gpus)
        if len(dnn_hidden_units) <= 0 and att_layer_num <= 0:
            raise ValueError("Either hidden_layer or att_layer_num must > 0")
        self.use_dnn = len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0
        self.device = device
        self.args = args

        # print(len(linear_feature_columns), len(dnn_feature_columns))
        # TODO check here
        if self.args.framework == 'dcc':
            field_num = 13
            embedding_size = 4
        else:
            field_num = len(self.embedding_dict)
            if not self.embedding_size:
                embedding_size = 4
            else:
                embedding_size = self.embedding_size

        # if len(dnn_hidden_units) and att_layer_num > 0:
        #     dnn_linear_in_feature = dnn_hidden_units[-1] + field_num * embedding_size
        # elif len(dnn_hidden_units) > 0:
        #     dnn_linear_in_feature = dnn_hidden_units[-1]
        # elif att_layer_num > 0:
        #     dnn_linear_in_feature = field_num * embedding_size
        # else:
        #     raise NotImplementedError

        # define the dim of the last layer
        # TODO check here
        if self.args.output_feature == 'all':
            dnn_linear_in_feature = dnn_hidden_units[-1] + field_num * embedding_size
        elif self.args.output_feature == 'dense':
            dnn_linear_in_feature = dnn_hidden_units[-1]
        elif self.args.output_feature == 'sparse':
            dnn_linear_in_feature = field_num * embedding_size
        else:
            raise NotImplementedError

        self.dnn_linear = nn.Linear(dnn_linear_in_feature, 1, bias=False).to(device)
        self.dnn_hidden_units = dnn_hidden_units
        self.att_layer_num = att_layer_num
        if self.use_dnn:
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
        self.int_layers = nn.ModuleList(
            [InteractingLayer(embedding_size, att_head_num, att_res, device=device) for _ in range(att_layer_num)])

        self.to(device)

    def generate_input(self, X):
        X = X.float()
        # sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
        #                                                                           self.embedding_dict)
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.linear_feature_columns,
                                                                                  self.embedding_dict)
        return sparse_embedding_list, dense_value_list


class AutoIntClient(AutoInt):
    '''
    client model for dcc framework
    '''
    def __init__(self, args, linear_feature_columns, dnn_feature_columns, att_layer_num=3,
                 att_head_num=2, att_res=True, dnn_hidden_units=(256, 128), dnn_activation='relu',
                 l2_reg_dnn=0, l2_reg_embedding=1e-5, dnn_use_bn=False, dnn_dropout=0, init_std=0.0001, seed=1024,
                 task='binary', device='cuda:0', gpus=None):

        super(AutoIntClient, self).__init__(args, linear_feature_columns, dnn_feature_columns, att_layer_num,
                 att_head_num, att_res, dnn_hidden_units, dnn_activation,
                 l2_reg_dnn, l2_reg_embedding, dnn_use_bn, dnn_dropout, init_std, seed,
                 task, device, gpus)

    def forward_sparse(self, X, deep_out):
        '''
        for dcc client, it only has the sparse features
        '''
        X = X.float()
        sparse_embedding_list, dense_value_list = super().generate_input(X)
        logit = self.linear_model(X)
        att_input = concat_fun(sparse_embedding_list, axis=1)
        for layer in self.int_layers:
            att_input = layer(att_input)
        att_output = torch.flatten(att_input, start_dim=1)
        stack_out = concat_fun([att_output, deep_out])
        logit += self.dnn_linear(stack_out)
        y_pred = self.out(logit)
        return y_pred, att_output

    def forward_dense(self, X, att_output):
        '''
        for dcc client, it has the dense features
        '''
        X = X.float()
        sparse_embedding_list, dense_value_list = super().generate_input(X)
        logit = self.linear_model(X)
        dnn_input = combined_dnn_input([], dense_value_list)
        deep_out = self.dnn(dnn_input)
        stack_out = concat_fun([att_output, deep_out])
        logit += self.dnn_linear(stack_out)
        y_pred = self.out(logit)
        return y_pred, deep_out

    def forward(self, X, X_out):
        if self.args.client_features == 'sparse':
            return self.forward_sparse(self, X, X_out)
        elif self.args.client_features == 'dense':
            return self.forward_dense(self, X, X_out)

    def forward_infer_feature_sparse(self, X):
        '''
        for dcc client, it only has the sparse features
        '''
        X = X.float()
        sparse_embedding_list, dense_value_list = super().generate_input(X)
        att_input = concat_fun(sparse_embedding_list, axis=1)
        for layer in self.int_layers:
            att_input = layer(att_input)
        att_output = torch.flatten(att_input, start_dim=1)
        return att_output

    def forward_infer_feature_dense(self, X):
        '''
        for dcc client, it has the dense features
        '''
        X = X.float()
        sparse_embedding_list, dense_value_list = super().generate_input(X)
        dnn_input = combined_dnn_input([], dense_value_list)
        deep_out = self.dnn(dnn_input)
        return deep_out

    def forward_infer_feature(self, X):
        if self.args.client_features == 'sparse':
            return self.forward_infer_feature_sparse(self, X)
        elif self.args.client_features == 'dense':
            return self.forward_infer_feature_dense(self, X)



class AutoIntServer(AutoInt):
    '''
    server model for dcc framework
    '''
    def __init__(self, args, linear_feature_columns, dnn_feature_columns, att_layer_num=3,
                 att_head_num=2, att_res=True, dnn_hidden_units=(256, 128), dnn_activation='relu',
                 l2_reg_dnn=0, l2_reg_embedding=1e-5, dnn_use_bn=False, dnn_dropout=0, init_std=0.0001, seed=1024,
                 task='binary', device='cuda:0', gpus=None):

        super(AutoIntServer, self).__init__(args, linear_feature_columns, dnn_feature_columns, att_layer_num,
                 att_head_num, att_res, dnn_hidden_units, dnn_activation,
                 l2_reg_dnn, l2_reg_embedding, dnn_use_bn, dnn_dropout, init_std, seed,
                 task, device, gpus)

        # sparse feature: 13 dims
        self.encode_layer = nn.Linear(8, 8).to(self.device)
    
    def forward_dense(self, X, att_output):
        '''
        for dcc server, it has the dense features
        '''
        X = X.float()
        sparse_embedding_list, dense_value_list = super().generate_input(X)
        logit = self.linear_model(X)
        dnn_input = combined_dnn_input([], dense_value_list)
        deep_out = self.dnn(dnn_input)
        stack_out = concat_fun([att_output, deep_out])
        logit += self.dnn_linear(stack_out)
        y_pred = self.out(logit)
        return y_pred, deep_out

    def forward_sparse(self, X, deep_out):
        '''
        for dcc server, it only has the sparse features
        '''
        X = X.float()
        sparse_embedding_list, dense_value_list = super().generate_input(X)
        logit = self.linear_model(X)
        att_input = concat_fun(sparse_embedding_list, axis=1)
        for layer in self.int_layers:
            att_input = layer(att_input)
        att_output = torch.flatten(att_input, start_dim=1)
        stack_out = concat_fun([att_output, deep_out])
        logit += self.dnn_linear(stack_out)
        y_pred = self.out(logit)
        return y_pred, att_output
    
    def forward(self, X, X_out):
        if self.args.server_features == 'dense':
            return self.forward_dense(self, X, X_out)
        elif self.args.server_features == 'sparse':
            return self.forward_sparse(self, X, X_out)

    def forward_infer_feature_dense(self, X):
        '''
        for dcc server, it has the dense features
        '''
        X = X.float()
        sparse_embedding_list, dense_value_list = super().generate_input(X)
        dnn_input = combined_dnn_input([], dense_value_list)
        deep_out = self.dnn(dnn_input)
        return deep_out

    def forward_infer_feature_sparse(self, X):
        '''
        for dcc server, it only has the sparse features
        '''
        X = X.float()
        sparse_embedding_list, dense_value_list = super().generate_input(X)
        att_input = concat_fun(sparse_embedding_list, axis=1)
        for layer in self.int_layers:
            att_input = layer(att_input)
        att_output = torch.flatten(att_input, start_dim=1)
        return att_output

    def forward_infer_feature(self, X):
        if self.args.server_features == 'dense':
            return self.forward_infer_feature_dense(self, X)
        elif self.args.server_features == 'sparse':
            return self.forward_infer_feature_sparse(self, X)


class AutoIntClient_gkt(AutoInt):
    def __init__(self, args, linear_feature_columns, dnn_feature_columns, att_layer_num=3,
                 att_head_num=2, att_res=True, dnn_hidden_units=(256, 128), dnn_activation='relu',
                 l2_reg_dnn=0, l2_reg_embedding=1e-5, dnn_use_bn=False, dnn_dropout=0, init_std=0.0001, seed=1024,
                 task='binary', device='cuda:0', gpus=None):

        super(AutoIntClient_gkt, self).__init__(args, linear_feature_columns, dnn_feature_columns, att_layer_num,
                 att_head_num, att_res, dnn_hidden_units, dnn_activation,
                 l2_reg_dnn, l2_reg_embedding, dnn_use_bn, dnn_dropout, init_std, seed,
                 task, device, gpus)
    
    def forward(self, X):
        X = X.float()
        sparse_embedding_list, dense_value_list = super().generate_input(X)
        logit = self.linear_model(X)

        att_input = concat_fun(sparse_embedding_list, axis=1)

        for layer in self.int_layers:
            att_input = layer(att_input)

        att_output = torch.flatten(att_input, start_dim=1)

        if self.args.client_features == 'all':  # Deep & Interacting Layer
            # for dnn_input, we only use the dense features
            dnn_input = combined_dnn_input([], dense_value_list)
            deep_out = self.dnn(dnn_input)
            stack_out = concat_fun([att_output, deep_out])
            logit += self.dnn_linear(stack_out)
            y_pred = self.out(logit)
            output = stack_out
        elif self.args.client_features == 'sparse':  # Only Interacting Layer
            logit += self.dnn_linear(att_output)
            y_pred = self.out(logit)
            output = att_output
        elif self.args.client_feature == 'dense':  # Only Interacting Layer
            output = deep_out
            logit += self.dnn_linear(deep_out)
            y_pred = self.out(logit)
        else:  # Error
            pass
        return y_pred, output

class AutoIntServer_gkt(AutoInt):
    def __init__(self, args, linear_feature_columns, dnn_feature_columns, att_layer_num=3,
                 att_head_num=2, att_res=True, dnn_hidden_units=(256, 128), dnn_activation='relu',
                 l2_reg_dnn=0, l2_reg_embedding=1e-5, dnn_use_bn=False, dnn_dropout=0, init_std=0.0001, seed=1024,
                 task='binary', device='cuda:0', gpus=None):

        super(AutoIntServer_gkt, self).__init__(args, linear_feature_columns, dnn_feature_columns, att_layer_num,
                 att_head_num, att_res, dnn_hidden_units, dnn_activation,
                 l2_reg_dnn, l2_reg_embedding, dnn_use_bn, dnn_dropout, init_std, seed,
                 task, device, gpus)

    def forward(self, output):
        '''
        for gkt server, it only has the dnn_linear layer
        '''
        logit = self.dnn_linear(output)
        y_pred = self.out(logit)

        return y_pred


######## Models for CIFAR10 ########

# resnet for clients

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetClient(nn.Module):

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False, groups=1,
                 width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, KD=False):
        super(ResNetClient, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.groups = groups
        self.base_width = width_per_group

        # initialization is defined here:https://github.com/pytorch/pytorch/tree/master/torch/nn/modules
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)  # init: kaiming_uniform
        # self.conv1 = nn.Conv2d(3, 32, 3)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, layers[0])
        # self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.fc = nn.Linear(16 * block.expansion, num_classes)
        # self.fc = nn.Linear(32 * block.expansion, num_classes)

        self.KD = KD
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, feature):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # B x 16 x 32 x 32
        # x = self.maxpool(x)
        extracted_features = x

        # concat the server's feature and the client's
        x = torch.cat((extracted_features, feature), 3) # TODO change the dim here

        x = self.layer1(x)  # B x 16 x 32 x 32
        # x = self.layer2(x)  # B x 32 x 16 x 16
        # x = self.layer3(x)  # B x 64 x 8 x 8

        x = self.avgpool(x)  # B x 64 x 1 x 1
        x_f = x.view(x.size(0), -1)  # B x 64
        logits = self.fc(x_f)  # B x num_classes
        return logits, extracted_features

    def forward_infer_feature(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # B x 16 x 32 x 32
        # x = self.maxpool(x)
        extracted_features = x

        return extracted_features

def resnet5_56(c, pretrained=False, path=None, **kwargs):
    """
    Constructs a ResNet-32 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """

    model = ResNetClient(BasicBlock, [1, 2, 2], num_classes=c, **kwargs)
    if pretrained:
        checkpoint = torch.load(path)
        state_dict = checkpoint['state_dict']

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # name = k[7:]  # remove 'module.' of dataparallel
            name = k.replace("module.", "")
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
    return model


def resnet8_56(c, pretrained=False, path=None, **kwargs):
    """
    Constructs a ResNet-32 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """

    model = ResNetClient(Bottleneck, [2, 2, 2], num_classes=c, **kwargs)
    if pretrained:
        checkpoint = torch.load(path)
        state_dict = checkpoint['state_dict']

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # name = k[7:]  # remove 'module.' of dataparallel
            name = k.replace("module.", "")
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
    return model

class ResNetClient_gkt(nn.Module):

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False, groups=1,
                 width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, KD=False):
        super(ResNetClient_gkt, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.groups = groups
        self.base_width = width_per_group

        # initialization is defined here:https://github.com/pytorch/pytorch/tree/master/torch/nn/modules
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)  # init: kaiming_uniform
        # self.conv1 = nn.Conv2d(3, 32, 3)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, layers[0])
        # self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.fc = nn.Linear(16 * block.expansion, num_classes)
        # self.fc = nn.Linear(32 * block.expansion, num_classes)

        self.KD = KD
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # B x 16 x 32 x 32
        # x = self.maxpool(x)
        extracted_features = x

        x = self.layer1(x)  # B x 16 x 32 x 32
        # x = self.layer2(x)  # B x 32 x 16 x 16
        # x = self.layer3(x)  # B x 64 x 8 x 8

        x = self.avgpool(x)  # B x 64 x 1 x 1
        x_f = x.view(x.size(0), -1)  # B x 64
        logits = self.fc(x_f)  # B x num_classes
        return logits, extracted_features


def resnet8_56_gkt(c, pretrained=False, path=None, **kwargs):
    model = ResNetClient_gkt(Bottleneck, [2, 2, 2], num_classes=c, **kwargs)

    return model

def resnet5_56_gkt(c, pretrained=False, path=None, **kwargs):
    model = ResNetClient_gkt(Bottleneck, [1, 2, 2], num_classes=c, **kwargs)

    return model

class ResNetServer(nn.Module):

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False, groups=1,
                 width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, KD=False):
        super(ResNetServer, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d()
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.KD = KD
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, feature):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # B x 16 x 32 x 32
        # x = self.maxpool(x)
        extracted_features = x

        # concat the clients' features to the server's
        x = torch.cat((feature, extracted_features), 3)

        x = self.layer1(x)  # B x 16 x 32 x 32
        x = self.layer2(x)  # B x 32 x 16 x 16
        x = self.layer3(x)  # B x 64 x 8 x 8

        x = self.avgpool(x)  # B x 64 x 1 x 1
        x_f = x.view(x.size(0), -1)  # B x 64
        logits = self.fc(x_f)  # B x num_classes
        return logits, extracted_features
        
    def forward_infer_feature(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # B x 16 x 32 x 32
        # x = self.maxpool(x)
        extracted_features = x

        return extracted_features

def resnet56_server(c, pretrained=False, path=None, **kwargs):
    """
    Constructs a ResNet-110 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    model = ResNetServer(Bottleneck, [6, 6, 6], num_classes=c, **kwargs)
    if pretrained:
        checkpoint = torch.load(path)
        state_dict = checkpoint['state_dict']

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # name = k[7:]  # remove 'module.' of dataparallel
            name = k.replace("module.", "")
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
    return model


class ResNetServer_gkt(nn.Module):

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False, groups=1,
                 width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, KD=False):
        super(ResNetServer_gkt, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d()
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.KD = KD
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, feature):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)  # B x 16 x 32 x 32
        # # x = self.maxpool(x)
        # extracted_features = x

        x = feature

        x = self.layer1(x)  # B x 16 x 32 x 32
        x = self.layer2(x)  # B x 32 x 16 x 16
        x = self.layer3(x)  # B x 64 x 8 x 8

        x = self.avgpool(x)  # B x 64 x 1 x 1
        x_f = x.view(x.size(0), -1)  # B x 64
        logits = self.fc(x_f)  # B x num_classes
        return logits

def resnet56_server_gkt(c, pretrained=False, path=None, **kwargs):

    model = ResNetServer_gkt(Bottleneck, [6, 6, 6], num_classes=c, **kwargs)

    return model
