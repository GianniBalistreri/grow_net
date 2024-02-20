"""

Neural network architectures

"""

import numpy as np
import math
import torch
import torch.nn as nn

from typing import List, Union


INITIALIZER: dict = dict(#constant=torch.nn.init.constant_,
                         #eye=torch.nn.init.eye_,
                         #dirac=torch.nn.init.dirac_,
                         #kaiming_normal=torch.nn.init.kaiming_normal_,
                         #kaiming_uniform=torch.nn.init.kaiming_uniform_,
                         #normal=torch.nn.init.normal_,
                         ones=torch.nn.init.ones_,
                         #orthogonal=torch.nn.init.orthogonal_,
                         #sparse=torch.nn.init.sparse_,
                         #trunc_normal=torch.nn.init.trunc_normal_,
                         uniform=torch.nn.init.uniform_,
                         xavier_normal=torch.nn.init.xavier_normal_,
                         xavier_uniform=torch.nn.init.xavier_uniform_,
                         zeros=torch.nn.init.zeros_
                         )
CACHE_DIR: str = 'data/cache'
BEST_MODEL_DIR: str = 'data/best_model'
OUTPUT_DIR: str = 'data/output'


def set_initializer(x, **kwargs):
    """
    Setup weights and bias initialization

    :param x:
        Data set

    :param kwargs: dict
        Key-word arguments for configuring initializer parameters
    """
    if kwargs.get('initializer') == 'constant':
        INITIALIZER.get(kwargs.get('initializer'))(x, val=kwargs.get('val'))
    elif kwargs.get('initializer') == 'dirac':
        INITIALIZER.get(kwargs.get('initializer'))(x, groups=1 if kwargs.get('groups') is None else kwargs.get('groups'))
    elif kwargs.get('initializer') == 'kaiming_normal':
        INITIALIZER.get(kwargs.get('initializer'))(x,
                                                   a=0 if kwargs.get('a') is None else kwargs.get('a'),
                                                   mode='fan_in' if kwargs.get('mode') is None else kwargs.get('mode'),
                                                   nonlinearity='leaky_relu' if kwargs.get('nonlinearity') is None else kwargs.get('nonlinearity')
                                                   )
    elif kwargs.get('initializer') == 'kaiming_uniform':
        INITIALIZER.get(kwargs.get('initializer'))(x,
                                                   a=0 if kwargs.get('a') is None else kwargs.get('a'),
                                                   mode='fan_in' if kwargs.get('mode') is None else kwargs.get('mode'),
                                                   nonlinearity='leaky_relu' if kwargs.get('nonlinearity') is None else kwargs.get('nonlinearity')
                                                   )
    elif kwargs.get('initializer') == 'normal':
        INITIALIZER.get(kwargs.get('initializer'))(x,
                                                   mean=0.0 if kwargs.get('mean') is None else kwargs.get('mean'),
                                                   std=1.0 if kwargs.get('std') is None else kwargs.get('std'),
                                                   )
    elif kwargs.get('initializer') == 'orthogonal':
        INITIALIZER.get(kwargs.get('initializer'))(x, gain=1 if kwargs.get('mean') is None else kwargs.get('mean'))
    elif kwargs.get('initializer') == 'sparse':
        INITIALIZER.get(kwargs.get('initializer'))(x,
                                                   sparsity=np.random.uniform(low=0.01, high=0.99) if kwargs.get('sparsity') is None else kwargs.get('sparsity'),
                                                   std=0.01 if kwargs.get('std') is None else kwargs.get('std'),
                                                   )
    elif kwargs.get('initializer') == 'uniform':
        INITIALIZER.get(kwargs.get('initializer'))(x,
                                                   a=0.0 if kwargs.get('a') is None else kwargs.get('a'),
                                                   b=1.0 if kwargs.get('b') is None else kwargs.get('b'),
                                                   )
    elif kwargs.get('initializer') == 'xavier_normal':
        INITIALIZER.get(kwargs.get('initializer'))(x, gain=1.0 if kwargs.get('gain') is None else kwargs.get('gain'))
    elif kwargs.get('initializer') == 'xavier_uniform':
        INITIALIZER.get(kwargs.get('initializer'))(x, gain=1.0 if kwargs.get('gain') is None else kwargs.get('gain'))
    else:
        INITIALIZER.get(kwargs.get('initializer'))(x)


class EnsembleNetworkException(Exception):
    """
    Class for handling exceptions for class EnsembleNetwork
    """
    pass


class EnsembleNetwork:
    """
    Class for building and training an ensemble of neural networks (MLP)
    """
    def __init__(self, c0, learning_rate: float = 0.002):
        """
        Class for configuration of an ensemble of neural networks type multi-layer perceptron (MLP)

        :param c0:
        :param learning_rate: nn.Parameter
            Learning rate
        """
        self.models: List[torch.nn.Module] = []
        self.c0 = c0
        self.learning_rate: float = learning_rate
        self.boost_rate: nn.Parameter = nn.Parameter(torch.tensor(self.learning_rate,
                                                                  requires_grad=True,
                                                                  device="cuda" if torch.cuda.is_available() else 'cpu'
                                                                  )
                                                     )

    def add(self, model: torch.nn.Module):
        """
        Add neural network to ensemble

        :param model: torch.nn.Module
            Pytorch neural network
        """
        self.models.append(model)

    def parameters(self) -> list:
        """
        Get parameters of all neural networks of the ensemble

        :return: list
            Neural network parameters
        """
        params: list = []
        for m in self.models:
            params.extend(m.parameters())
        params.append(self.boost_rate)
        return params

    def zero_grad(self):
        """
        Reset gradient to zero of all neural networks in ensemble
        """
        for m in self.models:
            m.zero_grad()

    def to_cuda(self):
        for m in self.models:
            m.cuda()

    def to_eval(self):
        for m in self.models:
            m.eval()

    def to_train(self):
        for m in self.models:
            m.train(True)

    def forward(self, x: torch.Tensor, update_gradient: bool):
        if len(self.models) == 0:
            return None, self.c0
        middle_feat_cum = None
        prediction = None
        if update_gradient:
            for m in self.models:
                if middle_feat_cum is None:
                    middle_feat_cum_tmp, prediction = m(x, middle_feat_cum)
                    middle_feat_cum = middle_feat_cum_tmp.clone()
                else:
                    middle_feat_cum_new, pred = m(x, middle_feat_cum)
                    middle_feat_cum = middle_feat_cum + middle_feat_cum_new
                    if prediction is None:
                        prediction = pred
                    else:
                        prediction = prediction + pred
        else:
            with torch.no_grad():
                for m in self.models:
                    if middle_feat_cum is None:
                        middle_feat_cum, prediction = m(x, middle_feat_cum)
                    else:
                        middle_feat_cum, pred = m(x, middle_feat_cum)
                        prediction += pred
        return middle_feat_cum, self.c0 + self.boost_rate * prediction

    def forward_grad(self, x):
        if len(self.models) == 0:
            return None, self.c0
        # at least one model
        middle_feat_cum = None
        prediction = None
        for m in self.models:
            if middle_feat_cum is None:
                middle_feat_cum_tmp, prediction = m(x, middle_feat_cum)
                middle_feat_cum = middle_feat_cum_tmp.clone()
            else:
                middle_feat_cum_new, pred = m(x, middle_feat_cum)
                middle_feat_cum = middle_feat_cum + middle_feat_cum_new
                if prediction is None:
                    prediction = pred
                else:
                    prediction = prediction + pred
        return middle_feat_cum, self.c0 + self.boost_rate * prediction


class MLPException(Exception):
    """
    Class for handling exceptions for class MLP
    """
    pass


class MLP(torch.nn.Module):
    """
    Class for applying Multi Layer Perceptron (Fully Connected Layer) using PyTorch as a Deep Learning Framework
    """
    def __init__(self,
                 parameters: dict,
                 input_size: int,
                 output_size: int
                 ):
        """
        :param parameters: dict
            Parameter settings

        :param input_size: int
            Number of input features

        :param output_size: int
            Output size:
                -> 1: Float value (Regression)
                -> 2: Classes (Binary Classification)
                -> >2: Classes (Multi-Class Classification)
        """
        super(MLP, self).__init__()
        self.params: dict = parameters
        self.input_size: int = input_size
        self.hidden_layers: Union[List[nn.Linear], List[SparseLinear]] = []
        self.dropout_layers: List[nn.functional] = []
        self.activation_functions: List[nn.functional] = []
        self.normalization: Union[List[nn.BatchNorm1d], List[nn.LayerNorm]] = []
        self.batch_size: int = self.params.get('batch_size')
        self.output_size: int = output_size
        if output_size == 1:
            self.activation_output: nn.functional = nn.functional.relu
        elif output_size == 2:
            self.activation_output: nn.functional = nn.functional.sigmoid
        else:
            self.activation_output: nn.functional = nn.functional.softmax
        if self.params.get('initializer') is None:
            self.params.update(dict(initializer=np.random.choice(a=list(INITIALIZER.keys()))))
        #set_initializer(x=x, **self.params)
        _l: int = 1
        if self.params.get('num_hidden_layers') > 1:
            for layer in range(1, self.params.get('num_hidden_layers') + 1, 1):
                if layer == self.params.get('num_hidden_layers'):
                    break
                self.dropout_layers.append(self.params[f'hidden_layer_{layer}_dropout_rate'])
                self.activation_functions.append(self.params[f'hidden_layer_{layer}_activation'])
                if self.params.get('normalization') == 'batch':
                    self.normalization.append(nn.BatchNorm1d(num_features=self.params.get(f'hidden_layer_{layer + 1}_neurons'),
                                                             eps=1e-5,
                                                             momentum=0.1,
                                                             affine=True,
                                                             track_running_stats=True
                                                             )
                                              )
                else:
                    self.normalization.append(nn.LayerNorm(normalized_shape=self.params.get(f'hidden_layer_{layer + 1}_neurons'),
                                                           eps=1e-5,
                                                           elementwise_affine=True
                                                           ))
                self.hidden_layers.append(torch.nn.Linear(in_features=self.params[f'hidden_layer_{layer}_neurons'],
                                                          out_features=self.params[f'hidden_layer_{layer + 1}_neurons'],
                                                          bias=True
                                                          )
                                          )
                _l += 1
        if self.params.get('use_sparse'):
            self.fully_connected_input_layer: torch.nn = SparseLinear(input_features=input_size,
                                                                      output_features=self.params['hidden_layer_1_neurons'],
                                                                      bias=True
                                                                      )
        else:
            self.fully_connected_input_layer: torch.nn = torch.nn.Linear(in_features=input_size,
                                                                         out_features=self.params['hidden_layer_1_neurons'],
                                                                         bias=True
                                                                         )
        self.fully_connected_output_layer: torch.nn = torch.nn.Linear(in_features=self.params[f'hidden_layer_{_l}_neurons'],
                                                                      #out_features=output_size,
                                                                      out_features=1,
                                                                      bias=True
                                                                      )

    def forward(self, x: torch.Tensor, penultimate_feature: torch.Tensor) -> tuple:
        """
        Feed forward algorithm

        :param x: torch.Tensor
            Input

        :param penultimate_feature: torch.Tensor
            Penultimate feature (middle feature) from previous network

        :return: Configured neural network
        """
        x = x.float()
        if penultimate_feature is not None:
            x = torch.cat([x, penultimate_feature], dim=1)
            if self.params.get('normalization') == 'batch':
                x = nn.BatchNorm1d(num_features=self.input_size,
                                   eps=1e-5,
                                   momentum=0.1,
                                   affine=True,
                                   track_running_stats=True
                                   )(x)
            else:
                x = nn.LayerNorm(normalized_shape=self.input_size,
                                 eps=1e-5,
                                 elementwise_affine=True
                                 )(x)
        x = nn.functional.leaky_relu(self.fully_connected_input_layer(x))
        for l, layer in enumerate(self.hidden_layers):
            x = self.activation_functions[l](layer(x))
            x = self.normalization[l](x)
            if self.params.get('use_alpha_dropout'):
                x = nn.functional.alpha_dropout(input=x, p=self.dropout_layers[l], training=True, inplace=False)
            else:
                x = nn.functional.dropout(input=x, p=self.dropout_layers[l], training=True, inplace=False)
        return x, self.fully_connected_output_layer(self.activation_output(x))


class SparseLinear(nn.Module):
    """
    Class for applying sparse linear layer
    """
    def __init__(self, input_features: int, output_features: int, bias: bool = True):
        super(SparseLinear, self).__init__()
        self.input_features: int = input_features
        self.output_features: int = output_features
        self.weight: nn.Parameter = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias: nn.Parameter = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)
        _standard_deviation: float = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-_standard_deviation, _standard_deviation)

    def forward(self, x):
        return sparse_linear(x, self.weight, self.bias)


class SparseLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = (input.t().mm(grad_output)).t()
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)
        return grad_input, grad_weight, grad_bias


sparse_linear = SparseLinearFunction.apply
