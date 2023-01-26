"""

Neural network architectures

"""

import numpy as np
import torch
import torch.nn as nn

from typing import List


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


class EnsembleException(Exception):
    """
    Class for handling exceptions for class Ensemble
    """
    pass


class Ensemble:
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
        self.boost_rate: nn.Parameter = nn.Parameter(torch.tensor(self.learning_rate, requires_grad=True, device="cuda"))

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

    def forward(self, x):
        if len(self.models) == 0:
            return None, self.c0
        middle_feat_cum = None
        prediction = None
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
                middle_feat_cum, prediction = m(x, middle_feat_cum)
            else:
                middle_feat_cum, pred = m(x, middle_feat_cum)
                prediction += pred
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
    def __init__(self, parameters: dict, input_size: int, output_size: int
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
        self.hidden_layers: List[torch.torch.nn.Linear] = []
        self.dropout_layers: List[nn.functional] = []
        self.use_alpha_dropout: bool = False
        self.activation_functions: List[nn.functional] = []
        self.batch_size: int = self.params.get('batch_size')
        self.output_size: int = output_size
        if output_size == 1:
            self.activation_output: nn.functional = nn.functional.relu
        elif output_size == 2:
            self.activation_output: nn.functional = nn.functional.sigmoid
        else:
            self.activation_output: nn.functional = nn.functional.softmax
        if self.params is None:
            self.fully_connected_layer: torch.nn = torch.torch.nn.Linear(in_features=input_size,
                                                                         out_features=output_size,
                                                                         bias=True
                                                                         )
        else:
            _l: int = 0
            for layer in range(0, len(self.params.keys()), 1):
                if self.params.get('hidden_layer_{}_neurons'.format(layer)) is not None:
                    if self.params['hidden_layer_{}_alpha_dropout'.format(layer)]:
                        self.use_alpha_dropout = True
                    self.dropout_layers.append(self.params['hidden_layer_{}_dropout'.format(layer)])
                    self.activation_functions.append(self.params['hidden_layer_{}_activation'.format(layer)])
                    if len(self.hidden_layers) == 0:
                        self.hidden_layers.append(torch.torch.nn.Linear(in_features=input_size,
                                                                        out_features=self.params['hidden_layer_{}_neurons'.format(layer)],
                                                                        bias=True
                                                                        )
                                                  )
                    else:
                        if layer + 1 < len(self.params.keys()):
                            _l += 1
                            self.hidden_layers.append(torch.torch.nn.Linear(in_features=self.params['hidden_layer_{}_neurons'.format(layer - 1)],
                                                                            out_features=self.params['hidden_layer_{}_neurons'.format(layer)],
                                                                            bias=True
                                                                            )
                                                      )
                        else:
                            _l = layer
                #else:
                #    break
            if len(self.hidden_layers) == 0:
                self.fully_connected_layer: torch.nn = torch.torch.nn.Linear(in_features=input_size,
                                                                             out_features=output_size,
                                                                             bias=True
                                                                             )
            else:
                self.fully_connected_layer: torch.nn = torch.torch.nn.Linear(in_features=self.params['hidden_layer_{}_neurons'.format(_l)],
                                                                             out_features=output_size,
                                                                             bias=True
                                                                             )

    def forward(self, x):
        """
        Feed forward algorithm

        :param x:
            Input

        :return: Configured neural network
        """
        if self.params.get('initializer') is None:
            self.params.update(dict(initializer=np.random.choice(a=list(INITIALIZER.keys()))))
        set_initializer(x=x, **self.params)
        x = x.float()
        for l, layer in enumerate(self.hidden_layers):
            x = self.activation_functions[l](layer(x))
            if self.use_alpha_dropout:
                x = nn.functional.alpha_dropout(input=x, p=self.dropout_layers[l], training=True, inplace=False)
            else:
                x = nn.functional.dropout(input=x, p=self.dropout_layers[l], training=True, inplace=False)
        return self.activation_output(self.fully_connected_layer(x))
