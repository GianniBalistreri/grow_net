"""

Generate grow networks (gradient boosting multi-layer perceptron)

"""

import boto3
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional

from .data_loader import LoadTabularData, CLOUD_PROVIDER
from .evaluate import EvalClf, EvalLTR, EvalReg, ML_METRIC
from .neural_network import EnsembleNetwork, MLP
from .utils import remove_single_queries
from datetime import datetime
from google.cloud import storage
from itertools import accumulate
from operator import mul
from typing import Dict, List, Tuple

torch.autograd.set_detect_anomaly(True)

ACTIVATION: dict = dict(weighted_sum=dict(celu=torch.nn.CELU,
                                          elu=torch.nn.ELU,
                                          gelu=torch.nn.GELU,
                                          hard_shrink=torch.nn.Hardshrink,
                                          hard_sigmoid=torch.nn.Hardsigmoid,
                                          hard_swish=torch.nn.Hardswish,
                                          hard_tanh=torch.nn.Hardtanh,
                                          leaky_relu=torch.nn.LeakyReLU,
                                          #linear=torch.nn.Linear,
                                          log_sigmoid=torch.nn.LogSigmoid,
                                          prelu=torch.nn.PReLU,
                                          rrelu=torch.nn.RReLU,
                                          relu=torch.nn.ReLU,
                                          selu=torch.nn.SELU,
                                          sigmoid=torch.nn.Sigmoid,
                                          silu=torch.nn.SiLU,
                                          soft_plus=torch.nn.Softplus,
                                          soft_shrink=torch.nn.Softshrink,
                                          soft_sign=torch.nn.Softsign,
                                          tanh=torch.nn.Tanh,
                                          tanh_shrink=torch.nn.Tanhshrink
                                          ),
                        others=dict(log_softmax=torch.nn.LogSoftmax,
                                    softmin=torch.nn.Softmin,
                                    softmax=torch.nn.Softmax,
                                    )
                        )
LOSS: dict = dict(reg=dict(mse=torch.nn.MSELoss,
                           l1=torch.nn.L1Loss,
                           l1_smooth=torch.nn.SmoothL1Loss,
                           #cosine_embedding=torch.nn.CosineEmbeddingLoss
                           ),
                  clf_binary=dict(cross_entropy=torch.nn.CrossEntropyLoss,
                                  binary_cross_entropy=torch.nn.BCELoss,
                                  binary_cross_entropy_logits=torch.nn.BCEWithLogitsLoss
                                  #hinge_embedding=torch.nn.functional.hinge_embedding_loss
                                  ),
                  clf_multi=dict(cross_entropy=torch.nn.CrossEntropyLoss,
                                 #multilabel_margin=torch.nn.MultiLabelMarginLoss,
                                 #multilabel_soft_margin=torch.nn.MultiLabelSoftMarginLoss
                                 )
                  )
OPTIMIZER: dict = dict(adam=torch.optim.Adam,
                       rmsprop=torch.optim.RMSprop,
                       sgd=torch.optim.SGD
                       )
TORCH_OBJECT_PARAM: Dict[str, List[str]] = dict(activation=['activation'],
                                                initializer=['initializer'],
                                                optimizer=['optimizer',
                                                           'learning_rate',
                                                           'momentum',
                                                           'dampening',
                                                           'weight_decay',
                                                           'nesterov',
                                                           'alpha',
                                                           'eps',
                                                           'centered',
                                                           'betas',
                                                           'amsgrad'
                                                           ]
                                                )


def geometric_progression(n: int = 10, ratio: int = 2) -> List[int]:
    """
    Generate list of geometric progression values

    n: int
        Amount of values of the geometric progression

    ratio: float
        Base ratio value of the geometric progression

    :return List[int]
        Geometric progression values
    """
    return list(accumulate([ratio] * n, mul))


class GrowNetException(Exception):
    """
    Class for handling exceptions for class NeuralNetwork
    """
    pass


class GrowNet:
    """
    Class for handling neural networks
    """
    def __init__(self,
                 target: str,
                 predictors: List[str],
                 target_type: str,
                 input_layer_size: int = None,
                 train_data_path: str = None,
                 test_data_path: str = None,
                 validation_data_path: str = None,
                 input_param: dict = None,
                 model_param: dict = None,
                 seed: int = 1234,
                 **kwargs
                 ):
        """
        :param target: str
            Name of the target feature

        :param predictors: List[str]
            Name of the predictor features

        :param target_type: str
            Abbreviate name of the target type / machine learning problem
                -> reg: Regression
                -> clf_binary: Binary Classification
                -> ltr_idiv: Learning to Rank
                -> ltr_mse: Learning to Rank
                -> ltr_pairwise: Learning to Rank

        :param input_layer_size: int
            Number of neurons in input layer
                -> stage 0: number of features
                -> stage 1 - x: number of features + number of neurons from last hidden layer of previous network

        :param train_data_path: str
            Complete file path of the training data

        :param test_data_path: str
            Complete file path of the testing data

        :param validation_data_path: str
            Complete file path of the validation data

        :param input_param: dict
            Pre-defined hyperparameter configuration

        :param model_param: dict
            Pre-configured model parameter

        :param seed: int
            Seed

        :param kwargs: dict
            Key-word arguments for configuring PyTorch parameter settings
        """
        self.target: str = target
        if target_type not in ['reg', 'clf_binary', 'clf_multi', 'ltr_idiv', 'ltr_mse', 'ltr_pairwise']:
            raise GrowNetException(f'Target type / machine learning problem {target_type} not supported')
        self.target_type: str = target_type
        self.output_size: int = 1
        self.predictors: List[str] = predictors
        self.input_size = len(self.predictors) if input_layer_size is None else input_layer_size
        self.train_data_path: str = train_data_path
        self.test_data_path: str = test_data_path
        self.validation_data_path: str = validation_data_path
        if len(self.train_data_path) == 0:
            raise GrowNetException('No training data found')
        if len(self.test_data_path) == 0:
            raise GrowNetException('No test data found')
        if len(self.validation_data_path) == 0:
            raise GrowNetException('No validation data found')
        self.mlp_models: List[MLP] = []
        self.ensemble_model: EnsembleNetwork = None
        self.input_param: dict = {} if input_param is None else input_param
        self.model_param: dict = {} if model_param is None else model_param
        self.model_param_mutated: dict = {}
        self.model_param_mutation: str = ''
        self.train_iter = None
        self.test_iter = None
        self.validation_iter = None
        self.seed: int = 1234 if seed <= 0 else seed
        self.kwargs: dict = kwargs

    def multi_layer_perceptron(self) -> MLP:
        """
        Generate Multi-Layer Perceptron (MLP) classifier parameter configuration randomly

        :return dict:
            Configured Multi-Layer Perceptron (MLP) hyperparameter set
        """
        return MLP(input_size=self.input_size, output_size=self.output_size, parameters=self.model_param)


class GrowNetGenerator(GrowNet):
    """
    Class for generating neural networks using PyTorch
    """
    def __init__(self,
                 target: str,
                 predictors: List[str],
                 target_type: str,
                 train_data_path: str = None,
                 test_data_path: str = None,
                 validation_data_path: str = None,
                 model_param: dict = None,
                 input_param: dict = None,
                 apply_rules: bool = True,
                 id_feature: str = None,
                 use_second_order_gradient_statistics: bool = True,
                 sep: str = ';',
                 cloud: str = None,
                 seed: int = 1234,
                 **kwargs
                 ):
        """
        :param apply_rules: bool
            Whether to apply rules to narrow down hyperparameter space building neural network architecture

        :param id_feature: str
            Name of the id feature (used for learn to rank use case only)

        :param use_second_order_gradient_statistics: bool
            Whether to use second order gradient statistics or first order gradient statistics

        :param sep: str
            Separator

        :param cloud: str
            Name of the cloud provider
                -> google: Google Cloud Storage
                -> aws: AWS Cloud

        :param seed: int
            Seed

        :param kwargs: dict
            Key-word arguments
        """
        super().__init__(target=target,
                         predictors=predictors,
                         target_type=target_type,
                         train_data_path=train_data_path,
                         test_data_path=test_data_path,
                         validation_data_path=validation_data_path,
                         model_param=model_param,
                         input_param=input_param,
                         seed=seed,
                         **kwargs
                         )
        self.id_feature: str = id_feature
        self.id: int = 0
        self.fitness_score: float = 0.0
        self.fitness: dict = {}
        self.train_time = None
        self.obs: list = []
        self.pred: list = []
        self.sep: str = sep
        self.cloud: str = cloud
        if self.cloud is None:
            self.bucket_name: str = None
        else:
            if self.cloud not in CLOUD_PROVIDER:
                raise GrowNetException('Cloud provider ({}) not supported'.format(cloud))
            self.bucket_name: str = self.train_data_path.split("//")[1].split("/")[0]
        self.loss_models: torch = None #torch.zeros((n_mlp_models, 3))
        self.apply_rules: bool = apply_rules
        self.use_second_order_gradient_statistics: bool = use_second_order_gradient_statistics
        self.dynamic_boosting_rates: List[float] = []
        self.middle_losses: List[float] = []
        self.batch_eval: Dict[str, dict] = dict(train=dict(ensemble_batch_loss=[],
                                                           batch_metric=[],
                                                           total_batch_loss=0,
                                                           total_batch_metric=0
                                                           ),
                                                val=dict(ensemble_batch_loss=[],
                                                         batch_metric=[],
                                                         total_batch_loss=0,
                                                         total_batch_metric=0
                                                         ),
                                                test=dict(ensemble_batch_loss=[],
                                                          batch_metric=[],
                                                          total_batch_loss=0,
                                                          total_batch_metric=0
                                                          )
                                                )
        self.epoch_eval: Dict[str, dict] = dict(train=dict(), val=dict(), test=dict())
        self._setup_param_space()

    def _apply_rules(self, batch_size_only: bool):
        """
        Apply rules for building neural networks using structured data as input

        :param batch_size_only: bool
            Whether to configure batch size for loading the data iterators only or not
        """
        if batch_size_only:
            # use GPU:
            if torch.cuda.is_available():
                self.model_param.update({'batch_size': np.random.choice(a=geometric_progression(n=2)),
                                         'accumulation': True,
                                         'accumulation_steps': np.random.choice(a=[8, 16, 32, 64, 128, 256, 512]),
                                         'normalization': 'layer'
                                         })
            else:
                self.model_param.update({'batch_size': np.random.choice(a=[16, 32, 64, 128, 256, 512, 1028, 2056]),
                                         'accumulation': False,
                                         'accumulation_steps': 0,
                                         'normalization': 'batch'
                                         })
        else:
            # use simple architecture:
            _hidden_layer_neurons: List[int] = geometric_progression(n=15)
            _threshold: int = 0
            for i in range(len(_hidden_layer_neurons), 0, -1):
                if _hidden_layer_neurons[i - 1] < self.train_iter.n_features:
                    _threshold = i - 1
                    break
            self.model_param.update({'num_hidden_layers': np.random.randint(low=1, high=3),
                                     'constant_hidden_layer_params': True,
                                     'hidden_layer_neurons': _hidden_layer_neurons[np.random.randint(low=0, high=_threshold)]
                                     })
            # minimize over-fitting:
            self.model_param.update({'n_mlp_models': np.random.randint(low=25, high=50),
                                     'epoch': np.random.randint(low=1, high=3),
                                     'corrective_epoch': np.random.randint(low=1, high=3),
                                     'dropout_rate': np.random.uniform(low=0.0, high=0.1),
                                     'use_alpha_dropout': False
                                     })
            # use other approved setting:
            self.model_param.update({'learning_rate': np.random.uniform(low=0.001, high=0.01),
                                     'activation': np.random.choice(a=['relu', 'leaky_relu']),
                                     'optimizer': np.random.choice(a=['adam', 'sgd']),
                                     'boosting_rate': 1.0,
                                     'weight_decay': np.random.uniform(low=0.0005, high=0.0015),
                                     'boosting_rate_scaler': np.random.randint(low=1, high=5),
                                     'learning_rate_corrective_steps': np.random.uniform(low=0.3, high=0.4),
                                     'sigma': 0.1,
                                     'gain_type': 'exp2'
                                     })

    def _batch_evaluation(self, iter_type: str, loss_value: float, metric_value: float):
        """
        Evaluation of each training batch

        :param iter_type: str
            Name of the iteration process:
                -> train: Training iteration
                -> test: Testing iteration
                -> val: Validation iteration

        :param loss_value: float
            Loss value of current batch

        :param metric_value: float
            Metric value of current batch
        """
        self.batch_eval[iter_type]['batch_loss'].append(loss_value)
        self.batch_eval[iter_type]['batch_metric'].append(metric_value)
        self.batch_eval[iter_type]['total_batch_loss'] += self.batch_eval[iter_type]['batch_loss'][-1]
        self.batch_eval[iter_type]['total_batch_metric'] += self.batch_eval[iter_type]['batch_metric'][-1]

    def _batch_learning(self, stage: int):
        """
        Train gradient using batch learning

        :param stage: int
            Stage number
        """
        _model: MLP = self.mlp_models[stage]
        for epoch in range(0, self.model_param.get('epoch'), 1):
            print('\nEpoch: {}'.format(epoch))
            if torch.cuda.is_available():
                _model.cuda()
            _predictions: List[int] = []
            _observations: List[int] = []
            if torch.cuda.is_available():
                self.ensemble_model.to_cuda()
            _optim: torch.optim = self.model_param.get('optimizer_mlp')
            _accumulation: int = 0
            for idx, batch in enumerate(self.train_iter.data_loader):
                _accumulation += 1
                _predictors, _target = batch
                if _target.size()[0] != self.model_param.get('batch_size'):
                    continue
                if torch.cuda.is_available():
                    _target = torch.as_tensor(_target, dtype=torch.float32).cuda().view(-1, 1)
                    _predictors = _predictors.cuda()
                else:
                    _target = torch.as_tensor(_target, dtype=torch.float32).view(-1, 1)
                _middle_feat, _out = self.ensemble_model.forward(x=_predictors, update_gradient=False)
                if torch.cuda.is_available():
                    _out = torch.as_tensor(_out, dtype=torch.float32).cuda().view(-1, 1)
                else:
                    _out = torch.as_tensor(_out, dtype=torch.float32).view(-1, 1)
                if self.target_type == 'reg':
                    _h: float = 1.0
                    _grad_direction: torch.Tensor = -(_out - _target)
                    _, _out = _model(_predictors, _middle_feat)
                    if torch.cuda.is_available():
                        _out = torch.as_tensor(data=_out, dtype=torch.float32).cuda().view(-1, 1)
                        _grad_direction = torch.as_tensor(data=_grad_direction, dtype=torch.float32).cuda().view(_out.shape[0], 1)
                    else:
                        _out = torch.as_tensor(data=_out, dtype=torch.float32).view(-1, 1)
                        _grad_direction = torch.as_tensor(data=_grad_direction, dtype=torch.float32).view(_out.shape[0], 1)
                    _loss: torch = self.model_param.get('loss_torch')()(self.ensemble_model.boost_rate * _out, _grad_direction)
                    _n: int = len(_target)
                else:
                    #_grad_direction: float = _target / (1.0 + torch.exp(_target * _out))
                    _h: float = 1 / ((1 + torch.exp(_target * _out)) * (1 + torch.exp(-_target * _out)))
                    _grad_direction: torch.Tensor = _target * (1.0 + torch.exp(-_target * _out))
                    _out = torch.as_tensor(_out)
                    nwtn_weights = (torch.exp(_out) + torch.exp(-_out)).abs()
                    _, _out = _model(_predictors, _middle_feat)
                    if torch.cuda.is_available():
                        _out = torch.as_tensor(_out, dtype=torch.float32).cuda().view(-1, 1)
                        _grad_direction = torch.as_tensor(data=_grad_direction, dtype=torch.float32).cuda().view(_out.shape[0], 1)
                    else:
                        _out = torch.as_tensor(_out, dtype=torch.float32).view(-1, 1)
                        _grad_direction = torch.as_tensor(data=_grad_direction, dtype=torch.float32).view(_out.shape[0], 1)
                    _loss: torch = self.model_param.get('loss_torch')()(self.ensemble_model.boost_rate * _out, _grad_direction)
                    _n: int = 1
                _loss = _loss * _h
                _loss = _loss.mean()
                _model.zero_grad()
                _loss.backward()
                if not self.model_param.get('accumulation') or _accumulation % self.model_param.get('accumulation_steps') == 0:
                    _optim.step()
                # self._clip_gradient(model=_model, clip_value=1e-1)
                self.middle_losses.append(_loss.item() * _n)
        self.ensemble_model.add(model=_model)

    def _batch_learning_ltr_non_pairwise(self, stage: int):
        """
        Train gradient using batch learning for non-pairwise ranking use case

        :param stage: int
            Stage number
        """
        _model: MLP = self.mlp_models[stage]
        _residual: torch.Tensor = None
        _second_gradient_order: torch.Tensor = None
        for epoch in range(0, self.model_param.get('epoch'), 1):
            print('\nEpoch: {}'.format(epoch))
            if torch.cuda.is_available():
                _model.cuda()
            _predictions: List[int] = []
            _observations: List[int] = []
            if torch.cuda.is_available():
                self.ensemble_model.to_cuda()
            _optim: torch.optim = self.model_param.get('optimizer_mlp')
            _accumulation: int = 0
            for query, target, predictors in self.train_iter.generate_query_batch(batch_size=self.model_param.get('batch_size'),
                                                                                  id_feature=self.id_feature,
                                                                                  predictors=self.predictors
                                                                                  ):
                _accumulation += 1
                if torch.cuda.is_available():
                    _target = torch.as_tensor(target + 1, dtype=torch.float32).cuda().view(-1, 1)
                    _predictors = torch.as_tensor(predictors, dtype=torch.float32).cuda()
                else:
                    _target = torch.as_tensor(target + 1, dtype=torch.float32).view(-1, 1)
                    _predictors = torch.as_tensor(predictors, dtype=torch.float32)
                _middle_feat, _out = self.ensemble_model.forward(x=_predictors, update_gradient=False)
                if torch.cuda.is_available():
                    _out = torch.as_tensor(_out, dtype=torch.float32).cuda().view(-1, 1)
                else:
                    _out = torch.as_tensor(_out, dtype=torch.float32).view(-1, 1)
                if self.target_type == 'ltr_idiv':
                    _out = torch.exp(_out)
                    _first_gradient_order: torch.Tensor = -(_target - _out)
                    _second_gradient_order: torch.Tensor = _out
                    if self.use_second_order_gradient_statistics:
                        _residual: torch.Tensor = -_first_gradient_order / _second_gradient_order
                    else:
                        _residual: torch.Tensor = -_first_gradient_order
                elif self.target_type == 'ltr_mse':
                    _residual: torch.Tensor = _target - _out
                _, _out = _model(_predictors, _middle_feat)
                if torch.cuda.is_available():
                    _out = torch.as_tensor(_out, dtype=torch.float32).cuda().view(-1, 1)
                else:
                    _out = torch.as_tensor(_out, dtype=torch.float32).view(-1, 1)
                _loss: torch = self.model_param.get('loss_torch')()(self.ensemble_model.boost_rate * _out, _residual)
                if self.target_type == 'ltr_idiv':
                    _loss = _second_gradient_order * _loss
                    _loss = _loss.mean()
                _model.zero_grad()
                _loss.backward()
                if not self.model_param.get('accumulation') or _accumulation % self.model_param.get('accumulation_steps') == 0:
                    _optim.step()
                # self._clip_gradient(model=_model, clip_value=1e-1)
        self.ensemble_model.add(model=_model)

    def _batch_learning_ltr_pairwise(self, stage: int):
        """
        Train gradient using batch learning for pairwise ranking use case

        :param stage: int
            Stage number
        """
        _model: MLP = self.mlp_models[stage]
        for epoch in range(0, self.model_param.get('epoch'), 1):
            print('\nEpoch: {}'.format(epoch))
            if torch.cuda.is_available():
                _model.cuda()
            _predictions: List[int] = []
            _observations: List[int] = []
            if torch.cuda.is_available():
                self.ensemble_model.to_cuda()
            _optim: torch.optim = self.model_param.get('optimizer_mlp')
            _accumulation: int = 0
            for query, target, predictors in self.train_iter.generate_query_batch(batch_size=self.model_param.get('batch_size'),
                                                                                  id_feature=self.id_feature,
                                                                                  predictors=self.predictors
                                                                                  ):
                _accumulation += 1
                _idx: List[str] = remove_single_queries(query=query, target=target)
                _query: np.array = query[_idx]
                _target: np.array = target[_idx]
                _predictors: np.array = predictors[_idx]
                if torch.cuda.is_available():
                    _predictors = torch.as_tensor(predictors, dtype=torch.float32).cuda()
                else:
                    _predictors = torch.as_tensor(predictors, dtype=torch.float32)
                _middle_feat, _out = self.ensemble_model.forward(x=_predictors, update_gradient=False)
                if torch.cuda.is_available():
                    _out = torch.as_tensor(_out, dtype=torch.float32).cuda().view(-1, 1)
                else:
                    _out = torch.as_tensor(_out, dtype=torch.float32).view(-1, 1)
                _unique_idx: np.array = np.unique(_query)
                _grad_batch: torch.Tensor = None
                for i in range(0, len(_unique_idx), 1):
                    _idx_i: np.array = np.where(_query == _unique_idx[i])[0]
                    _target_i: np.array = _target[_idx_i]
                    if torch.cuda.is_available():
                        _idx_i = torch.as_tensor(_idx_i, dtype=torch.float32).cuda()
                    else:
                        _idx_i = torch.as_tensor(_idx_i, dtype=torch.float32)
                    _out_i: torch.Tensor = torch.index_select(_out, 0, _idx_i)
                    _first_gradient_order, _second_gradient_order = grad_calc_(y_i, out_i, gain_type, opt.sigma, N, device)
                    if self.use_second_order_gradient_statistics:
                        _residual: torch.Tensor = -_first_gradient_order / _second_gradient_order
                    else:
                        _residual: torch.Tensor = -_first_gradient_order
                    if _grad_batch is None:
                        _grad_batch = _residual
                        _second_gradient_order_batch: torch.Tensor = _second_gradient_order
                    else:
                        _second_gradient_order_batch = torch.cat((_second_gradient_order_batch, _second_gradient_order), dim=0)
                        _grad_batch = torch.cat((_grad_batch, _residual), dim=0)
                _, _out = _model(_predictors, _middle_feat)
                if torch.cuda.is_available():
                    _out = torch.as_tensor(_out, dtype=torch.float32).cuda().view(-1, 1)
                else:
                    _out = torch.as_tensor(_out, dtype=torch.float32).view(-1, 1)
                _loss: torch = self.model_param.get('loss_torch')()(self.ensemble_model.boost_rate * _out, _residual)
                _loss = _second_gradient_order * _loss
                _loss = _loss.mean()
                _model.zero_grad()
                _loss.backward()
                _optim.step()
                # self._clip_gradient(model=_model, clip_value=1e-1)
                if not self.model_param.get('accumulation') or _accumulation % self.model_param.get('accumulation_steps') == 0:
                    _optim.step()
                # self._clip_gradient(model=_model, clip_value=1e-1)
        self.ensemble_model.add(model=_model)

    def _calculate_gradient_ltr_pairwise(self, obs: np.array, pred: np.array):
        #rank_df = pd.DataFrame(data={"y": obs, "doc": np.arange(obs.shape[0])})
        #rank_df = rank_df.sort_values("y").reset_index(drop=True)
        pos_pairs_score_diff = 1.0 / (1.0 + torch.exp(self.model_param.get('sigma') * (pred - pred.t())))
        y_tensor = torch.tensor(obs, dtype=torch.float32).view(-1, 1)
        rel_diff = y_tensor - y_tensor.t()
        pos_pairs = (rel_diff > 0).type(torch.float32)
        neg_pairs = (rel_diff < 0).type(torch.float32)
        Sij = pos_pairs - neg_pairs
        grad_ord1 = self.model_param.get('sigma') * (0.5 * (1 - Sij) - pos_pairs_score_diff)
        grad_ord2 = self.model_param.get('sigma') * self.model_param.get('sigma') * pos_pairs_score_diff * (1 - pos_pairs_score_diff)
        return torch.sum(grad_ord1, 1, keepdim=True), torch.sum(grad_ord2, 1, keepdim=True)

    def _calculate_loss_ltr_pairwise(self, obs: np.array, pred: np.array):
        rank_df = pd.DataFrame({"y": y_true, "doc": np.arange(y_true.shape[0])})
        rank_df = rank_df.sort_values("y").reset_index(drop=True)
        rank_order = rank_df.sort_values("doc").index.values + 1

        pos_pairs_score_diff = 1.0 / (1.0 + torch.exp(self.model_param.get('sigma') * (y_pred - y_pred.t())))
        y_tensor = torch.tensor(y_true, dtype=torch.float32, device=device).view(-1, 1)

        if self.model_param.get('gain_type') == "exp2":
            gain_diff = torch.pow(2.0, y_tensor) - torch.pow(2.0, y_tensor.t())
        elif self.model_param.get('gain_type') == "identity":
            gain_diff = y_tensor - y_tensor.t()
        else:
            gain_diff = None
        rank_order_tensor = torch.tensor(rank_order, dtype=torch.float32, device=device).view(-1, 1)
        decay_diff = 1.0 / torch.log2(rank_order_tensor + 1.0) - 1.0 / torch.log2(rank_order_tensor.t() + 1.0)
        delta_ndcg = torch.abs(N * gain_diff * decay_diff)
        grad_ord1 = self.model_param.get('sigma') * (-pos_pairs_score_diff * delta_ndcg)
        grad_ord2 = (self.model_param.get('sigma') * self.model_param.get('sigma')) * pos_pairs_score_diff * (1 - pos_pairs_score_diff) * delta_ndcg
        return torch.sum(grad_ord1, 1, keepdim=True), torch.sum(grad_ord2, 1, keepdim=True)

    @staticmethod
    def _clip_gradient(model: torch.nn, clip_value: float):
        """
        Clip gradient during network training

        :param model:

        :param clip_value: float
            Clipping threshold
        """
        params = list(filter(lambda p: p.grad is not None, model.parameters()))
        for p in params:
            p.grad.data.clamp_(-clip_value, clip_value)

    def _config_activation_functions(self):
        """
        Configure activation functions
        """
        if self.model_param.get('num_hidden_layers') > 0:
            _param_names: List[str] = ['hidden_layer_{}_activation'.format(hl) for hl in range(1, self.model_param.get('num_hidden_layers') + 1, 1)]
        else:
            _param_names: List[str] = ['activation_torch']
        _activation_torch: dict = dict()
        for layer in range(0, self.model_param.get('num_hidden_layers'), 1):
            if self.model_param.get('activation') == 'celu':
                _activation_torch.update({_param_names[layer]: ACTIVATION['weighted_sum'][self.model_param.get('activation')](alpha=1.0 if self.model_param.get('alpha') is None else self.model_param.get('alpha'))})
            elif self.model_param.get('activation') == 'elu':
                _activation_torch.update({_param_names[layer]: ACTIVATION['weighted_sum'][self.model_param.get('activation')](alpha=1.0 if self.model_param.get('alpha') is None else self.model_param.get('alpha'))})
            elif self.model_param.get('activation') == 'hard_shrink':
                _activation_torch.update({_param_names[layer]: ACTIVATION['weighted_sum'][self.model_param.get('activation')](lambd=0.5 if self.model_param.get('lambd') is None else self.model_param.get('lambd'))})
            elif self.model_param.get('activation') == 'hard_tanh':
                _activation_torch.update({_param_names[layer]: ACTIVATION['weighted_sum'][self.model_param.get('activation')](min_val=-1.0 if self.model_param.get('min_val') is None else self.model_param.get('min_val'),
                                                                                                                              max_val=1.0 if self.model_param.get('max_val') is None else self.model_param.get('max_val')
                                                                                                                              )
                                          })
            elif self.model_param.get('activation') == 'leaky_relu':
                _activation_torch.update({_param_names[layer]: ACTIVATION['weighted_sum'][self.model_param.get('activation')](negative_slope=0.01 if self.model_param.get('negative_slope') is None else self.model_param.get('negative_slope'))})
            elif self.model_param.get('activation') == 'prelu':
                _activation_torch.update({_param_names[layer]: ACTIVATION['weighted_sum'][self.model_param.get('activation')](num_parameters=1 if self.model_param.get('num_parameters') is None else self.model_param.get('num_parameters'),
                                                                                                                              init=0.25 if self.model_param.get('init') is None else self.model_param.get('init')
                                                                                                                              )
                                          })
            elif self.model_param.get('activation') == 'rrelu':
                _activation_torch.update({_param_names[layer]: ACTIVATION['weighted_sum'][self.model_param.get('activation')](lower=0.125 if self.model_param.get('lower') is None else self.model_param.get('lower'),
                                                                                                                              upper=0.3333333333333333 if self.model_param.get('upper') is None else self.model_param.get('upper')
                                                                                                                              )
                                          })
            elif self.model_param.get('activation') == 'soft_plus':
                _activation_torch.update({_param_names[layer]: ACTIVATION['weighted_sum'][self.model_param.get('activation')](beta=1 if self.model_param.get('beta') is None else self.model_param.get('beta'),
                                                                                                                              threshold=20 if self.model_param.get('threshold') is None else self.model_param.get('threshold')
                                                                                                                              )
                                          })
            elif self.model_param.get('activation') == 'soft_shrink':
                _activation_torch.update({_param_names[layer]: ACTIVATION['weighted_sum'][self.model_param.get('activation')](lambd=0.5 if self.model_param.get('lambd') is None else self.model_param.get('lambd'))})
            else:
                self.model_param.update({_param_names[layer]: ACTIVATION['weighted_sum'][self.model_param.get('activation')]()})
        self.model_param.update(_activation_torch)

    def _config_hidden_layers(self):
        """
        Configure hyperparameter of the hidden layers
        """
        if self.model_param.get('num_hidden_layers') is None or self.model_param.get('num_hidden_layers') <= 0:
            self.model_param['num_hidden_layers'] = np.random.randint(low=self.model_param.get('num_hidden_layers_low'),
                                                                      high=self.model_param.get('num_hidden_layers_high')
                                                                      )
        _hidden_layer_settings: dict = self._get_param_space()
        for hidden in range(1, self.model_param['num_hidden_layers'] + 1, 1):
            if not self.model_param.get('constant_hidden_layer_params') and hidden > 1:
                _hidden_layer_settings: dict = self._get_param_space()
            self.model_param.update({'hidden_layer_{}_neurons'.format(hidden): _hidden_layer_settings.get('hidden_neurons')})
            self.model_param.update({'hidden_layer_{}_dropout_rate'.format(hidden): _hidden_layer_settings.get('dropout_rate')})
            self.model_param.update({'hidden_layer_{}_use_alpha_dropout'.format(hidden): _hidden_layer_settings.get('use_alpha_dropout')})
            self.model_param.update({'hidden_layer_{}_activation'.format(hidden): _hidden_layer_settings.get('activation')})

    def _config_optimizer(self, params, ensemble_network: bool):
        """
        Configure optimizer for ensemble network or multi-layer perceptron

        :param params
            Network parameters

        :param ensemble_network: bool
            Whether to configure optimizer for ensemble network or multi-layer perceptron
        """
        _model_param_name: str = 'optimizer_ensemble' if ensemble_network else 'optimizer_mlp'
        _learning_rate_param_name: str = 'boosting_rate' if ensemble_network else 'learning_rate'
        if self.model_param.get('optimizer') == 'sgd':
            self.model_param.update(
                {_model_param_name: torch.optim.SGD(params=params,
                                                    lr=self.model_param.get(_learning_rate_param_name),
                                                    momentum=0 if self.model_param.get(
                                                        'momentum') is None else self.model_param.get('momentum'),
                                                    dampening=0 if self.model_param.get(
                                                        'dampening') is None else self.model_param.get('dampening'),
                                                    weight_decay=0 if self.model_param.get(
                                                        'weight_decay') is None else self.model_param.get(
                                                        'weight_decay'),
                                                    nesterov=False if self.model_param.get(
                                                        'nesterov') is None else self.model_param.get('nesterov')
                                                    )
                 })
        elif self.model_param.get('optimizer') == 'rmsprop':
            self.model_param.update({_model_param_name: torch.optim.RMSprop(
                params=params,
                lr=self.model_param.get(_learning_rate_param_name),
                alpha=0.99 if self.model_param.get('alpha') is None else self.model_param.get('alpha'),
                eps=1e-08 if self.model_param.get('eps') is None else self.model_param.get('eps'),
                weight_decay=0 if self.model_param.get('weight_decay') is None else self.model_param.get(
                    'weight_decay'),
                momentum=0 if self.model_param.get('momentum') is None else self.model_param.get('momentum'),
                centered=False if self.model_param.get('centered') is None else self.model_param.get('centered')
                )
                                     })
        elif self.model_param.get('optimizer') == 'adam':
            self.model_param.update(
                {_model_param_name: torch.optim.Adam(params=params,
                                                     lr=self.model_param.get(_learning_rate_param_name),
                                                     betas=(0.9, 0.999) if self.model_param.get(
                                                         'betas') is None else self.model_param.get('betas'),
                                                     eps=1e-08 if self.model_param.get(
                                                         'eps') is None else self.model_param.get('eps'),
                                                     weight_decay=0.0 if self.model_param.get(
                                                         'weight_decay') is None else self.model_param.get(
                                                         'weight_decay'),
                                                     amsgrad=False if self.model_param.get(
                                                         'amsgrad') is None else self.model_param.get('amsgrad')
                                                     )
                 })

    def _config_params(self,
                       loss: bool = False,
                       activation: bool = False,
                       hidden_layers: bool = False
                       ):
        """
        Finalize configuration of hyperparameter settings of the neural network

        :param loss: bool
            Configure loss function initially based on the size of the output layer

        :param activation: bool
            Configure activation functions for all layers

        :param hidden_layers: bool
            Configure hidden_layers initially
        """
        if hidden_layers:
            self._config_hidden_layers()
        if activation:
            self._config_activation_functions()
        if loss:
            self.model_param.update(dict(sigma=np.random.uniform(low=0.05, high=0.5),
                                         gain_type=np.random.choice(a=['exp2', 'identity'])
                                         )
                                    )
            if self.model_param.get('loss') is None:
                _loss: str = np.random.choice(a=list(LOSS.get('reg').keys()))
            else:
                _loss: str = self.model_param.get('loss')
            self.model_param.update(dict(loss_torch=LOSS[self.target_type][_loss]))
            if self.model_param.get('loss_corrective') is None:
                _loss_corrective: str = np.random.choice(a=list(LOSS.get(self.target_type).keys()))
            else:
                _loss_corrective: str = self.model_param.get('loss_corrective')
            self.model_param.update(dict(loss_corrective=LOSS['reg'][_loss_corrective]))

    def _corrective_step(self):
        """
        Step to correct ensemble network learning
        """
        _optimizer: torch.optim = self.model_param.get('optimizer_ensemble')
        for epoch in range(0, self.model_param.get('corrective_epoch'), 1):
            print('\nCorrective Epoch: {}'.format(epoch))
            stage_loss = []
            _accumulation: int = 0
            for idx, batch in enumerate(self.train_iter.data_loader):
                _accumulation += 1
                _predictors, _target = batch
                if _target.size()[0] != self.model_param.get('batch_size'):
                    continue
                if torch.cuda.is_available():
                    _target = torch.as_tensor(_target, dtype=torch.float32).cuda().view(-1, 1)
                    _predictors = _predictors.cuda()
                else:
                    _target = torch.as_tensor(_target, dtype=torch.float32).view(-1, 1)
                _, _out = self.ensemble_model.forward(x=_predictors, update_gradient=True)
                if torch.cuda.is_available():
                    _out = torch.as_tensor(_out, dtype=torch.float32).cuda().view(-1, 1)
                else:
                    _out = torch.as_tensor(_out, dtype=torch.float32).view(-1, 1)
                if self.target_type == 'reg':
                    _loss: torch = self.model_param.get('loss_corrective')()(_out, _target)
                    _n: int = len(_target)
                else:
                    _target = (_target + 1.0) / 2.0
                    _loss: torch = self.model_param.get('loss_corrective')()(_out, _target).mean()
                    _n: int = 1
                _optimizer.zero_grad()
                _loss.backward()
                if not self.model_param.get('accumulation') or _accumulation % self.model_param.get('accumulation_steps') == 0:
                    _optimizer.step()
                stage_loss.append(_loss.item() * _n)

    def _corrective_step_ltr_non_pairwise(self):
        """
        Step to correct ensemble network learning for non-pairwise ranking use case
        """
        _loss: torch = None
        _optimizer: torch.optim = self.model_param.get('optimizer_ensemble')
        for epoch in range(0, self.model_param.get('corrective_epoch'), 1):
            print('\nCorrective Epoch: {}'.format(epoch))
            stage_loss = []
            _accumulation: int = 0
            for query, target, predictors in self.train_iter.generate_query_batch(
                    batch_size=self.model_param.get('batch_size'),
                    id_feature=self.id_feature,
                    predictors=self.predictors
            ):
                _accumulation += 1
                if torch.cuda.is_available():
                    _target = torch.as_tensor(target, dtype=torch.float32).cuda().view(-1, 1)
                    _predictors = torch.as_tensor(predictors, dtype=torch.float32).cuda()
                else:
                    _target = torch.as_tensor(target, dtype=torch.float32).view(-1, 1)
                    _predictors = torch.as_tensor(predictors, dtype=torch.float32)
                _, _out = self.ensemble_model.forward(x=_predictors, update_gradient=True)
                if torch.cuda.is_available():
                    _out = torch.as_tensor(_out, dtype=torch.float32).cuda().view(-1, 1)
                else:
                    _out = torch.as_tensor(_out, dtype=torch.float32).view(-1, 1)
                if self.target_type == 'ltr_idiv':
                    _out = torch.exp(_out)
                    _loss: torch = torch.mean(_target * torch.log(_target / _out) - (_target - _out))
                elif self.target_type == 'ltr_mse':
                    _loss: torch = self.model_param.get('loss_torch')()(_out, _target)
                _optimizer.zero_grad()
                _loss.backward()
                if not self.model_param.get('accumulation') or _accumulation % self.model_param.get('accumulation_steps') == 0:
                    _optimizer.step()
                #stage_loss.append(_loss.item() * _n)

    def _corrective_step_ltr_pairwise(self):
        """
        Step to correct ensemble network learning for pairwise ranking use case
        """
        _loss: torch = None
        _optimizer: torch.optim = self.model_param.get('optimizer_ensemble')
        for epoch in range(0, self.model_param.get('corrective_epoch'), 1):
            print('\nCorrective Epoch: {}'.format(epoch))
            stage_loss = []
            _accumulation: int = 0
            for query, target, predictors in self.train_iter.generate_query_batch(
                    batch_size=self.model_param.get('batch_size'),
                    id_feature=self.id_feature,
                    predictors=self.predictors
            ):
                _accumulation += 1

                if not self.model_param.get('accumulation') or _accumulation % self.model_param.get('accumulation_steps') == 0:
                    _optimizer.step()
                #stage_loss.append(_loss.item() * _n)

    def _epoch_eval(self, iter_types: List[str]):
        """
        Evaluation of each training epoch

        :param iter_types: List[str]
            Names of the iteration process:
                -> train: Training iteration
                -> test: Testing iteration
                -> val: Validation iteration
        """
        for iter_type in iter_types:
            for metric in self.fitness[iter_type].keys():
                if metric in self.epoch_eval[iter_type].keys():
                    self.epoch_eval[iter_type][metric].append(self.fitness[iter_type][metric])
                else:
                    self.epoch_eval[iter_type].update({metric: [self.fitness[iter_type][metric]]})

    def _eval(self, data_set_type: str):
        """
        Evaluate pre-trained model

        :param data_set_type: str
            Name of the data set type
                -> train: Training data
                -> val: Validation data
                -> test: Testing data
        """
        self.ensemble_model.to_eval()
        if self.target_type.find('ltr') >= 0:
            _eval = EvalLTR
        elif self.target_type == 'reg':
            _eval = EvalReg
        else:
            _eval = EvalClf
        _target_values: list = []
        _prediction_values: list = []
        if data_set_type == 'train':
            _data_loader = self.train_iter.data_loader
        elif data_set_type == 'val':
            _data_loader = self.validation_iter.data_loader
        else:
            _data_loader = self.test_iter.data_loader
        for idx, batch in enumerate(_data_loader):
            _predictors, _target = batch
            if _target.size()[0] != self.model_param.get('batch_size'):
                continue
            if torch.cuda.is_available():
                _target = torch.as_tensor(_target, dtype=torch.float32).cuda().view(-1, 1)
                _predictors = _predictors.cuda()
            else:
                _target = torch.as_tensor(_target, dtype=torch.float32).view(-1, 1)
            with torch.no_grad():
                _, _pred = self.ensemble_model.forward(x=_predictors, update_gradient=False)
            if torch.cuda.is_available():
                _pred = torch.as_tensor(_pred, dtype=torch.float32).cuda().view(-1, 1)
            else:
                _pred = torch.as_tensor(_pred, dtype=torch.float32).view(-1, 1)
            _target_values.extend(_target.squeeze().tolist())
            if self.target_type == 'reg':
                _prediction_values.extend(_pred.squeeze().tolist())
            else:
                _prob: torch.nn = 1.0 - 1.0 / torch.exp(_pred)
                _prediction_values.extend([round(p, ndigits=0) for p in _prob.squeeze().tolist()])
        self._get_fitness_score(iter_type=data_set_type, obs=_target_values, pred=_prediction_values)

    def _get_c0(self) -> float:
        """
        Get c0 value for the ensemble network
        """
        if self.output_size == 1:
            return float(np.mean(self.train_iter.target_values))
        else:
            _positive: int = 0
            _negative: int = 0
            for value in self.train_iter.target_values:
                if value > 0:
                    _positive += 1
                else:
                    _negative += 1
            return float(np.log(_positive / _negative))

    def _get_fitness_score(self, iter_type: str, obs: list, pred: list):
        """
        Internal evaluation for applying machine learning metric methods

        :param iter_type: str
            Name of the iteration process:
                -> train: Training iteration
                -> test: Testing iteration
                -> val: Validation iteration

        :param obs: list
            Observations of target feature

        :param pred: list
            Predictions
        """
        self.fitness.update({iter_type: {}})
        if self.target_type == 'reg':
            _eval = EvalReg
            _eval_metric: List[str] = ML_METRIC.get('reg')
        elif self.target_type.find('clf'):
            _eval = EvalClf
            _eval_metric: List[str] = ML_METRIC.get('clf_binary')
        else:
            _eval = EvalLTR
            _eval_metric: List[str] = ML_METRIC.get('ltr')
        for metric in _eval_metric:
            self.fitness[iter_type].update({metric: copy.deepcopy(getattr(_eval(obs=np.array(obs),
                                                                                pred=np.array(pred)
                                                                                ),
                                                                          metric
                                                                          )()
                                                                  )
                                            })

    def _get_param_space(self) -> dict:
        """
        Get randomly drawn hyperparameter settings

        :return dict:
            Hyperparameter settings
        """
        return dict(n_mlp_models=np.random.randint(low=self.model_param.get('n_mlp_models_low'),
                                                   high=self.model_param.get('n_mlp_models_high')
                                                   ) if self.kwargs.get('n_mlp_models') is None else self.kwargs.get('n_mlp_models'),
                    weight_decay=np.random.uniform(low=self.model_param.get('weight_decay_low'),
                                                   high=self.model_param.get('weight_decay_low')
                                                   ) if self.kwargs.get('weight_decay') is None else self.kwargs.get('weight_decay'),
                    boosting_rate=np.random.uniform(low=self.model_param.get('boosting_rate_low'),
                                                    high=self.model_param.get('boosting_rate_high')
                                                    ) if self.kwargs.get('boosting_rate') is None else self.kwargs.get('boosting_rate'),
                    learning_rate_corrective_steps=np.random.uniform(low=self.model_param.get('learning_rate_corrective_steps_low'),
                                                                     high=self.model_param.get('learning_rate_corrective_steps_high')
                                                                     ) if self.kwargs.get('learning_rate_corrective_steps') is None else self.kwargs.get('learning_rate_corrective_steps'),
                    corrective_epoch=np.random.randint(low=self.model_param.get('corrective_epoch_low'),
                                                       high=self.model_param.get('corrective_epoch_high')
                                                       ) if self.kwargs.get('corrective_epoch') is None else self.kwargs.get('corrective_epoch'),
                    boosting_rate_scaler=np.random.randint(low=self.model_param.get('boosting_rate_scaler_low'),
                                                           high=self.model_param.get('boosting_rate_scaler_high')
                                                           ) if self.kwargs.get('boosting_rate_scaler') is None else self.kwargs.get('boosting_rate_scaler'),
                    epoch=np.random.randint(low=self.kwargs.get('epoch_low'),
                                            high=self.kwargs.get('epoch_high')
                                            ) if self.kwargs.get('epoch') is None else self.kwargs.get('epoch'),
                    hidden_neurons=np.random.choice(a=geometric_progression(n=12)) if self.kwargs.get('hidden_neurons') is None else self.kwargs.get('hidden_neurons'),
                    constant_hidden_layer_params=np.random.choice(a=[False, True]) if self.kwargs.get('constant_hidden_layer_params') is None else self.kwargs.get('constant_hidden_layer_params'),
                    learning_rate=np.random.uniform(low=self.model_param.get('learning_rate_low'),
                                                    high=self.model_param.get('learning_rate_high')
                                                    ) if self.kwargs.get('learning_rate') is None else self.kwargs.get('learning_rate'),
                    batch_size=np.random.choice(a=geometric_progression(n=8)) if self.kwargs.get('batch_size') is None else self.kwargs.get('batch_size'),
                    normalization=np.random.choice(a=['batch', 'layer']) if self.kwargs.get('normalization') is None else self.kwargs.get('normalization'),
                    accumulation=np.random.choice(a=[False, True]) if self.kwargs.get('accumulation') is None else self.kwargs.get('accumulation'),
                    accumulation_step=np.random.randint(low=self.kwargs.get('accumulation_step_low'),
                                                        high=self.kwargs.get('accumulation_step_high')
                                                        ) if self.kwargs.get('accumulation_step') is None else self.kwargs.get('accumulation_step'),
                    #early_stopping=[False, True],
                    #patience=np.random.uniform(low=2, high=20),
                    dropout_rate=np.random.uniform(low=self.kwargs.get('dropout_rate_low'),
                                                   high=self.kwargs.get('dropout_rate_high')
                                                   ) if self.kwargs.get('dropout_rate') is None else self.kwargs.get('dropout_rate'),
                    use_alpha_dropout=np.random.choice(a=[False, True]) if self.kwargs.get('use_alpha_dropout') is None else self.kwargs.get('use_alpha_dropout'),
                    use_sparse=np.random.choice(a=[False, True]) if self.kwargs.get('use_sparse') is None else self.kwargs.get('use_sparse'),
                    activation=np.random.choice(a=list(ACTIVATION['weighted_sum'].keys())) if self.kwargs.get('activation') is None else self.kwargs.get('activation'),
                    optimizer=np.random.choice(a=list(OPTIMIZER.keys())) if self.kwargs.get('optimizer') is None else self.kwargs.get('optimizer')
                    )

    def _import_data_torch(self):
        """
        Import data sets (Training, Testing, Validation) from file
        """
        if self.target_type.find('ltr') >= 0:
            self.train_iter = LoadTabularData(file_path=self.train_data_path,
                                              sep=self.sep,
                                              cloud=self.cloud,
                                              bucket_name=self.bucket_name,
                                              target_name=self.target
                                              )
            self.test_iter = LoadTabularData(file_path=self.train_data_path,
                                             sep=self.sep,
                                             cloud=self.cloud,
                                             bucket_name=self.bucket_name,
                                             target_name=self.target
                                             )
            self.validation_iter = LoadTabularData(file_path=self.train_data_path,
                                                   sep=self.sep,
                                                   cloud=self.cloud,
                                                   bucket_name=self.bucket_name,
                                                   target_name=self.target
                                                   )
        else:
            self.train_iter = LoadTabularData(file_path=self.train_data_path,
                                              sep=self.sep,
                                              cloud=self.cloud,
                                              bucket_name=self.bucket_name,
                                              target_name=self.target
                                              )
            self.train_iter.load(batch_size=int(self.model_param.get('batch_size')), shuffle=True)
            self.test_iter = LoadTabularData(file_path=self.train_data_path,
                                             sep=self.sep,
                                             cloud=self.cloud,
                                             bucket_name=self.bucket_name,
                                             target_name=self.target
                                             )
            self.test_iter.load(batch_size=int(self.model_param.get('batch_size')), shuffle=True)
            self.validation_iter = LoadTabularData(file_path=self.train_data_path,
                                                   sep=self.sep,
                                                   cloud=self.cloud,
                                                   bucket_name=self.bucket_name,
                                                   target_name=self.target
                                                   )
            self.validation_iter.load(batch_size=int(self.model_param.get('batch_size')), shuffle=True)

    def _setup_param_space(self):
        """
        Setup neurol network architecture hyperparameter space
        """
        if self.kwargs.get('n_mlp_models_low') is None:
            self.kwargs['n_mlp_models_low'] = 10
        if self.kwargs.get('n_mlp_models_high') is None:
            self.kwargs['n_mlp_models_high'] = 500
        if self.kwargs.get('weight_decay_low') is None:
            self.kwargs['weight_decay_low'] = 0.0001
        if self.kwargs.get('weight_decay_high') is None:
            self.kwargs['weight_decay_high'] = 0.1
        if self.kwargs.get('boosting_rate_low') is None:
            self.kwargs['boosting_rate_low'] = 0.1
        if self.kwargs.get('boosting_rate_high') is None:
            self.kwargs['boosting_rate_high'] = 1.0
        if self.kwargs.get('learning_rate_corrective_steps_low') is None:
            self.kwargs['learning_rate_corrective_steps_low'] = 0.1
        if self.kwargs.get('learning_rate_corrective_steps_high') is None:
            self.kwargs['learning_rate_corrective_steps_high'] = 0.5
        if self.kwargs.get('corrective_epoch_low') is None:
            self.kwargs['corrective_epoch_low'] = 1
        if self.kwargs.get('corrective_epoch_high') is None:
            self.kwargs['corrective_epoch_high'] = 5
        if self.kwargs.get('boosting_rate_scaler_low') is None:
            self.kwargs['boosting_rate_scaler_low'] = 1
        if self.kwargs.get('boosting_rate_scaler_high') is None:
            self.kwargs['boosting_rate_scaler_high'] = 6
        if self.kwargs.get('num_hidden_layers_low') is None:
            self.kwargs['num_hidden_layers_low'] = 10
        if self.kwargs.get('num_hidden_layers_high') is None:
            self.kwargs['num_hidden_layers_high'] = 500
        if self.kwargs.get('learning_rate_low') is None:
            self.kwargs['learning_rate_low'] = 0.0001
        if self.kwargs.get('learning_rate_high') is None:
            self.kwargs['learning_rate_high'] = 0.4
        if self.kwargs.get('accumulation_step_low') is None:
            self.kwargs['accumulation_step_low'] = 10
        if self.kwargs.get('accumulation_step_high') is None:
            self.kwargs['accumulation_step_high'] = 1000
        if self.kwargs.get('epoch_low') is None:
            self.kwargs['epoch_low'] = 1
        if self.kwargs.get('epoch_high') is None:
            self.kwargs['epoch_high'] = 5
        if self.kwargs.get('dropout_rate_low') is None:
            self.kwargs['dropout_rate_low'] = 0.0
        if self.kwargs.get('dropout_rate_high') is None:
            self.kwargs['dropout_rate_high'] = 0.4
        if self.kwargs.get('alpha_dropout_rate_low') is None:
            self.kwargs['alpha_dropout_rate_low'] = 0.0
        if self.kwargs.get('alpha_dropout_rate_high') is None:
            self.kwargs['alpha_dropout_rate_high'] = 0.4
        if self.kwargs.get('hidden_neurons_high') is None:
            self.kwargs['hidden_neurons_high'] = 9
        if self.kwargs.get('batch_size_high') is None:
            self.kwargs['batch_size_high'] = 10
        if self.kwargs.get('num_hidden_layers_high') is None:
            self.kwargs['num_hidden_layers_high'] = 10

    def generate_model(self):
        """
        Generate supervised machine learning model with randomized parameter configuration
        """
        if len(self.input_param.keys()) == 0:
            if self.apply_rules:
                self._apply_rules(batch_size_only=True)
            else:
                self.model_param.update(self._get_param_space())
                self._config_params(loss=True,
                                    activation=True,
                                    hidden_layers=True
                                    )
        else:
            self.model_param = copy.deepcopy(self.input_param)
        if len(self.predictors) > 0:
            if self.target != '':
                self._import_data_torch()
                if self.apply_rules:
                    self._apply_rules(batch_size_only=False)
                    self._config_params(loss=True,
                                        activation=True,
                                        hidden_layers=True
                                        )
            else:
                raise GrowNetException('No target feature found')
        else:
            raise GrowNetException('No predictors found')
        _idx: int = 0 if len(self.model_param_mutated.keys()) == 0 else len(self.model_param_mutated.keys()) + 1
        self.model_param_mutated.update({str(_idx): {'multi_layer_perceptron': {}}})
        for param in list(self.model_param.keys()):
            self.model_param_mutated[str(_idx)]['multi_layer_perceptron'].update({param: copy.deepcopy(self.model_param.get(param))})
        self.model_param_mutation = 'new_model'
        for i in range(0, self.model_param.get('n_mlp_models'), 1):
            if i > 0:
                self.input_size = len(self.predictors) + self.model_param.get(f'hidden_layer_{self.model_param.get("num_hidden_layers")}_neurons')
            self.mlp_models.append(getattr(GrowNet(target=self.target,
                                                   predictors=self.predictors,
                                                   input_layer_size=self.input_size,
                                                   output_layer_size=self.output_size,
                                                   train_data_path=self.train_data_path,
                                                   test_data_path=self.test_data_path,
                                                   validation_data_path=self.validation_data_path,
                                                   model_param=self.model_param,
                                                   **self.kwargs
                                                   ),
                                           'multi_layer_perceptron'
                                           )()
                                   )
        self.ensemble_model = EnsembleNetwork(c0=self._get_c0(),
                                              learning_rate=self.model_param.get('boosting_rate')
                                              )

    def generate_params(self, param_rate: float = 0.1, force_param: dict = None):
        """
        Generate parameter for supervised learning models

        :param param_rate: float
            Rate of parameters of each model to mutate

        :param force_param: dict
            Parameter config to force explicitly
        """
        if param_rate > 1:
            _rate: float = 1.0
        else:
            if param_rate > 0:
                _rate: float = param_rate
            else:
                _rate: float = 0.1
        _params: dict = getattr(GrowNet(target=self.target,
                                        predictors=self.predictors,
                                        output_layer_size=self.output_size,
                                        train_data_path=self.train_data_path,
                                        test_data_path=self.test_data_path,
                                        validation_data_path=self.validation_data_path,
                                        input_param=self.input_param,
                                        model_param=self.model_param,
                                        seed=self.seed,
                                        **self.kwargs
                                        ),
                                'multi_layer_perceptron_param'
                                )()
        for fixed in ['hidden_layers', 'hidden_layer_size_category']:
            if fixed in list(self.model_param.keys()):
                del _params[fixed]
        _force_param: dict = {} if force_param is None else force_param
        _param_choices: List[str] = [p for p in list(_params.keys()) if p not in list(_force_param.keys())]
        _gen_n_params: int = round(len(_params.keys()) * _rate)
        if _gen_n_params == 0:
            _gen_n_params = 1
        self.model_param_mutated.update({len(self.model_param_mutated.keys()) + 1: {'multi_layer_perceptron_param': {}}})
        for param in list(_force_param.keys()):
            self.model_param.update({param: copy.deepcopy(_force_param.get(param))})
        _old_model_param: dict = copy.deepcopy(self.model_param)
        _ignore_param: List[str] = IGNORE_PARAM_FOR_OPTIMIZATION
        if self.learning_type == 'batch':
            _ignore_param.append('batch_size')
        elif self.learning_type == 'stochastic':
            _ignore_param.append('sample_size')
        _parameters: List[str] = [p for p in _param_choices if p not in _ignore_param]
        for _ in range(0, _gen_n_params, 1):
            while True:
                _param: str = np.random.choice(a=_parameters)
                if _old_model_param.get(_param) is not None:
                    if self.model_param.get(_param) is not None:
                        break
            if _param == 'loss':
                self._config_params(loss=True)
            elif _param == 'hidden_layers':
                self._config_params(hidden_layers=True)
            else:
                if _param in self._get_param_space().keys():
                    self.model_param.update({_param: copy.deepcopy(self._get_param_space().get(_param))})
                elif _param in self._get_param_space().keys():
                    self.model_param.update({_param: copy.deepcopy(self._get_param_space().get(_param))})
            self.model_param_mutated[list(self.model_param_mutated.keys())[-1]]['multi_layer_perceptron_param'].update({_param: self.model_param.get(_param)})
        self.model_param_mutation = 'new_model'
        if len(self.predictors) > 0:
            if self.target != '':
                self._import_data_torch()
            else:
                raise GrowNetException('No target feature found')
        else:
            raise GrowNetException('No predictors found')
        for i in range(0, self.model_param.get('n_mlp_models'), 1):
            if i > 0:
                self.input_size = len(self.predictors) + self.model_param.get(f'hidden_layer_{self.model_param.get("num_hidden_layers")}_neurons')
            self.mlp_models.append(getattr(GrowNet(target=self.target,
                                                   predictors=self.predictors,
                                                   input_layer_size=self.input_size,
                                                   output_layer_size=self.output_size,
                                                   train_data_path=self.train_data_path,
                                                   test_data_path=self.test_data_path,
                                                   validation_data_path=self.validation_data_path,
                                                   model_param=self.model_param,
                                                   **self.kwargs
                                                   ),
                                           'multi_layer_perceptron'
                                           )()
                                   )
        self.ensemble_model = EnsembleNetwork(c0=self._get_c0(),
                                              learning_rate=self.model_param.get('boosting_rate')
                                              )

    def get_vanilla_model(self):
        """
        Get 'vanilla' typed neural network (one hidden layer only)
        """
        if len(self.input_param.keys()) == 0:
            self.model_param = self.get_vanilla_model_params()
        else:
            self.model_param = copy.deepcopy(self.input_param)
        self._config_params(loss=True,
                            activation=True,
                            hidden_layers=False
                            )
        if len(self.predictors) > 0:
            if self.target != '':
                self._import_data_torch()
            else:
                raise GrowNetException('No target feature found')
        else:
            raise GrowNetException('No predictors found')
        for i in range(0, self.model_param.get('n_mlp_models'), 1):
            if i > 0:
                self.input_size = len(self.predictors) + self.model_param.get(f'hidden_layer_{self.hidden_layer_size}_neurons')
            self.mlp_models.append(getattr(GrowNet(target=self.target,
                                                   predictors=self.predictors,
                                                   input_layer_size=self.input_size,
                                                   output_layer_size=self.output_size,
                                                   train_data_path=self.train_data_path,
                                                   test_data_path=self.test_data_path,
                                                   validation_data_path=self.validation_data_path,
                                                   model_param=self.model_param,
                                                   **self.kwargs
                                                   ),
                                           'multi_layer_perceptron'
                                           )()
                                   )
        self.ensemble_model = EnsembleNetwork(c0=self._get_c0(),
                                              learning_rate=self.model_param.get('boosting_rate')
                                              )
        self.model_param_mutation = 'new_model'

    @staticmethod
    def get_vanilla_model_params() -> dict:
        """
        Get hyperparameter of vanilla model
        """
        return dict(n_mlp_models=40,
                    learning_rate=0.005,
                    batch_size=2048,
                    batch_normalization=True,
                    accumulation=False,
                    accumulation_steps=16,
                    epoch=1,
                    alpha_dropout=False,
                    use_sparse=False,
                    activation='leaky_relu',
                    optimizer='adam',
                    loss='mse',
                    loss_corrective='mse',
                    boosting_rate=1.0,
                    weight_decay=0.001,
                    corrective_epoch=1,
                    learning_rate_corrective_steps=0.38,
                    boosting_rate_scaler=3,
                    num_hidden_layers=2,
                    hidden_layer_1_neurons=16,
                    hidden_layer_1_dropout=0.1,
                    hidden_layer_1_activation='leaky_relu',
                    hidden_layer_1_alpha_dropout=0.0,
                    hidden_layer_2_neurons=16,
                    hidden_layer_2_dropout=0.1,
                    hidden_layer_2_activation='leaky_relu',
                    hidden_layer_2_alpha_dropout=0.0,
                    )

    def eval(self, validation: bool = True):
        """
        Evaluate pre-trained machine learning model

        :param validation: bool
            Whether to run validation or testing iteration
        """
        self._eval(data_set_type='val' if validation else 'test')

    #def load(self, file_path: str):
    #    """
    #    Load pre-trained grow network
    #    """
    #    if self.cloud is not None:
    #        if self.cloud == 'aws':
    #            _bucket_name: str = file_path.split("//")[1].split("/")[0]
    #            _aws_s3_file_name: str = '/'.join(file_path.split('//')[1].split('/')[1:])
    #            _aws_s3_file_path: str = '/'.join(file_path.split('//')[1].split('/')[1:-1])
    #            _s3 = boto3.resource('s3')
    #            _s3.Bucket(_bucket_name).download_file(_aws_s3_file_path, _aws_s3_file_name)
    #        elif self.cloud == 'google':
    #            _bucket_name: str = file_path.split("//")[1].split("/")[0]
    #            _google_cloud_file_name: str = '/'.join(file_path.split('//')[1].split('/')[1:])
    #            _google_cloud_file_path: str = '/'.join(file_path.split('//')[1].split('/')[1:-1])
    #            _client = storage.Client()
    #            _bucket = _client.get_bucket(bucket_or_name=_bucket_name)
    #            _blob = _bucket.blob(blob_name=_google_cloud_file_name)
    #            _blob.download_to_filename(filename=_google_cloud_file_name.split('/')[-1])
    #    d = torch.load(f=file_path)
    #    net = DynamicNet(d['c0'], d['lr'])
    #    net.boost_rate = d['boost_rate']
    #    for stage, m in enumerate(d['models']):
    #        submod = builder(stage)
    #        submod.load_state_dict(m)
    #        net.add(submod)

    #def predict(self):
    #    """
    #    Get prediction from pre-trained grow network using PyTorch
    #    """
    #    if self.test_iter is None:
    #        self.eval(validation=True)
    #    else:
    #        self.eval(validation=False)

    def save(self, file_path: str):
        """
        Save PyTorch model to disk

        :param file_path: str
            Complete file path of the PyTorch model to save
        """
        _model_states: List[dict] = [m.state_dict() for m in self.ensemble_model.models]
        _obj: dict = dict(model_states=_model_states,
                          c0=self.ensemble_model.c0,
                          learning_rate=self.ensemble_model.learning_rate,
                          boosting_rate=self.ensemble_model.boost_rate
                          )
        torch.save(obj=_obj, f=file_path)
        if self.cloud is not None:
            if self.cloud == 'aws':
                _bucket_name: str = file_path.split("//")[1].split("/")[0]
                _aws_s3_file_name: str = '/'.join(file_path.split('//')[1].split('/')[1:])
                _aws_s3_file_path: str = '/'.join(file_path.split('//')[1].split('/')[1:-1])
                _s3 = boto3.resource('s3')
                _s3.Bucket(_bucket_name).upload_file(_aws_s3_file_name, _aws_s3_file_path)
            elif self.cloud == 'google':
                _bucket_name: str = file_path.split("//")[1].split("/")[0]
                _google_cloud_file_name: str = '/'.join(file_path.split('//')[1].split('/')[1:])
                _google_cloud_file_path: str = '/'.join(file_path.split('//')[1].split('/')[1:-1])
                _client = storage.Client()
                _bucket = _client.get_bucket(bucket_or_name=_bucket_name)
                _blob = _bucket.blob(blob_name=_google_cloud_file_name)
                _blob.upload_from_filename(filename=_google_cloud_file_name)

    def train(self):
        """
        Train neural network using deep learning framework 'PyTorch'
        """
        _t0: datetime = datetime.now()
        self.loss_models: torch = torch.zeros((self.model_param.get('n_mlp_models'), 3))
        self._config_optimizer(params=self.ensemble_model.parameters(), ensemble_network=True)
        for stage in range(0, self.model_param.get('n_mlp_models'), 1):
            print('\nStage: {}'.format(stage))
            self._config_optimizer(params=self.mlp_models[stage].parameters(), ensemble_network=False)
            self.ensemble_model.to_train()
            if self.target_type.find('ltr') >= 0:
                self._batch_learning_ltr(stage=stage)
            else:
                self._batch_learning(stage=stage)
            if stage != 0:
                if stage % int(self.model_param.get('learning_rate_corrective_steps') * self.model_param.get('n_mlp_models')) == 0:
                    self.model_param.update({'boosting_rate': (self.model_param.get('boosting_rate') / 2) / self.model_param.get('boosting_rate_scaler')})
                    self.model_param.update({'weight_decay': self.model_param.get('weight_decay') / 2})
                self._config_optimizer(params=self.ensemble_model.parameters(), ensemble_network=True)
                self._corrective_step()
        self._eval(data_set_type='train')
        self.train_time = (datetime.now() - _t0).seconds

    def update_model_param(self, param: dict):
        """
        Update model parameter config

        :param param: dict
        """
        if len(param.keys()) > 0:
            self.model_param.update(param)
