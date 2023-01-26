"""

Train and evaluate Gradient Boosting Neural Network (GrowNet)

"""


class GrowNet:
    """
    Class for building, training and evaluating gradient boosting neural networks
    """
    def __int__(self,
                file_path_train_data_set: str,
                file_path_test_data_set: str,
                file_path_eval_data_set: str = None,
                n_networks: int = 100,
                mlp_param: dict = None,
                use_same_param_for_ensemble: bool = True,
                force_ml_type: str = None,
                **kwargs: dict
                ):
        """
        :param file_path_train_data_set: str
            Complete file path for train data set

        :param file_path_test_data_set: str
            Complete file path for test data set

        :param file_path_eval_data_set: str
            Complete file path for evaluation data set

        :param n_networks: int
            Number of neural networks to generate

        :param mlp_param: dict
            Hyper-parameter configuration of multi-layer perceptron

        :param use_same_param_for_ensemble: bool
            Whether to use same neural network architecture for all neural networks in ensemble or not

        :param force_ml_type: str
            Force machine learning type
                -> rank: Ranking
                -> reg: Regression

        :param kwargs: dict
            Key-word arguments
        """
        self.file_path_train_data_set: str = file_path_train_data_set
        self.file_path_test_data_set: str = file_path_test_data_set
        self.file_path_eval_data_set: str = file_path_eval_data_set
        self.n_networks: int = n_networks
        self.mlp_param: dict = mlp_param
        self.use_same_param_for_ensemble: bool = use_same_param_for_ensemble
        self.force_ml_type: str = force_ml_type
