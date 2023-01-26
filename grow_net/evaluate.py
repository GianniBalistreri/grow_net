"""

Evaluate neural networks

"""

import copy
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, auc, classification_report, cohen_kappa_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.metrics import mean_absolute_error, mean_gamma_deviance, mean_poisson_deviance, mean_squared_error, mean_squared_log_error, mean_tweedie_deviance, r2_score
from typing import Dict, List

ML_METRIC: Dict[str, List[str]] = dict(reg=['mae', 'mgd', 'mpd', 'mse', 'msle', 'mtd', 'r2', 'rmse', 'rmse_norm'],
                                       clf_binary=['accuracy', 'classification_report', 'confusion', 'f1', 'mcc', 'precision', 'recall', 'roc_auc'],
                                       clf_multi=['accuracy', 'classification_report', 'cohen_kappa', 'confusion', 'f1', 'mcc', 'precision', 'recall']
                                       )


class EvalClf:
    """
    Class for evaluating supervised machine learning models for classification problems
    """
    def __init__(self,
                 obs: np.array,
                 pred: np.array,
                 average: str = 'macro',
                 probability: bool = False,
                 extract_prob: int = 0,
                 labels: List[str] = None
                 ):
        """
        :param obs: np.array
            Observation

        :param pred: np.array
            Prediction

        :param probability: bool
            Prediction is probability value or not

        :param average: str
            Name of the average method to use

        :param extract_prob: int
            Number of class to use probability to classify category

        :param labels: List[str]
            Class labels
        """
        self.obs: np.array = copy.deepcopy(obs)
        _extract_prob: int = extract_prob if extract_prob >= 0 else 0
        if probability:
            self.pred: np.array = np.array([np.argmax(prob) for prob in copy.deepcopy(pred)])
        else:
            self.pred: np.array = copy.deepcopy(pred)
        self.average: str = average
        if self.average not in [None, 'micro', 'macro', 'weighted', 'samples']:
            self.average = 'macro'
        self.labels: List[str] = labels

    def accuracy(self) -> float:
        """
        Generate accuracy score
        """
        return accuracy_score(y_true=self.obs, y_pred=self.pred, normalize=True, sample_weight=None)

    def classification_report(self) -> dict:
        """
        Generate classification report containing several metric values

        :return pd.DataFrame:
            Classification report
        """
        return classification_report(y_true=self.obs,
                                     y_pred=self.pred,
                                     target_names=self.labels,
                                     sample_weight=None,
                                     digits=2,
                                     output_dict=True,
                                     zero_division='warn'
                                     )

    def cohen_kappa(self) -> float:
        """
        Cohen Kappa score classification metric for multi-class problems

        :return: float
            Cohen's Cappa Score
        """
        return cohen_kappa_score(y1=self.obs, y2=self.pred, labels=None, weights=None, sample_weight=None)

    def confusion(self, normalize: str = None) -> pd.DataFrame:
        """
        Confusion matrix for classification problems

        :param normalize: str
            Normalizing method:
                -> true: Confusion matrix normalized by observations
                -> pred: Confusion matrix normalized by predictions
                -> all: Confusion matrix normalized by both observations and predictions
                -> None: No normalization

        :return: pd.DataFrame
            Confusion Matrix
        """
        return confusion_matrix(y_true=self.obs, y_pred=self.pred, labels=None, sample_weight=None, normalize=normalize)

    def f1(self) -> float:
        """
        F1 metric of confusion matrix for classification problems

        :return: float
            F1-Score
        """
        return f1_score(y_true=self.obs,
                        y_pred=self.pred,
                        labels=None,
                        pos_label=1,
                        average=self.average,
                        sample_weight=None,
                        zero_division='warn'
                        )

    def mcc(self) -> float:
        """
        Matthews correlation coefficient metric for classification problems
        """
        return matthews_corrcoef(y_true=self.obs,
                                 y_pred=self.pred,
                                 sample_weight=None
                                 )

    def precision(self) -> float:
        """
        Precision metric of confusion matrix for classification problems

        :return: float
            Precision Score
        """
        return precision_score(y_true=self.obs,
                               y_pred=self.pred,
                               labels=None,
                               pos_label=1,
                               average=self.average,
                               sample_weight=None,
                               zero_division='warn'
                               )

    def recall(self) -> float:
        """
        Recall metric of confusion matrix for classification problems

        :return: float
            Recall score
        """
        return recall_score(y_true=self.obs,
                            y_pred=self.pred,
                            labels=None,
                            pos_label=1,
                            average=self.average,
                            sample_weight=None,
                            zero_division='warn'
                            )

    def roc_auc(self) -> float:
        """
        Area Under Receiver Operating Characteristic Curve classification metric for binary problems

        :return: float
            Area-Under-Curve Score (AUC)
        """
        if len(list(pd.unique(self.obs))) == 1 or len(list(pd.unique(self.pred))) == 1:
            return 0.0
        else:
            return roc_auc_score(y_true=self.obs,
                                 y_score=self.pred,
                                 average=self.average,
                                 sample_weight=None,
                                 max_fpr=None,
                                 multi_class='raise',
                                 labels=None
                                 )

    def roc_auc_multi(self, meth: str = 'ovr') -> float:
        """
        Area Under Receiver Operating Characteristic Curve classification metric for binary problems

        :param: meth: Method of multi-class roc-auc score
                        -> ovr: Computes score for each class against the rest
                        -> ovo: Computes score for each class pairwise

        :return: float
            Area-Under_Curve Score for multi-class problems
        """
        _meth: str = meth if meth in ['ovr', 'ovo'] else 'ovr'
        return roc_auc_score(y_true=self.obs,
                             y_score=self.pred,
                             average=self.average,
                             sample_weight=None,
                             max_fpr=None,
                             multi_class=_meth,
                             labels=None
                             )

    def roc_curve(self) -> dict:
        """
        Calculate true positive rates & false positive rates for generating roc curve

        :return: dict
            Calculated true positive, false positive rates and roc-auc score
        """
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(0, len(pd.unique(self.obs).tolist()), 1):
            fpr[i], tpr[i], _ = roc_curve(self.obs, self.pred)
            roc_auc[i] = auc(fpr[i], tpr[i])
        return dict(true_positive_rate=tpr, false_positive_rate=fpr, roc_auc=roc_auc)


class EvalRank:
    """
    Evaluate ranking model
    """
    def __int__(self):
        pass


class EvalReg:
    """
    Class for evaluating supervised machine learning models for regression problems
    """
    def __init__(self, obs: np.array, pred: np.array, multi_output: str = 'uniform_average'):
        """
        :param obs: np.array
            Observation

        :param pred: np.array
            Prediction

        :param multi_output: str
            Method to handle multi output
                -> uniform_average: Errors of all outputs are averaged with uniform weight
                -> raw_values: Returns a full set of errors in case of multi output input
        """
        self.obs: np.array = obs
        self.std_obs: float = self.obs.std()
        self.pred: np.array = pred
        self.multi_output: str = multi_output if multi_output in ['raw_values', 'uniform_average'] else 'uniform_average'

    def mae(self) -> float:
        """
        Mean absolute error metric for regression problems

        :return: float
            Mean-Absolute-Error Score
        """
        return mean_absolute_error(y_true=self.obs, y_pred=self.pred, sample_weight=None, multioutput=self.multi_output)

    def mgd(self) -> float:
        """
        Mean gamma deviance error metric for regression problems

        :return: float
            Mean-Gama-Deviance-Error Score
        """
        return mean_gamma_deviance(y_true=self.obs, y_pred=self.pred, sample_weight=None)

    def mpd(self) -> float:
        """
        Mean poisson deviance error metric for regression problems

        :return: float
            Mean-Poisson-Deviance-Error Score
        """
        return mean_poisson_deviance(y_true=self.obs, y_pred=self.pred, sample_weight=None)

    def mse(self) -> float:
        """
        Mean squared error metric for regression problems

        :return: float
            Mean-Squared-Error Score
        """
        return mean_squared_error(y_true=self.obs,
                                  y_pred=self.pred,
                                  sample_weight=None,
                                  multioutput=self.multi_output,
                                  squared=True
                                  )

    def msle(self) -> float:
        """
        Mean squared log error metric for regression problems

        :return: float
            Mean-Squared-Log-Error Score
        """
        return mean_squared_log_error(y_true=self.obs,
                                      y_pred=self.pred,
                                      sample_weight=None,
                                      multioutput=self.multi_output
                                      )

    def mtd(self) -> float:
        """
        Mean tweedie deviance error metric for regression problems

        :return: float
            Mean-Tweedie-Deviance-Error Score
        """
        return mean_tweedie_deviance(y_true=self.obs, y_pred=self.pred, sample_weight=None)

    def r2(self) -> float:
        """
        R2 coefficient of determination

        :return float
            R2 Score
        """
        return r2_score(y_true=self.obs,
                        y_pred=self.pred,
                        sample_weight=None,
                        multioutput=self.multi_output
                        )

    def rmse(self) -> float:
        """
        Root mean squared error metric for regression problems

        :return: float
            Root-Mean-Squared-Error Score
        """
        return mean_squared_error(y_true=self.obs,
                                  y_pred=self.pred,
                                  sample_weight=None,
                                  multioutput=self.multi_output,
                                  squared=False
                                  )

    def rmse_norm(self) -> float:
        """
        Normalized root mean squared error metric by standard deviation for regression problems

        :return: float
            Normalized Root-Mean-Squared-Error Score (by Standard Deviation)
        """
        return self.rmse() / self.std_obs
