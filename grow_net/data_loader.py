"""

Load data set into PyTorch tensor

"""

import boto3
import numpy as np
import pandas as pd
import torch

from collections import Iterable
from google.cloud import storage
from torch.utils.data import DataLoader, Dataset
from typing import List

CLOUD_PROVIDER: List[str] = ['aws', 'gcp']


class LoadTabularDataException(Exception):
    """
    Class for handling exceptions for class LoadTabularData
    """
    pass


class LoadTabularData:
    """
    Class for loading tabular (structured) data set
    """
    def __init__(self,
                 target_name: str,
                 file_path: str,
                 sep: str = ';',
                 cloud: str = None,
                 bucket_name: str = None,
                 region: str = None
                 ):
        """
        :param target_name: str
            Name of the target feature

        :param file_path: str
            Complete file path

        :param sep: str
            Separator value

        :param cloud: str
            Name of the cloud provider
                -> aws: Amazon Web Service
                -> gcp: Google Cloud Platform

        :param bucket_name: str
            Name of the bucket of the cloud provider

        :param region: str
            Name of the region (AWS S3-Bucket)
        """
        self.target_name: str = target_name
        self.target_values: np.array = None
        self.n_features: int = 0
        self.data_loader: DataLoader = None
        self.full_path: str = file_path.replace('\\', '/')
        self.file_name: str = self.full_path.split('/')[-1]
        self.file_path: str = self.full_path.replace(self.file_name, '')
        _file_type = self.file_name.split('.')
        if len(_file_type) > 1:
            self.file_type = _file_type[len(_file_type) - 1]
        else:
            self.file_type = None
        self.sep: str = sep
        if cloud is not None:
            if cloud not in CLOUD_PROVIDER:
                raise LoadTabularDataException(f'Cloud provider ({cloud}) not supported. Please use {CLOUD_PROVIDER}')
        self.cloud: str = cloud
        self.bucket_name: str = bucket_name
        self.region: str = region
        self.aws_s3_file_name: str = None
        self.aws_s3_file_path: str = None
        self.google_cloud_file_name: str = None
        self.google_cloud_file_path: str = None
        if self.cloud == 'aws':
            self.aws_s3_file_name = '/'.join(self.full_path.split('//')[1].split('/')[1:])
            self.aws_s3_file_path = '/'.join(self.full_path.split('//')[1].split('/')[1:-1])
        if self.cloud == 'google':
            self.google_cloud_file_name = '/'.join(self.full_path.split('//')[1].split('/')[1:])
            self.google_cloud_file_path = '/'.join(self.full_path.split('//')[1].split('/')[1:-1])

    def _aws_s3(self) -> bytes:
        """
        Import file from AWS S3 bucket

        return: bytes
            File object as bytes
        """
        #_s3_resource = boto3.resource('s3')
        #return _s3_resource.Bucket(self.bucket_name).Object(self.aws_s3_file_name).get()['Body'].read()
        _s3 = boto3.client('s3')
        return _s3.get_object(Bucket=self.bucket_name, Key=self.aws_s3_file_name)['Body']

    def _google_cloud_storage(self):
        """
        Download files from Google Cloud Storage.
        """
        _client = storage.Client()
        _bucket = _client.get_bucket(bucket_or_name=self.bucket_name)
        _blob = _bucket.blob(blob_name=self.google_cloud_file_name)
        _blob.download_to_filename(filename=self.google_cloud_file_name.split('/')[-1])

    def generate_query_batch(self,
                             batch_size: int,
                             id_feature: str,
                             predictors: List[str]
                             ) -> Iterable:
        """
        Generate query batches for ranking model

        :param batch_size: int
            Size of data batch

        :param id_feature: str
            Name of the id feature

        :param predictors: List[str]
            Names of the predictors
        """
        _idx: int = 0
        _df: pd.DataFrame = self.text_as_df()
        while _idx * batch_size < _df.shape[0]:
            _df_sample: pd.DataFrame = _df.iloc[_idx * batch_size: (_idx + 1) * batch_size, :]
            yield _df_sample[id_feature].values, _df_sample[predictors].values, _df_sample[self.target_name].values
            _idx += 1

    def load(self,
             batch_size: int,
             shuffle: bool = True
             ):
        """
        Load data set

        :param batch_size: int
            Size of the batch

        :param shuffle: bool
            Whether to shuffle the data set or not
        """
        _data_set: TabularDataset = TabularDataset(data_set=self.text_as_df(), target=self.target_name)
        self.n_features = _data_set.n_features
        self.target_values = _data_set.target_values
        self.data_loader = DataLoader(dataset=_data_set, batch_size=batch_size, shuffle=shuffle)

    def text_as_df(self) -> pd.DataFrame:
        """
        Import text file (csv, tsv, txt) as Pandas DataFrame

        :return: Pandas DataFrame
            Content of the text file
        """
        if self.cloud == 'aws':
            return pd.read_csv(filepath_or_buffer=self._aws_s3(),
                               sep=self.sep
                               )
        elif self.cloud == 'google':
            self._google_cloud_storage()
            return pd.read_csv(filepath_or_buffer=self.google_cloud_file_path,
                               sep=self.sep
                               )
        else:
            return pd.read_csv(filepath_or_buffer=self.full_path,
                               sep=self.sep
                               )


class TabularDataset(Dataset):
    """
    Class for handling data set import using PyTorch
    """
    def __init__(self, data_set: pd.DataFrame, target: str = None):
        """
        :param data_set: pd.DataFrame
            Dataset

        :param target: str
            Name of the target feature
        """
        _data_set: pd.DataFrame = data_set
        self.n_features: int = _data_set.shape[1]
        if target is None:
            self.target_values: np.ndarray = None
        else:
            self.target_values: np.ndarray = _data_set[target].values
        del _data_set[target]
        self.data_set: np.ndarray = _data_set.values

    def __len__(self):
        """
        Get number of cases of the dataset
        """
        return len(self.data_set)

    def __getitem__(self, idx):
        """
        Get specific data from the dataset

        :param idx: int
            Index value

        return torch.FloatTensor
            Specific data from the dataset as Torch tensor
        """
        return torch.FloatTensor(self.data_set[idx]), torch.FloatTensor(np.array(self.target_values[idx]))
