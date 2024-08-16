import sqlite3
from typing import Dict, Any, Optional
import random
from datasets import Dataset
import datasets
import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from finlm.tokenizer import FinLMTokenizer
from transformers import AutoTokenizer
import numpy as np
import logging

datasets.disable_progress_bars()


class FinLMDataset:
    """
        A class for defining a dataset which retrieves chunked text sequences, tokenizes them and returns batches of
        input_ids and attention_mask tensors which are needed for pretraining financial models.

        Attributes
        ----------
        tokenizer : FinLMTokenizer
            The tokenizer used to tokenize text sequences.
        mask_token_id : int
            The ID of the mask token in the tokenizer.
        special_token_ids : set
            A set of IDs corresponding to all special tokens, excluding the mask token.
        max_sequence_length : int
            The maximum number of tokens allowed per sequence.
        db_name : str
            The path to the SQLite database where text sequences are stored.
        database_retrieval : dict[str, dict[str, int]]
            A dictionary specifying the retrieval information for each table in the database.
        table_names : list[str]
            A list of table names from which sequences are to be retrieved.
        batch_size : int
            The number of sequences returned per batch.
    """

    def __init__(
            self,
            tokenizer_path: str,
            max_sequence_length: int,
            db_name: str,
            database_retrieval: dict[str, dict[str, int]],
            batch_size: int
    ) -> None:

        """
        Initializes the FinLMDataset with necessary parameters.

        Parameters
        ----------
        tokenizer_path : str
            Path to the pretrained tokenizer.
        max_sequence_length : int
            The maximum number of tokens allowed per sequence.
        db_name : str
            Path to the SQLite database containing chunked text sequences.
        database_retrieval : dict[str, dict[str, int]]
            A dictionary specifying retrieval instructions for each table in the database.
        batch_size : int
            The number of sequences to return in each batch.
        """

        self.logger = logging.getLogger(self.__class__.__name__)
        self.tokenizer = FinLMTokenizer(tokenizer_path)
        self.mask_token_id = self.tokenizer.mask_token_id
        self.special_token_ids = set(self.tokenizer.all_special_ids).difference(set([self.tokenizer.mask_token_id]))
        self.max_sequence_length = max_sequence_length
        self.db_name = db_name
        self.database_retrieval = database_retrieval
        self.table_names = list(self.database_retrieval.keys())
        self.batch_size = batch_size
        if not hasattr(self, "n_total_sequences"):
            self._get_n_dataset_sequences()

    def _get_n_dataset_sequences(self):

        """
        Determines the total number of sequences available in each table of the database.

        This method counts the total number of sequences in each table of the SQLite database specified 
        in `db_name`. It logs the number of sequences found in each table and stores these counts in 
        `self.n_total_sequences`.

        Attributes Updated
        ------------------
        n_total_sequences : dict
            A dictionary where the keys are table names and the values are the number of sequences in each table.
        """

        self.logger.info("Counting the number of sequences in the database...")
        conn = sqlite3.connect(self.db_name)
        self.n_total_sequences = {}
        for t_name in self.table_names:
            res = conn.execute(f"SELECT COUNT(*) FROM {t_name}")
            n_seqs_tmp = res.fetchall()
            self.n_total_sequences[t_name] = n_seqs_tmp[0][0]
        conn.close()

        self.logger.info("-" * 100)
        self.logger.info(
            f"The database includes {len(self.table_names)} sheets with the following names and number of sequences:")
        for t_name, n_seq in self.n_total_sequences.items():
            self.logger.info(f"{t_name}: {n_seq}")
        self.logger.info("-" * 100)

    def _retrieve_sequences_from_database(self):

        """
        Retrieves sequences from the database based on the `database_retrieval` dictionary.

        This method queries the database to collect sequences as specified in `database_retrieval`. 
        The sequences are then shuffled and stored in `self.sequences`. 

        Together with the self.set_dataset_offsets method this method is used to update sequences in the dataset during training.
        For instance, if the LIMIT = 1000, in the first epoch data is collected with sequences 0,...,999 and OFFSET = 0. In the
        next epoch, the OFFSET can be updated to OFFSET = 1000 and sequences 1000,...,1999 are collected. By this means, new 
        sequences can be used without the need to load large datasets with space larger than memory space. 

        Attributes Updated
        ------------------
        sequences : list
            A list of sequences retrieved from the database and shuffled.
        """

        self.logger.info("Starting to retrieve sequences from database.")
        sequences = []
        conn = sqlite3.connect(self.db_name)
        for key in self.database_retrieval.keys():
            sql_query = f"SELECT * FROM {key} LIMIT {self.database_retrieval[key]['limit']} OFFSET {self.database_retrieval[key]['offset']};"
            res = conn.execute(sql_query)
            seqs = res.fetchall()
            seqs = [seq[0] for seq in seqs]
            sequences.extend(seqs)
        conn.close()
        random.shuffle(sequences)
        self.sequences = sequences
        self.logger.info("Sequences have been retrieved from database.")

    def _tokenization(self, text_sequence: list[str]):

        """
        Tokenizes a list of text sequences using the pretrained tokenizer.

        This helper method is used by the `map` method from the Hugging Face `Dataset` class. It tokenizes 
        the input text sequences to ensure they adhere to the model's input format.

        Parameters
        ----------
        text_sequence : list[str]
            A list of text sequences where each sequence is expected to be a dictionary with a "text" key.

        Returns
        -------
        dict
            A dictionary containing the tokenized `input_ids`, `attention_mask`, and other relevant information.
        """

        return self.tokenizer(text_sequence["text"], padding='max_length', truncation=True,
                              max_length=self.max_sequence_length)

    def _create_hf_dataset(self):

        """
        Creates a Hugging Face `Dataset` from a list of sequences and tokenizes them.

        This method initializes a Hugging Face `Dataset` object from the sequences retrieved from the database.
        It then tokenizes all sequences using the `_tokenization` method and sets the format of the dataset 
        to be compatible with PyTorch tensors.

        Attributes Updated
        ------------------
        hf_dataset : datasets.Dataset
            The Hugging Face `Dataset` containing the tokenized sequences.
        data_loader : torch.utils.data.DataLoader
            A PyTorch DataLoader created from the tokenized dataset for batching during training.
        """

        self.hf_dataset = Dataset.from_list([{"text": text} for text in self.sequences])
        self.logger.info("Starting to tokenize the sequences.")
        self.hf_dataset = self.hf_dataset.map(self._tokenization)
        self.logger.info("Tokenization is finished.")
        self.hf_dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask"])
        self.data_loader = DataLoader(self.hf_dataset, batch_size=self.batch_size, shuffle=False)

    @classmethod
    def from_dict(cls, data_config: Dict[str, Any]) -> 'FinLMDataset':

        """
        Instantiates the `FinLMDataset` class from a configuration dictionary.

        This class method allows the creation of a `FinLMDataset` object using a dictionary that contains
        all the necessary parameters for initialization.

        Parameters
        ----------
        data_config : Dict[str, Any]
            A dictionary where keys correspond to the parameters required by the `__init__` method.

        Returns
        -------
        FinLMDataset
            An instance of the `FinLMDataset` class.
        """

        return cls(**data_config)

    def prepare_data_loader(self):

        """
        Prepares the data loader by retrieving sequences and creating a Hugging Face dataset.

        This method first retrieves sequences from the database using `_retrieve_sequences_from_database` 
        and then tokenizes them by calling `_create_hf_dataset`. The result is a PyTorch DataLoader 
        ready for iteration during training.

        Attributes Updated
        ------------------
        data_loader : torch.utils.data.DataLoader
            A PyTorch DataLoader created from the tokenized dataset.
        """
        self._retrieve_sequences_from_database()
        self._create_hf_dataset()

    def set_dataset_offsets(self, epoch):

        """
        Updates the offsets in the `database_retrieval` dictionary based on the current epoch.

        This method adjusts the `offset` in the `database_retrieval` dictionary for each table according 
        to the current epoch. If the offset exceeds the total number of sequences available, it resets 
        the offset to zero and logs a message.

        Parameters
        ----------
        epoch : int
            The current epoch number, used to calculate the new offset for sequence retrieval.
        """

        for key in self.database_retrieval.keys():
            self.database_retrieval[key]["offset"] = epoch * self.database_retrieval[key]["limit"]
            if (self.database_retrieval[key]["offset"] + self.database_retrieval[key]["offset"]) > \
                    self.n_total_sequences[key]:
                logging.info(
                    f"Remaining number of sequences for table {key} is too little, starting to retrieve sentences from the start.")
                self.database_retrieval[key]["offset"] = 0

    def __iter__(self):

        """
        Iterates over the dataset using the PyTorch DataLoader.

        This method provides an iterator over the batches of data prepared by the DataLoader. 
        If the DataLoader has not been created yet, it calls `prepare_data_loader` to initialize it.

        Yields
        ------
        dict
            A batch of data ready for model training or evaluation.
        """

        if not hasattr(self, "data_loader"):
            self.prepare_data_loader()
        for batch in self.data_loader:
            yield batch


class FinetuningDataset:
    """
    A dataset class designed for fine-tuning language models.

    This class handles the preparation of datasets for fine-tuning, including tokenization,
    shuffling, and splitting into training and test datasets.

    Attributes
    ----------
    tokenizer_path : str
        Path to the pretrained tokenizer to be used for tokenization.
    tokenizer : FinLMTokenizer
        The tokenizer instance used for tokenizing the text sequences.
    max_sequence_length : int
        The maximum number of tokens per sequence.
    dataset : Dataset
        The Hugging Face `Dataset` object containing the raw data to be processed.
    text_column : str
        The name of the column in the dataset containing the text sequences to be tokenized.
    dataset_columns : list[str]
        A list of column names to include in the final tokenized dataset.
    shuffle_data : bool
        A flag indicating whether the dataset should be shuffled.
    shuffle_data_random_seed : Optional[int]
        The random seed to use for shuffling, if applicable.
    training_data_fraction : float
        The fraction of the dataset to use for training; the rest is used for testing.
    train_size : int
        The number of samples in the training dataset.
    training_data : Dataset
        The tokenized and processed training dataset.
    test_data : Dataset
        The tokenized and processed test dataset.
    logger : logging.Logger
        Logger instance for logging messages.

    Methods
    -------
    _tokenization(text_sequence: list[str]) -> dict
        Tokenizes a list of text sequences using the pretrained tokenizer.
    _map_dataset() -> None
        Tokenizes the entire dataset and sets the format to be compatible with PyTorch.
    _prepare_training_and_test_data() -> None
        Tokenizes, shuffles, and splits the dataset into training and test datasets.
    """

    def __init__(self,
                 tokenizer_path: str,
                 max_sequence_length: int,
                 dataset: Dataset,
                 text_column: str,
                 dataset_columns: list[str]
                 ):

        """
        Initializes the `FinetuningDataset` with the necessary parameters.

        Parameters
        ----------
        tokenizer_path : str
            Path to the pretrained tokenizer to be used for tokenization.
        max_sequence_length : int
            The maximum number of tokens per sequence.
        dataset : Dataset
            The Hugging Face `Dataset` object containing the raw data to be processed.
        text_column : str
            The name of the column in the dataset containing the text sequences to be tokenized.
        dataset_columns : list[str]
            A list of column names to include in the final tokenized dataset.
        shuffle_data : bool, optional
            Whether to shuffle the dataset before splitting (default is True).
        shuffle_data_random_seed : Optional[int], optional
            Random seed for shuffling the dataset, if applicable (default is None).
        training_data_fraction : float, optional
            Fraction of the dataset to use for training (default is 0.80).

        Attributes Initialized
        ----------------------
        tokenizer : FinLMTokenizer
            The tokenizer instance used for tokenizing the text sequences.
        train_size : int
            The number of samples in the training dataset.
        training_data : Dataset
            The tokenized and processed training dataset.
        test_data : Dataset
            The tokenized and processed test dataset.
        """

        self.tokenizer_path = tokenizer_path
        try:
            self.tokenizer = FinLMTokenizer(self.tokenizer_path)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.max_sequence_length = max_sequence_length
        self.dataset = dataset
        self.text_column = text_column
        self.dataset_columns = dataset_columns
        self.logger = logging.getLogger(self.__class__.__name__)
        self._map_dataset()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns a single tokenized item at the specified index.

        Parameters
        ----------
        idx : int
            The index of the item to retrieve.

        Returns
        -------
        dict
            A dictionary containing the tokenized input data and labels for the item.
        """
        return {key: self.dataset[idx][key] for key in self.dataset_columns}

    def _tokenization(self, text_sequence: list[str]):

        """
        Tokenizes a list of text sequences using the pretrained tokenizer.

        This helper method is used by the `map` method from the Hugging Face `Dataset` class. It tokenizes 
        the input text sequences using the specified tokenizer.

        Parameters
        ----------
        text_sequence : list[str]
            A list of text sequences where each sequence is expected to be a dictionary 
            with a key corresponding to `self.text_column`.

        Returns
        -------
        dict
            A dictionary containing the tokenized `input_ids`, `attention_mask`, and other relevant information.
        """

        return self.tokenizer(text_sequence[self.text_column], padding='max_length', truncation=True,
                              max_length=self.max_sequence_length)

    def _map_dataset(self):

        """
        Tokenizes the entire dataset and sets the format for use with PyTorch.

        This method applies the `_tokenization` method to each sequence in the dataset, 
        converting the dataset into a format compatible with PyTorch. It also logs the 
        progress of the tokenization process.
        """

        self.logger.info("Starting to tokenize the sequences.")
        self.dataset = self.dataset.map(self._tokenization)
        self.logger.info("Tokenization is finished.")
        self.dataset.set_format(type="torch", columns=self.dataset_columns)

    def num_labels(self):
        return np.unique(self.dataset["label"], return_counts=True)

    def select(self, indices):
        new_finetuning_dataset = FinetuningDataset(
            self.tokenizer_path,
            self.max_sequence_length,
            self.dataset.select(indices),
            self.text_column,
            self.dataset_columns
        )
        return new_finetuning_dataset


class FinetuningDocumentDataset(TorchDataset):
    def __init__(self, documents, labels, tokenizer_path, sequence_length):
        self.documents = documents
        self.labels = labels
        self.tokenizer_path = tokenizer_path
        try:
            self.tokenizer = FinLMTokenizer(self.tokenizer_path)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        document = self.documents[idx]
        label = self.labels[idx]

        encoded_sequences = []
        attention_masks = []

        for sequence in document:
            encoded = self.tokenizer(
                sequence,
                truncation=True,
                padding='max_length',
                max_length=self.sequence_length,
                return_tensors='pt'
            )
            encoded_sequences.append(encoded['input_ids'].squeeze(0))
            attention_masks.append(encoded['attention_mask'].squeeze(0))

        return {
            'input_ids': torch.stack(encoded_sequences),
            'attention_mask': torch.stack(attention_masks),
            'label': torch.tensor(label, dtype=torch.long)
        }

    def select(self, indices):
        """
        Returns a new DocumentDataset with only the selected indices.
        """
        selected_documents = [self.documents[i] for i in indices]
        selected_labels = [self.labels[i] for i in indices]
        return FinetuningDocumentDataset(selected_documents, selected_labels, self.tokenizer_path, self.sequence_length)

    def num_labels(self):
        labels = np.array([batch["label"].item() for batch in self])
        return np.unique(labels, return_counts=True)


def collate_fn_fixed_sequences(batch, max_sequences=4):
    max_sequence_length = max([item['input_ids'].shape[1] for item in batch])

    input_ids_batch = []
    attention_mask_batch = []
    labels_batch = []
    sequences_mask_batch = []

    for item in batch:
        input_ids = item['input_ids']
        attention_mask = item['attention_mask']
        label = item['label']

        # Adjust the number of sequences to match num_sequences
        n_sequences = input_ids.shape[0]
        n_sequence_mask = [0] * max_sequences
        n_sequence_mask[:n_sequences] = [1] * n_sequences
        if n_sequences > max_sequences:
            input_ids = input_ids[:max_sequences, :]
            attention_mask = attention_mask[:max_sequences, :]
            n_sequence_mask = n_sequence_mask[:max_sequences]
        elif n_sequences < max_sequences:
            padding = torch.zeros((max_sequences - input_ids.shape[0], max_sequence_length), dtype=torch.long)
            input_ids = torch.cat([input_ids, padding], dim=0)
            attention_mask = torch.cat([attention_mask, padding], dim=0)

        input_ids_batch.append(input_ids)
        attention_mask_batch.append(attention_mask)
        labels_batch.append(label)
        sequences_mask_batch.append(n_sequence_mask)

    return {
        'input_ids': torch.stack(input_ids_batch),
        'attention_mask': torch.stack(attention_mask_batch),
        'label': torch.tensor(labels_batch, dtype=torch.long),
        'sequence_mask': torch.tensor(sequences_mask_batch, dtype=torch.long)
    }
