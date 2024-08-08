import sqlite3
from typing import Dict, Any, Optional
import random
from datasets import Dataset
import datasets
from torch.utils.data import DataLoader
from finlm.tokenizer import FinLMTokenizer
import logging
datasets.disable_progress_bars()


class FinLMDataset:
    def __init__(
            self,
            tokenizer_path: str,
            max_sequence_length: int,
            db_name: str,
            database_retrieval: dict[str, dict[str, int]],
            batch_size: int
            ) -> None:
        
        """
            A class for defining a dataset which retrieves chunked text sequences, tokenizes them and returns batches of
            input_ids and attention_mask tensors which are needed for pretraining financial models.

            tokenizer_path: path to self trained tokeniezer
            max_sequence_length: the maximum number of tokens per sequence
            db_name: the location of the database where the chunked text sequences are
            table_names: the table names from which the sequences should be retrieved
            limits: a dictionary with table names as keys and the desired number of sequences as integers
            offsets: a dictionary with table names as keys and the offsets as integers
            batch_size: the number of sequences which should be returned per batch
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
        
        """Determines the complete number of available sequences in each table of the database, NOT the number of sequences which is used."""
        
        self.logger.info("Counting the number of sequences in the database...")
        conn = sqlite3.connect(self.db_name)
        self.n_total_sequences = {}
        for t_name in self.table_names:
            res = conn.execute(f"SELECT COUNT(*) FROM {t_name}")
            n_seqs_tmp = res.fetchall()
            self.n_total_sequences[t_name] = n_seqs_tmp[0][0]
        conn.close()

        self.logger.info("-"*100)
        self.logger.info(f"The database includes {len(self.table_names)} sheets with the following names and number of sequences:")
        for t_name, n_seq in zip(self.table_names, self.n_total_sequences):
            self.logger.info(f"{t_name}: {n_seq}")
        self.logger.info("-"*100)

    def _retrieve_sequences_from_database(self):    

        """Uses the database_retrieval dictionary to collect sequences as desired."""

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

        """A helper function which is used by the map method from the dataset below."""

        return self.tokenizer(text_sequence["text"], padding='max_length', truncation=True, max_length=self.max_sequence_length)

    def _create_hf_dataset(self):

        """This function creates a dataset from datasets' Dataset class. Tokenizes all sequences and sets the format to torch."""

        self.hf_dataset = Dataset.from_list([{"text": text} for text in self.sequences])
        self.logger.info("Starting to tokenize the sequences.")
        self.hf_dataset = self.hf_dataset.map(self._tokenization)
        self.logger.info("Tokenization is finished.")
        self.hf_dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask"])
        self.data_loader = DataLoader(self.hf_dataset, batch_size = self.batch_size, shuffle = False)

    @classmethod
    def from_dict(cls, data_config: Dict[str, Any]) -> 'FinLMDataset':
        return cls(**data_config)

    def prepare_data_loader(self):
        
        """Create dictionary with sql query information and collect data accordingly."""
        self._retrieve_sequences_from_database()
        self._create_hf_dataset()

    def set_dataset_offsets(self, epoch):
        for key in self.database_retrieval.keys():
            self.database_retrieval[key]["offset"] = epoch * self.database_retrieval[key]["limit"]
            if (self.database_retrieval[key]["offset"] + self.database_retrieval[key]["offset"]) > self.n_total_sequences[key]:
                logging.info(f"Remaining number of seqeuences for table {key} is too little, starting to retrieve sentences from the start.")
                self.database_retrieval[key]["offset"] = 0

    def __iter__(self):
        
        """Use the data_loader created by the torch's DataLoader class for iteration."""

        if not hasattr(self, "data_loader"):
            self.prepare_data_loader()
        for batch in self.data_loader:
            yield batch



class FinetuningDataset:
    def __init__(self,
            tokenizer_path: str,
            max_sequence_length: int,
            dataset: Dataset,
            text_column: str,
            dataset_columns: list[str],
            shuffle_data: bool = True,
            shuffle_data_random_seed: Optional[int] = None,
            training_data_fraction: float = 0.80
        ):

        self.tokenizer_path = tokenizer_path
        self.tokenizer = FinLMTokenizer(self.tokenizer_path)
        self.max_sequence_length = max_sequence_length
        self.dataset = dataset
        self.text_column = text_column
        self.dataset_columns = dataset_columns
        self.shuffle_data = shuffle_data
        self.shuffle_data_random_seed = shuffle_data_random_seed
        self.training_data_fraction = training_data_fraction
        self.logger = logging.getLogger(self.__class__.__name__)
        self._prepare_training_and_test_data()


    def _tokenization(self, text_sequence: list[str]):

        """A helper function which is used by the map method from the dataset below."""

        return self.tokenizer(text_sequence[self.text_column], padding='max_length', truncation=True, max_length=self.max_sequence_length)

    def _map_dataset(self):

        """This function creates a dataset from datasets' Dataset class. Tokenizes all sequences and sets the format to torch."""

        self.logger.info("Starting to tokenize the sequences.")
        self.dataset = self.dataset.map(self._tokenization)
        self.logger.info("Tokenization is finished.")
        self.dataset.set_format(type="torch", columns=self.dataset_columns)

    def _prepare_training_and_test_data(self):
        self._map_dataset()
        if self.shuffle_data:
            if self.shuffle_data_random_seed:
                self.dataset = self.dataset.shuffle(self.shuffle_data_random_seed)
            else:
                self.dataset = self.dataset.shuffle()

        self.train_size = int(len(self.dataset) * self.training_data_fraction)
        self.training_data = self.dataset.select(range(self.train_size))
        self.test_data = self.dataset.select(range(self.train_size, len(self.dataset)))



