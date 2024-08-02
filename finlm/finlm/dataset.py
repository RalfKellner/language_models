import sqlite3
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
            n_10k_seq: int,
            n_8k_seq: int,
            n_ec_seq: int,
            n_news_seq: int,
            batch_size: int,
            random_sql = True,
            **kwargs
            ) -> None:
        
        """
            A class for defining a dataset which retrieves chunked text sequences, tokenizes them and returns batches of
            input_ids and attention_mask tensors which are needed for pretraining financial models.

            tokenizer_path: path to self trained tokeniezer
            max_sequence_length: the maximum number of tokens per sequence
            db_name: the location of the database where the chunked text sequences are
            n_10k_seq: the number of desired text sequences of chunked 10K form filling
            n_8k_seq: the number of desired text sequences of chunked 8K press releases
            n_ec_seq: the number of desired text sequences of chunked earning call transcripts
            n_news_seq: the number of desired text sequences of chunked TR news
            batch_size: the number of sequences which should be returned per batch
            random_sql: if set to True, sequences in each table are shuffled before retrieval
            
            Important: if one does not to retrieve random sequences but to use LIMIT and OFFSET, [n_10k_seq, n_8k_seq, n_ec_seq, n_news_seq]
                are used as LIMIT and OFFSET can be set as a list using a kwargs argument called 'offsets', e.g., offsets = [10, 10, 10, 10]
        """


        self.logger = logging.getLogger(self.__class__.__name__)
        self.tokenizer = FinLMTokenizer(tokenizer_path)
        self.max_sequence_length = max_sequence_length
        self.db_name = db_name
        self.n_sequences = [n_10k_seq, n_8k_seq, n_ec_seq, n_news_seq]
        self.batch_size = batch_size
        self._get_table_names()
        self._get_n_sequences()
        self.logger.info("-"*100)
        self.logger.info(f"The database includes {len(self.table_names)} sheets with the following names and number of sequences:")
        for t_name, n_seq in zip(self.table_names, self.n_seqs):
            self.logger.info(f"{t_name}: {n_seq}")
        self.logger.info("-"*100)
        self._create_database_retrieval(**kwargs)
        self.random_sql = random_sql

    def _get_table_names(self):

        """Determines table names in the database."""
        
        conn = sqlite3.connect(self.db_name)
        res = conn.execute("SELECT name from sqlite_master")
        table_names = res.fetchall()
        conn.close()
        table_names = [t_name[0] for t_name in table_names]
        self.table_names = table_names

    def _get_n_sequences(self):
        
        """Determines the complete number of available sequences in each table of the database, NOT the number of sequences which is used."""
        
        self.logger.info("Counting the number of sequences in the database...")
        conn = sqlite3.connect(self.db_name)
        self.n_seqs = []
        for t_name in self.table_names:
            res = conn.execute(f"SELECT COUNT(*) FROM {t_name}")
            n_seqs_tmp = res.fetchall()
            self.n_seqs.append(n_seqs_tmp[0][0])
        conn.close()
    
    def _create_database_retrieval(self, **kwargs):
        """
            This function creates a dictionary with the keys: t_name, limit and offset. This arguments are used for the sql query for each table
            when data is retrieved. Using offsets only makes senses if random_sql is set to False, but then can be used to iteratively collect
            new sequences from the database during training as the memory space is not enough to import all sequences at once. Furthermore, 
            we should not need all sequences which are available as this would by far exceed the number of sequences which are typically used
            for pretraining, e.g., Electra small 500k * 128.
        """
        if len(kwargs) == 0:
            self.database_retrieval = {}
            for t_name, limit in zip(self.table_names, self.n_sequences):
                self.database_retrieval[t_name] = {}
                self.database_retrieval[t_name]["limit"] = limit
                self.database_retrieval[t_name]["offset"] = 0
        else:
            assert "offsets" in kwargs.keys(), "Please provide a list of limits and offsets which should be retrieved from the database."
            assert len(kwargs["offsets"]) == len(self.table_names), "The number of limit and offset values must be equal to the number of tables from which data is retrieved."
            self.database_retrieval = {}
            for t_name, limit, offset in zip(self.table_names, self.n_sequences, kwargs["offsets"]):
                self.database_retrieval[t_name] = {}
                self.database_retrieval[t_name]["limit"] = limit
                self.database_retrieval[t_name]["offset"] = offset

    def _retrieve_sequences_from_database(self):    

        """Uses the database_retrieval dictionary to collect sequences as desired."""

        self.logger.info("Starting to retrieve sequences from database.")
        sequences = []
        conn = sqlite3.connect(self.db_name)
        for key in self.database_retrieval.keys():

            if self.random_sql:
                sql_query = f"SELECT * FROM {key} ORDER BY RANDOM() LIMIT {self.database_retrieval[key]['limit']};"
            else:
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

    def _prepare_data_loader(self):
        
        """Create dictionary with sql query information and collect data accordingly."""

        self._retrieve_sequences_from_database()
        self._create_hf_dataset()

    def __iter__(self):
        
        """Use the data_loader created by the torch's DataLoader class for iteration."""

        if not hasattr(self, "data_loader"):
            self._prepare_data_loader()
        for batch in self.data_loader:
            yield batch


