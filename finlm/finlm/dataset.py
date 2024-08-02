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
            ):
        
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
        conn = sqlite3.connect(self.db_name)
        res = conn.execute("SELECT name from sqlite_master")
        table_names = res.fetchall()
        conn.close()
        table_names = [t_name[0] for t_name in table_names]
        self.table_names = table_names

    def _get_n_sequences(self):
        self.logger.info("Counting the number of sequences in the database...")
        conn = sqlite3.connect(self.db_name)
        self.n_seqs = []
        for t_name in self.table_names:
            res = conn.execute(f"SELECT COUNT(*) FROM {t_name}")
            n_seqs_tmp = res.fetchall()
            self.n_seqs.append(n_seqs_tmp[0][0])
        conn.close()
    
    def _create_database_retrieval(self, **kwargs):
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

    def _tokenization(self, text_sequence):
        return self.tokenizer(text_sequence["text"], padding='max_length', truncation=True, max_length=self.max_sequence_length)

    def _create_hf_dataset(self):
        self.hf_dataset = Dataset.from_list([{"text": text} for text in self.sequences])
        self.logger.info("Starting to tokenize the sequences.")
        self.hf_dataset = self.hf_dataset.map(self._tokenization)
        self.logger.info("Tokenization is finished.")
        self.hf_dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask"])
        self.data_loader = DataLoader(self.hf_dataset, batch_size = self.batch_size, shuffle = False)

    def _prepare_data_loader(self):
        self._retrieve_sequences_from_database()
        self._create_hf_dataset()

    def __iter__(self):
        if not hasattr(self, "data_loader"):
            self._prepare_data_loader()
        for batch in self.data_loader:
            yield batch


