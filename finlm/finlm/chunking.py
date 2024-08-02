import sqlite3
import pandas as pd
from finlm.utils import split_text_into_chunks_by_words, count_numbers_and_punctuation
import logging

class Chunker:
    def __init__(self, db_in, sheet_in, limit = None, offset = None):
        self.db_in = db_in
        self.sheet_in = sheet_in
        self.limit = limit
        self.offset = offset
        self.n_documents = self.count_documents()
        self.logger = logging.getLogger(self.__class__.__name__)

    def count_documents(self):
        # determine the number of filings
        conn_in = sqlite3.connect(self.db_in)
        count_n = conn_in.execute(f"SELECT COUNT(*) FROM {self.sheet_in};")
        n_docs = count_n.fetchall()
        conn_in.close()
        n_docs = n_docs[0][0]
        return n_docs
    
    def chunk_to_database(self, db_out, sheet_out, chunk_size_in_words, ignore_first_sentences, ignore_last_sentences):
        conn_out = sqlite3.connect(db_out)
        documents_excluded = 0
        for doc_id, document in enumerate(self):
            try:
                text_chunks = split_text_into_chunks_by_words(document, chunk_size_in_words, ignore_first_sentences, ignore_last_sentences)
                text_chunks_df = pd.DataFrame({"sequence": text_chunks}, index = list(range(len(text_chunks))))
                text_chunks_df.to_sql(sheet_out, conn_out, if_exists="append", index = False)
            except Exception as e:
                documents_excluded += 1
            if doc_id % 5000 == 0:
                self.logger.info(f"{doc_id/self.n_documents:.2%} of all documents have been chunked.")
        conn_out.close()
        self.logger.info(f"A fraction of {documents_excluded/self.n_documents:.4f} has been excluded while chunking.")


class Form10KChunker(Chunker):
    def __init__(self, db_in, sheet_in, limit = None, offset = None):
        super().__init__(db_in, sheet_in, limit, offset)

    # define the generator
    def __iter__(self):
        conn_in = sqlite3.connect(self.db_in)

        sql_query = f"SELECT * FROM {self.sheet_in}"
        if self.limit:
            sql_query += f" LIMIT {self.limit}"
        if self.offset:
            sql_query += f" OFFSET {self.offset}"

        res = conn_in.execute(sql_query)
        
        yield_10ks = True
        while yield_10ks:
            row = res.fetchone()
            if row:
                yield row[7]
            else:
                yield_10ks = False
        conn_in.close()


class Form8KChunker(Chunker):
    def __init__(self, db_in, sheet_in, limit = None, offset = None):
        super().__init__(db_in, sheet_in, limit, offset)

    def __iter__(self):
        conn_in = sqlite3.connect(self.db_in)

        sql_query = f"SELECT * FROM {self.sheet_in}"
        if self.limit:
            sql_query += f" LIMIT {self.limit}"
        if self.offset:
            sql_query += f" OFFSET {self.offset}"

        res = conn_in.execute(sql_query)
        yield_8ks = True
        while yield_8ks:
            row = res.fetchone()
            if row:
                # just collect exhibits which are likely press releases and, thus, in a more natural company language
                if (row[8] in ['EX-99.1', 'EX-99.2', 'EX-99.3', 'EX-99','EX-99.4', 'EX-99.6', 'EX-99.9']):    
                    # exclude text with too many numbers and punctuation, these are either table exhibits or odd retrievals with excessive punctuation
                    freq_punct_and_numbers = count_numbers_and_punctuation(row[9])
                    if freq_punct_and_numbers > 0.30:
                        continue
                    else:
                        yield row[9]
            else:
                yield_8ks = False
        conn_in.close()


class EarningCallChunker(Chunker):
    def __init__(self, db_in, sheet_in, limit = None, offset = None):
        super().__init__(db_in, sheet_in, limit, offset)

    def __iter__(self):
        conn_in = sqlite3.connect(self.db_in)

        sql_query = f"SELECT DISTINCT content FROM {self.sheet_in}"
        if self.limit:
            sql_query += f" LIMIT {self.limit}"
        if self.offset:
            sql_query += f" OFFSET {self.offset}"

        res = conn_in.execute(sql_query)
        yield_ecs = True
        id_ec = 0
        while yield_ecs:
            row = res.fetchone()
            if row:
                text = row[0]
                sentences = text.split("\n")
                if len(sentences) < 5:
                    id_ec += 1
                    continue
                else:
                    try:
                        sentences = [sentence.split(':', maxsplit = 1)[1][1:] for sentence in sentences if len(sentence) > 5]
                        ec_text = " ".join(sentences)
                        id_ec += 1
                        yield ec_text
                    except:
                        self.logger.info(f"Something seems to be wrong with earning call number {id_ec}")
                        id_ec += 1
            else:
                yield_ecs = False
        conn_in.close()


class TRNewsChunker(Chunker):
    def __init__(self, db_in, sheet_in, limit = None, offset = None):
        super().__init__(db_in, sheet_in, limit, offset)

    def __iter__(self):
        conn_in = sqlite3.connect(self.db_in)

        sql_query = f"SELECT * FROM {self.sheet_in}"
        if self.limit:
            sql_query += f" LIMIT {self.limit}"
        if self.offset:
            sql_query += f" OFFSET {self.offset}"

        res = conn_in.execute(sql_query)
        yield_news = True
        while yield_news:
            row = res.fetchone()
            if row:
                text = row[1]
                news_text = " ".join(text.split("\n"))
                yield news_text
            else:
                yield_news = False
        conn_in.close()
