import re
import string
import sqlite3
import pandas as pd
import logging
import sqlite3
import time

class Chunker:
    def __init__(self, db_in: str, sheet_in: str, limit: int = None, offset: int = None) -> None:

        """
            A parent class for chunking classes which are specifically designed for different financial text sources.

            db_in: name of the database location which is supposed to be processed
            sheet_in: table name of the database where the text files are located
            limit: limited number of observations from the table
            offset: collect observations from here on
        """
        
        self.db_in = db_in
        self.sheet_in = sheet_in
        self.limit = limit
        self.offset = offset
        self.n_documents = self.count_documents()
        self.logger = logging.getLogger(self.__class__.__name__)

    def count_documents(self) -> int:
        
        """A method to determine the number of sequences in a sheet."""
        
        conn_in = sqlite3.connect(self.db_in)
        count_n = conn_in.execute(f"SELECT COUNT(*) FROM {self.sheet_in};")
        n_docs = count_n.fetchall()
        conn_in.close()
        n_docs = n_docs[0][0]
        return n_docs
    
    def chunk_to_database(
            self, db_out: str, 
            sheet_out: str, 
            chunk_size_in_words: int, 
            ignore_first_sentences: int, 
            ignore_last_sentences: int) -> None:

        """
            A method to chunk documents into approximate equal text chunks.

            db_out: the location of the database where to commit the chunked text sequences
            sheet_out: the table name where to commit the chunked text sequences
            chunk_size_in_words: number of desired words per chunk, Note: words, not tokens!
            ignore_first_sentences: the number of first sentences which are not going to be exported via the chunking process
                makes sense for documents such as 10K form filings as the first sentences often include just formal definitions which do not include 
                relevant content.
            ignore_last_sentences: the number of last sentences which are not going to be exported vie the chunking process
        """

        conn_out = sqlite3.connect(db_out)
        documents_excluded = 0
        for doc_id, document in enumerate(self): # self is an element from the database from which documents are retrieved by the generator functions as defined in children classes
            try:
                text_chunks = self.split_text_into_chunks_by_words(document, chunk_size_in_words, ignore_first_sentences, ignore_last_sentences)
                text_chunks_df = pd.DataFrame({"sequence": text_chunks}, index = list(range(len(text_chunks))))
                text_chunks_df.to_sql(sheet_out, conn_out, if_exists="append", index = False)
            except Exception as e:
                documents_excluded += 1
            if doc_id % 5000 == 0:
                self.logger.info(f"{doc_id/self.n_documents:.2%} of all documents have been chunked.")
        conn_out.close()
        self.logger.info(f"A fraction of {documents_excluded/self.n_documents:.4f} has been excluded while chunking.")

    
    @staticmethod
    def split_text_into_chunks_by_words(
            text: str, 
            max_words_per_chunk: str, 
            ignore_first_sentences: int = None, 
            ignore_last_sentences: int = None) -> list[str]:
        
        """
        Splits the input text into chunks based on the number of words, keeping sentences together.
        
        Args:
        text (str): The input text to be split.
        max_words_per_chunk (int): The maximum number of words per chunk.
        
        Returns:
        List[str]: A list of text chunks.
        """
        # Split the text into sentences using a regular expression
        sentences = re.split(r'(?<=[.!?]) +', text)

        if ignore_first_sentences and not(ignore_last_sentences):
            assert ignore_first_sentences < len(sentences), "The number of sentences must be larger than the number of sentences to be ignored"
            sentences = sentences[ignore_first_sentences:]
        elif not(ignore_first_sentences) and ignore_last_sentences:
            assert ignore_last_sentences < len(sentences), "The number of sentences must be larger than the number of sentences to be ignored"
            sentences = sentences[:-ignore_last_sentences]
        elif ignore_first_sentences and ignore_last_sentences:
            assert (ignore_last_sentences + ignore_last_sentences) < len(sentences), "The number of sentences must be larger than the number of sentences to be ignored"
            sentences = sentences[ignore_first_sentences:-ignore_last_sentences]
        
        # Initialize variables
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        # Iterate over sentences and form chunks
        for sentence in sentences:
            sentence_word_count = len(sentence.split())
            
            # If adding the next sentence exceeds the max words per chunk, finalize the current chunk
            if current_word_count + sentence_word_count > max_words_per_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_word_count = 0
            
            # Add the sentence to the current chunk
            current_chunk.append(sentence)
            current_word_count += sentence_word_count
        
        # Add the last chunk if there are any sentences left
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks


class Form10KChunker(Chunker):
    def __init__(self, db_in: str, sheet_in: str, limit: int = None, offset: int = None) -> None:
        """A class which collects 10K form flings from our database, chunks them into approximate equal seqences and exports these to a new database"""
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
                yield row[7]  # this is the report column
            else:
                yield_10ks = False
        conn_in.close()


class Form8KChunker(Chunker):
    def __init__(self, db_in: str, sheet_in: str, limit: int = None, offset: int = None) -> None:
        """
            A class which collects 8K form flings from our database, chunks them into approximate equal seqences and exports these to a new database.
            We try to collect press release statements which are usually accompanied by Exhibit identifiers as defined below. Furthermore, in the
            database a few of these releases look add and can be identified by a large fraction of punctuation and numbers which is why this method
            is included and used in the 8K chunker class. 
        """
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
                    freq_punct_and_numbers = self.count_numbers_and_punctuation(row[9])
                    if freq_punct_and_numbers > 0.30:
                        continue
                    else:
                        yield row[9]
            else:
                yield_8ks = False
        conn_in.close()

    @staticmethod
    def count_numbers_and_punctuation(s: str) -> float:
        # Counters for digits and punctuation
        num_digits = sum(c.isdigit() for c in s)
        num_punctuation = sum(c in string.punctuation for c in s)
        len_string = len(s)
        
        if len_string == 0:
            return 0.0
        else:
            return (num_digits + num_punctuation) / len_string



class EarningCallChunker(Chunker):
    def __init__(self, db_in: str, sheet_in: str, limit: int = None, offset: int = None) -> None:
        """A class which collects earning call transcripts collected at Financial Modeling Prep API, chunks them into approximate equal seqences and exports these to a new database"""
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
                # transcripts with very little sentences are not reasonable and we have a few empty transcripts which are filtered by this
                if len(sentences) < 5:
                    id_ec += 1
                    continue
                else:
                    try:
                        # this makes sure that names of earning call participants are excluded and each sentence has more than 5 string symbols
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
    def __init__(self, db_in: str, sheet_in: str, limit: int = None, offset: int = None) -> None:
        """A class which collects news from the Thomson Reuters news database, chunks them into approximate equal seqences and exports these to a new database"""
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


def rename_table(database, old_table_name, new_table_name, retries=5, delay=1):
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(database)
        conn.execute('PRAGMA busy_timeout = 5000')  # Set busy timeout to 5000 milliseconds
        cur = conn.cursor()
        
        # Define the rename command
        rename_command = f"ALTER TABLE {old_table_name} RENAME TO {new_table_name}"
        
        for attempt in range(retries):
            try:
                # Execute the rename command
                cur.execute(rename_command)
                conn.commit()
                print(f"Table renamed to {new_table_name} successfully.")
                return True
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e):
                    print(f"Attempt {attempt + 1} of {retries}: Database is locked, retrying after {delay} seconds...")
                    time.sleep(delay)
                else:
                    raise
        print("Failed to rename the table after multiple attempts.")
    finally:
        # Close the connection
        conn.close()

def shuffle_and_create_new_table(database, original_table, shuffled_table):
    # Connect to the database
    conn = sqlite3.connect(database)
    cur = conn.cursor()

    # Step 1: Create the new table with the same schema as the original table
    cur.execute(f"CREATE TABLE {shuffled_table} AS SELECT * FROM {original_table} WHERE 0")
    conn.commit()

    # Step 2: Insert shuffled data into the new table
    cur.execute(f"INSERT INTO {shuffled_table} SELECT * FROM {original_table} ORDER BY RANDOM()")
    conn.commit()

    # Close the connection
    conn.close()