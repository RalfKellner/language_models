import re
import string
import sqlite3
import pandas as pd
import logging
import sqlite3
import time
import string


class Chunker:
    
    """
    A base class for chunking text documents from a database.

    This class provides methods for counting documents in a database table and chunking text documents
    into approximately equal text chunks based on the number of words.

    Attributes
    ----------
    db_in : str
        The path to the input SQLite database.
    sheet_in : str
        The name of the table in the database containing the text documents.
    limit : int, optional
        The maximum number of documents to process (default is None).
    offset : int, optional
        The starting point from which to begin processing documents (default is None).
    n_documents : int
        The total number of documents in the specified table.
    logger : logging.Logger
        Logger instance for logging messages related to chunking operations.

    Methods
    -------
    count_documents() -> int
        Counts the number of documents in the specified table.
    chunk_to_database(db_out: str, sheet_out: str, chunk_size_in_words: int, ignore_first_sentences: int, ignore_last_sentences: int) -> None
        Chunks the documents into equal text chunks and saves them to a new database.
    split_text_into_chunks_by_words(text: str, max_words_per_chunk: int, ignore_first_sentences: int = None, ignore_last_sentences: int = None) -> list[str]
        Splits the input text into chunks based on the number of words, keeping sentences together.
    """

    def __init__(self, db_in: str, sheet_in: str, limit: int = None, offset: int = None) -> None:

        """
        Initializes the Chunker with database and table information.

        Parameters
        ----------
        db_in : str
            The path to the input SQLite database.
        sheet_in : str
            The name of the table in the database containing the text documents.
        limit : int, optional
            The maximum number of documents to process (default is None).
        offset : int, optional
            The starting point from which to begin processing documents (default is None).
        """
        
        self.db_in = db_in
        self.sheet_in = sheet_in
        self.limit = limit
        self.offset = offset
        self.n_documents = self.count_documents()
        self.logger = logging.getLogger(self.__class__.__name__)

    def count_documents(self) -> int:
        
        """
        Counts the number of documents in the specified table.

        Returns
        -------
        int
            The total number of documents in the specified table.
        """
        
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
            ignore_last_sentences: int,
            filter_numbers_and_punctuation: float = 0.15) -> None:

        """
        Chunks the documents into equal text chunks and saves them to a new database.

        This method processes each document in the specified table, splitting it into chunks of 
        approximately equal size (based on word count) and saving the resulting chunks to a new table.

        Parameters
        ----------
        db_out : str
            The path to the output SQLite database where the chunked sequences will be saved.
        sheet_out : str
            The name of the table where the chunked sequences will be saved.
        chunk_size_in_words : int
            The desired number of words per chunk.
        ignore_first_sentences : int
            The number of initial sentences to exclude from chunking.
        ignore_last_sentences : int
            The number of final sentences to exclude from chunking.
        """

        conn_out = sqlite3.connect(db_out)
        documents_excluded = 0
        n_raw_chunks = 0
        number_and_punctuation_frac = 0
        for doc_id, document in enumerate(self): # self is an element from the database from which documents are retrieved by the generator functions as defined in children classes
            try:
                # get raw text chunks
                text_chunks = self.split_text_into_chunks_by_words(document, chunk_size_in_words, ignore_first_sentences, ignore_last_sentences)
                # count the number of chunks before filtering
                n_raw_chunks += len(text_chunks)
                # determine the fraction of numbers and punctuation for each chunk
                chunk_numbers_and_punctution = [self.count_numbers_and_punctuation(chunk) for chunk in text_chunks]
                # count this value as well for averaging at the end of the loop
                number_and_punctuation_frac += sum(chunk_numbers_and_punctution)
                # filter chunks with large fraction of numbers and punctuation
                text_chunks = [chunk for chunk_i, chunk in enumerate(text_chunks) if chunk_numbers_and_punctution[chunk_i] <= filter_numbers_and_punctuation]
                if len(text_chunks) > 0:
                    text_chunks_df = pd.DataFrame({"sequence": text_chunks}, index = list(range(len(text_chunks))))
                    text_chunks_df.to_sql(sheet_out, conn_out, if_exists="append", index = False)
            except Exception as e:
                documents_excluded += 1
            if doc_id % 5000 == 0:
                self.logger.info(f"{doc_id/self.n_documents:.2%} of all documents have been chunked.")
        average_number_and_punctuation = number_and_punctuation_frac/n_raw_chunks
        self.logger.info(f"The average of numbers and punctuation over all raw chunks was {average_number_and_punctuation}. Chunks with a fraction higher than {filter_numbers_and_punctuation} have been removed before export to sequence database.")
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

        Parameters
        ----------
        text : str
            The input text to be split into chunks.
        max_words_per_chunk : int
            The maximum number of words allowed in each chunk.
        ignore_first_sentences : int, optional
            The number of initial sentences to exclude from chunking (default is None).
        ignore_last_sentences : int, optional
            The number of final sentences to exclude from chunking (default is None).

        Returns
        -------
        list[str]
            A list of text chunks, each containing up to `max_words_per_chunk` words.
        """

            # Split the text into sentences using a regular expression
        sentence_pattern = r'(?<!\d)(?<!\w\.\w)(?<![A-Z][a-z]\.)(?<=\.|\?)(?<!\d\.)\s'
        sentences = re.split(sentence_pattern, text) # old sentence pattern: r'(?<=[.!?]) +'

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

    @staticmethod
    def count_numbers_and_punctuation(text):
        count = len(text)
        count_np = sum(1 for char in text if char.isdigit() or char in string.punctuation)
        return count_np/count


class Form10KChunker(Chunker):

    """
    A class for chunking 10-K form filings from a database.

    This class inherits from `Chunker` and is specifically designed to process 10-K forms, 
    chunking them into approximately equal sequences and exporting the chunks to a new database.

    Methods
    -------
    __iter__()
        A generator that yields the text of 10-K forms from the database.
    """

    def __init__(self, db_in: str, sheet_in: str, limit: int = None, offset: int = None) -> None:
        
        """
        Initializes the Form10KChunker with database and table information.

        Parameters
        ----------
        db_in : str
            The path to the input SQLite database.
        sheet_in : str
            The name of the table in the database containing the 10-K forms.
        limit : int, optional
            The maximum number of documents to process (default is None).
        offset : int, optional
            The starting point from which to begin processing documents (default is None).
        """

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

    """
    A class for chunking 8-K form filings from a database.

    This class inherits from `Chunker` and is specifically designed to process 8-K forms, 
    focusing on press release statements (by Exhibit identifier) and filtering out documents with excessive punctuation or numbers.

    Methods
    -------
    __iter__()
        A generator that yields the text of 8-K forms from the database.
    count_numbers_and_punctuation(s: str) -> float
        Calculates the fraction of characters in a string that are digits or punctuation.
    """

    def __init__(self, db_in: str, sheet_in: str, limit: int = None, offset: int = None) -> None:
        
        """
        Initializes the Form8KChunker with database and table information.

        Parameters
        ----------
        db_in : str
            The path to the input SQLite database.
        sheet_in : str
            The name of the table in the database containing the 8-K forms.
        limit : int, optional
            The maximum number of documents to process (default is None).
        offset : int, optional
            The starting point from which to begin processing documents (default is None).
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

        """
        Calculates the fraction of characters in a string that are digits or punctuation.

        Parameters
        ----------
        s : str
            The input string to analyze.

        Returns
        -------
        float
            The fraction of characters in the string that are digits or punctuation.
        """

        # Counters for digits and punctuation
        num_digits = sum(c.isdigit() for c in s)
        num_punctuation = sum(c in string.punctuation for c in s)
        len_string = len(s)
        
        if len_string == 0:
            return 0.0
        else:
            return (num_digits + num_punctuation) / len_string



class EarningCallChunker(Chunker):
    
    """
    A class for chunking earnings call transcripts from a database.

    This class inherits from `Chunker` and is specifically designed to process earnings call transcripts, 
    filtering out participant names and short sentences, and then exporting the cleaned text to a new database.

    Methods
    -------
    __iter__()
        A generator that yields the cleaned text of earnings call transcripts from the database.
    """

    def __init__(self, db_in: str, sheet_in: str, limit: int = None, offset: int = None) -> None:
        
        """
        Initializes the EarningCallChunker with database and table information.

        Parameters
        ----------
        db_in : str
            The path to the input SQLite database.
        sheet_in : str
            The name of the table in the database containing the earnings call transcripts.
        limit : int, optional
            The maximum number of documents to process (default is None).
        offset : int, optional
            The starting point from which to begin processing documents (default is None).
        """

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
                        #sentences = [sentence.split(':', maxsplit = 1)[1][1:] for sentence in sentences if len(sentence) > 5]
                        sentences = [
                            sentence.split(':', maxsplit=1)[1] if ':' in sentence[:20] else sentence
                            for sentence in sentences
                            if len(sentence) > 10
                        ]
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

    """
    A class for chunking Thomson Reuters news articles from a database.

    This class inherits from `Chunker` and is specifically designed to process news articles, 
    splitting them into chunks and exporting the chunks to a new database.

    Methods
    -------
    __iter__()
        A generator that yields the text of news articles from the database.
    """

    def __init__(self, db_in: str, sheet_in: str, limit: int = None, offset: int = None) -> None:
        
        """
        Initializes the TRNewsChunker with database and table information.

        Parameters
        ----------
        db_in : str
            The path to the input SQLite database.
        sheet_in : str
            The name of the table in the database containing the news articles.
        limit : int, optional
            The maximum number of documents to process (default is None).
        offset : int, optional
            The starting point from which to begin processing documents (default is None).
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
        yield_news = True
        while yield_news:
            row = res.fetchone()
            if row:
                # check before starting new chunking
                yield row[1].replace("\n", " ")
            else:
                yield_news = False
        conn_in.close()


class EsgReportChunker(Chunker):

    def __init__(self, db_in: str, sheet_in: str, limit: int = None, offset: int = None) -> None:
        super().__init__(db_in, sheet_in, limit, offset)
        self.sentence_pattern = r'(?<!\d)(?<!\w\.\w)(?<![A-Z][a-z]\.)(?<=\.|\?)(?<!\d\.)\s'


    def __iter__(self):
        conn_in = sqlite3.connect(self.db_in)

        sql_query = f"SELECT * FROM {self.sheet_in}"
        if self.limit:
            sql_query += f" LIMIT {self.limit}"
        if self.offset:
            sql_query += f" OFFSET {self.offset}"

        res = conn_in.execute(sql_query)

        yield_esg_reports = True
        count = 0
        while yield_esg_reports:
            row = res.fetchone()
            if row:
                count += 1
                try:
                    content = row[4].replace("\n", " ")
                    sentences = re.split(self.sentence_pattern, content)
                    sentences = [sentence.strip() for sentence in sentences]
                    filtered_sentences = self.filter_sentences(sentences)
                    filtered_text = " ".join(filtered_sentences)
                    yield filtered_text
                except:
                    self.logger.info(f"Problem with report number: {count}")
            else:
                yield_esg_reports = False
        conn_in.close()

    @staticmethod
    def calculate_numeric_percentage(sentence):
        # Calculate the percentage of numeric content in the sentence
        total_chars = len(sentence)
        numeric_chars = sum(1 for char in sentence if char.isdigit())
        return (numeric_chars / total_chars) * 100

    def filter_sentences(self, sentences, max_numeric_percentage=15, max_percent_symbols=5):
        filtered_sentences = []
        for sentence in sentences:
            if len(sentence.strip()) > 0:
                numeric_percentage = self.calculate_numeric_percentage(sentence)
                percent_symbol_count = sentence.count("%")
                
                if percent_symbol_count <= max_percent_symbols and numeric_percentage <= max_numeric_percentage:
                    filtered_sentences.append(sentence)
        
        return filtered_sentences


def rename_table(database, old_table_name, new_table_name, retries=5, delay=1):

    """
    Renames a table in a SQLite database, with retries in case the database is locked.

    This function attempts to rename a table in the specified SQLite database. If the database is locked,
    it will retry the operation for a specified number of times with a delay between attempts.

    Parameters
    ----------
    database : str
        The path to the SQLite database file.
    old_table_name : str
        The current name of the table to be renamed.
    new_table_name : str
        The new name for the table.
    retries : int, optional
        The number of times to retry the operation if the database is locked (default is 5).
    delay : int, optional
        The number of seconds to wait between retry attempts (default is 1).

    Returns
    -------
    bool
        Returns True if the table was renamed successfully; otherwise, prints a failure message.

    Raises
    ------
    sqlite3.OperationalError
        If an unexpected database operation error occurs other than a locked database.
    """

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
    
    """
    Creates a new table with shuffled data from an existing table in a SQLite database.

    This function creates a new table in the specified SQLite database with the same schema as an existing table.
    The new table is populated with the data from the original table, but the rows are shuffled randomly.

    Parameters
    ----------
    database : str
        The path to the SQLite database file.
    original_table : str
        The name of the original table from which data is to be copied and shuffled.
    shuffled_table : str
        The name of the new table to create with shuffled data.

    Returns
    -------
    None
    """
    
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