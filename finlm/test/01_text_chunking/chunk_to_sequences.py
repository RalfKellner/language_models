from finlm.chunking import Form10KChunker, Form8KChunker, EarningCallChunker, TRNewsChunker, rename_table, shuffle_and_create_new_table
import sqlite3
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

chunk_size = 200
database_out = "chunked_sequences.sqlite"

logging.info("Start with 10k chunking.")
form_10k_chunker = Form10KChunker("/data/edgar/filings_data/filings.sqlite", "form_tenk", limit = 5)
form_10k_chunker.chunk_to_database(database_out, "form_tenk", chunk_size, ignore_first_sentences=50, ignore_last_sentences=25)
logging.info("10k is finished.")
logging.info("Start with 8k chunking.")
form_8k_chunker = Form8KChunker("/data/edgar/filings_data/filings.sqlite", "form_eightk", limit = 5)
form_8k_chunker.chunk_to_database(database_out, "form_eightk", chunk_size, ignore_first_sentences=None, ignore_last_sentences=None)
logging.info("8k chunking is finished.")
logging.info("Start with earning call chunking.")
ec_chunker = EarningCallChunker("/data/ecs/fmp/ec_fmp_full_2024.sqlite", "earningcalls", limit = 5)
ec_chunker.chunk_to_database(database_out, "earning_calls", chunk_size, ignore_first_sentences=10, ignore_last_sentences=10)
logging.info("Earning call chunking is finished.")
logging.info("Start with TR news chunking.")
news_chunker = TRNewsChunker("/data/trnews/trnews.sqlite", "trnews", limit = 5)
news_chunker.chunk_to_database(database_out, "tr_news", chunk_size, ignore_first_sentences=None, ignore_last_sentences=None)
logging.info("TR news chunking is finished.")


# After chunking is finished, rename the tables, shuffle each table, save this and remove the original sequences
old_table_names = ["form_tenk", "form_eightk", "earning_calls", "tr_news"] 
new_table_names = ["form_tenk_og", "form_eightk_og", "earning_calls_og", "tr_news_og"] 
logging.info("Starting to randomize sequences in each table...")

for old_name, new_name in zip(old_table_names, new_table_names):
    logging.info(f"...first rename {old_name} to {new_name}")
    # rename tables
    rename_table(database_out, old_name, new_name)
    # shuffle and create new tables
    logging.info(f"...now shuffling table {new_name} and create a shuffled version of it with the name {old_name}")
    shuffle_and_create_new_table(database_out, new_name, old_name)
    # drop the original table
    logging.info(f"...delete table {new_name}")
    conn = sqlite3.connect(database_out)
    conn.execute(f"DROP TABLE {new_name}")
    conn.close()

logging.info("Randomization of sequences is finished.")

