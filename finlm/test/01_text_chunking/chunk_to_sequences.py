from finlm.chunking import Form10KChunker, Form8KChunker, EarningCallChunker, TRNewsChunker, EsgReportChunker, SSRNChunker
from finlm.chunking import rename_table, shuffle_and_create_new_table
import sqlite3
import os
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

chunk_size = 450
database_out = "/data/language_models/test_outputs/01_chunking_results/chunked_sequences.sqlite"

conn = sqlite3.connect(database_out)
sql_query = conn.execute("SELECT name from sqlite_master;")
table_names = sql_query.fetchall()
table_names = [name[0] for name in table_names]

#if os.path.exists(database_out):
#    raise ValueError("Database for chunked sequences already exists, empty test folder or rename database.")

all_table_names = ["form_tenk", "form_eightk", "earning_calls", "tr_news", "esg_reports", "ssrn_papers"] 

# 10Ks
logging.info("Start with 10k chunking.")
if "form_tenk" in table_names:
    logging.info("It seems that chunks for this text type has already been chunked, moving on.")
    all_table_names.remove("form_tenk")
else:
    form_10k_chunker = Form10KChunker("/data/edgar/filings_data/filings.sqlite", "form_tenk", limit = 5)
    form_10k_chunker.chunk_to_database(database_out, "form_tenk", chunk_size, ignore_first_sentences=50, ignore_last_sentences=25)
    logging.info("10k is finished.")

# 8Ks
logging.info("Start with 8k chunking.")
if "form_eightk" in table_names:
    logging.info("It seems that chunks for this text type has already been chunked, moving on.")
    all_table_names.remove("form_eightk")
else:
    form_8k_chunker = Form8KChunker("/data/edgar/filings_data/filings.sqlite", "form_eightk", limit = 5)
    form_8k_chunker.chunk_to_database(database_out, "form_eightk", chunk_size, ignore_first_sentences=None, ignore_last_sentences=None)
    logging.info("8k chunking is finished.")

# ECs
logging.info("Start with earning call chunking.")
if "earning_calls" in table_names:
    logging.info("It seems that chunks for this text type has already been chunked, moving on.")
    all_table_names.remove("earning_calls")
else:    
    ec_chunker = EarningCallChunker("/data/ecs/fmp/ec_fmp_full_2024.sqlite", "earningcalls_extended", limit = 5)
    ec_chunker.chunk_to_database(database_out, "earning_calls", chunk_size, ignore_first_sentences=10, ignore_last_sentences=10)
    logging.info("Earning call chunking is finished.")

# TR News
logging.info("Start with TR news chunking.")
if "tr_news" in table_names:
    logging.info("It seems that chunks for this text type has already been chunked, moving on.")
    all_table_names.remove("tr_news")
else:
    news_chunker = TRNewsChunker("/data/trnews/trnews.sqlite", "news", limit = 5)
    news_chunker.chunk_to_database(database_out, "tr_news", chunk_size, ignore_first_sentences=2, ignore_last_sentences=3)
    logging.info("TR news chunking is finished.")

# ESG
logging.info("Start with ESG report chunking.")
if "esg_reports" in table_names:
    logging.info("It seems that chunks for this text type has already been chunked, moving on.")
    all_table_names.remove("esg_reports")
else:
    esg_chunker = EsgReportChunker("/data/esg_reports/Raw_pdf_data.sqlite", "Raw_pdf_data", limit = 5)
    esg_chunker.chunk_to_database(database_out, "esg_reports", chunk_size, ignore_first_sentences=5, ignore_last_sentences=5)
    logging.info("ESG report chunking is finished.")

# SSRN
logging.info("Start with SSRN chunking.")
if "ssrn_papers" in table_names:
    logging.info("It seems that chunks for this text type has already been chunked, moving on.")
    all_table_names.remove("ssrn_papers")
else:
    ssrn_chunker = SSRNChunker("/data/SSRN_Text_Data/ssrn_papers.sqlite", "raw_texts", limit = 5)
    ssrn_chunker.chunk_to_database(database_out, "ssrn_papers", chunk_size, ignore_first_sentences=0, ignore_last_sentences=0)
    logging.info("SSRN report chunking is finished.")


if len(all_table_names) > 0:
    # After chunking is finished, rename the tables, shuffle each table, save this and remove the original sequences
    old_table_names = all_table_names
    new_table_names = [t_name + "_og" for t_name in old_table_names] 
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
    logging.info("Chunking is finished.")
else:
    logging.info("No chunking has been done, it seems all tables already exist.")

