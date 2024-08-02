from finlm.chunking import Form10KChunker, Form8KChunker, EarningCallChunker, TRNewsChunker
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

chunk_size = 200

logging.info("Start with 10k chunking.")
form_10k_chunker = Form10KChunker("/data/edgar/filings_data/filings.sqlite", "form_tenk")
form_10k_chunker.chunk_to_database("/data/finlm_sequences/finlm_chunks.sqlite", "form_tenk", chunk_size, ignore_first_sentences=50, ignore_last_sentences=25)
logging.info("10k is finished.")
logging.info("Start with 8k chunking.")
form_8k_chunker = Form8KChunker("/data/edgar/filings_data/filings.sqlite", "form_eightk")
form_8k_chunker.chunk_to_database("/data/finlm_sequences/finlm_chunks.sqlite", "form_eightk", chunk_size, ignore_first_sentences=None, ignore_last_sentences=None)
logging.info("8k chunking is finished.")
logging.info("Start with earning call chunking.")
ec_chunker = EarningCallChunker("/data/ecs/fmp/ec_fmp_full_2024.sqlite", "earningcalls")
ec_chunker.chunk_to_database("/data/finlm_sequences/finlm_chunks.sqlite", "earning_calls", chunk_size, ignore_first_sentences=10, ignore_last_sentences=10)
logging.info("Earning call chunking is finished.")
logging.info("Start with TR news chunking.")
news_chunker = TRNewsChunker("/data/trnews/trnews.sqlite", "trnews")
news_chunker.chunk_to_database("/data/finlm_sequences/finlm_chunks.sqlite", "tr_news", chunk_size, ignore_first_sentences=None, ignore_last_sentences=None)
logging.info("TR news chunking is finished.")