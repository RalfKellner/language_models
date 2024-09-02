from tokenizers import Tokenizer, models, trainers, normalizers, pre_tokenizers
from tokenizers.processors import TemplateProcessing
from tokenizers.normalizers import NFD, StripAccents, Lowercase, Strip
from tokenizers.pre_tokenizers import Whitespace, Digits
import sqlite3
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# define generator
def raw_text_generator():
    conn_filings = sqlite3.connect("/data/edgar/filings_data/filings.sqlite", check_same_thread=False)
    conn_trnews = sqlite3.connect("/data/trnews/trnews.sqlite", check_same_thread=False)
    conn_ecs = sqlite3.connect("/data/ecs/fmp/ec_fmp_full_2024.sqlite", check_same_thread=False)
    conn_esg = sqlite3.connect("/data/esg_reports/Raw_pdf_data.sqlite", check_same_thread=False)

    res_10k = conn_filings.execute("SELECT * FROM form_tenk;")
    res_8k = conn_filings.execute("SELECT * FROM form_eightk;")
    res_trnews = conn_trnews.execute("SELECT * FROM news;")
    res_ecs = conn_ecs.execute("SELECT * FROM earningcalls_extended;")
    res_esg = conn_esg.execute("SELECT * FROM Raw_pdf_data")

    yield_10ks = True
    while yield_10ks:
        row = res_10k.fetchone()
        if row:
            yield row[7]
        else:
            yield_10ks = False 

    yield_8ks = True
    while yield_8ks:
        row = res_8k.fetchone()
        if row:
            yield row[9]
        else:
            yield_8ks = False

    yield_trnews = True
    while yield_trnews:
        row = res_trnews.fetchone()
        if row:
            yield row[1].replace("\n", " ")
        else:
            yield_trnews = False

    yield_ecs = True
    while yield_ecs:
        row = res_ecs.fetchone()
        if row:
            yield row[3]
        else:
            yield_ecs = False

    yield_esg = True
    while yield_esg:
        row = res_esg.fetchone()
        if row and not(row[4] == None):
            yield row[4].replace("\n", " ")
        else:
            yield_esg = False

    conn_filings.close()
    conn_trnews.close()
    conn_ecs.close()
    conn_esg.close()

# initialize generator 
raw_texts = raw_text_generator()

# Initialize a tokenizer object
tokenizer = Tokenizer(models.WordPiece(unk_token="[unk]"))

# Setup normalization pipeline
tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), Strip(), StripAccents()])

# Setup pre-tokenizer
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Digits(individual_digits=True)])

# Initialize a trainer for WordPiece model
trainer = trainers.WordPieceTrainer(
    vocab_size=50000,
    min_frequency=5,
    special_tokens=["[pad]", "[seq]", "[/seq]", "[unk]", "[mask]"]
)

logging.info("Training tokenizer starts.")
# Train the tokenizer
tokenizer.train_from_iterator(raw_texts, trainer)

# Define post-processing for the tokenizer
tokenizer.post_processor = TemplateProcessing(
    single="[seq] $A [/seq]",
    special_tokens=[
        ("[seq]", tokenizer.token_to_id("[seq]")),
        ("[/seq]", tokenizer.token_to_id("[/seq]"))
    ],
)

# Save the tokenizer to disk
tokenizer.save("finlm-wordpiece-tokenizer.json")
logging.info("Training ended, tokenizer has been saved.")
