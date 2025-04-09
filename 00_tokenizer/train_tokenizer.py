from tokenizers import normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing
from tokenizers import Tokenizer


#from tokenizers import Tokenizer, models, trainers, normalizers, pre_tokenizers
#from tokenizers.processors import TemplateProcessing
#from tokenizers.normalizers import NFD, StripAccents, Lowercase, Strip
#from tokenizers.pre_tokenizers import Whitespace, Digits

import os
import sqlite3
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


tokenizer_name = "finlm-bpe-encoder-tokenizer.json"
if tokenizer_name in os.listdir():
    raise ValueError("Tokenizer with this name already exists.")

# define generator
def raw_text_generator():
    conn_filings = sqlite3.connect("/data/edgar/filings_data/filings.sqlite", check_same_thread=False)
    conn_trnews = sqlite3.connect("/data/trnews/trnews.sqlite", check_same_thread=False)
    conn_ecs = sqlite3.connect("/data/ecs/fmp/ec_fmp_full_2024.sqlite", check_same_thread=False)
    conn_esg = sqlite3.connect("/data/esg_reports/Raw_pdf_data.sqlite", check_same_thread=False)
    conn_ssrn = sqlite3.connect("/data/SSRN_Text_Data/ssrn_papers.sqlite", check_same_thread=False)

    res_10k = conn_filings.execute("SELECT * FROM form_tenk;")
    res_8k = conn_filings.execute("SELECT * FROM form_eightk;")
    res_trnews = conn_trnews.execute("SELECT * FROM news;")
    res_ecs = conn_ecs.execute("SELECT * FROM earningcalls_extended;")
    res_esg = conn_esg.execute("SELECT * FROM Raw_pdf_data")
    res_ssrn = conn_ssrn.execute("Select * From raw_texts")

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

    yield_ssrn = True
    while yield_ssrn:
        row = res_ssrn.fetchone()
        if row:
            yield row[0]
        else:
            yield_ssrn = False

    conn_filings.close()
    conn_trnews.close()
    conn_ecs.close()
    conn_esg.close()
    conn_ssrn.close()

# initialize generator 
raw_texts = raw_text_generator()

# define normalizer
normalizer = normalizers.Sequence([normalizers.NFC(), normalizers.Strip()])

# define pre_tokenizer
pre_tokenizer = pre_tokenizers.Sequence(
    [
        pre_tokenizers.ByteLevel(add_prefix_space = True, trim_offsets = True), 
        pre_tokenizers.Digits(individual_digits = True)
    ]
)

# define model for the tokenizer
model = BPE(unk_token="[unk]")

# define special tokens
special_tokens = ["[seq]", "[pad]", "[mask]", "[unk]"]

# initialize tokenizer
tokenizer = Tokenizer(model = model)
# add normalizer
tokenizer.normalizer = normalizer
# add pre_tokenizer
tokenizer.pre_tokenizer = pre_tokenizer

# define trainer
trainer = BpeTrainer(
    vocab_size = 50254,
    special_tokens = special_tokens,
    show_progress = True
)

logging.info("Training tokenizer starts.")
# Train the tokenizer
tokenizer.train_from_iterator(raw_texts, trainer)

# add post_processing routine
tokenizer.post_processor = TemplateProcessing(
    single="[seq] $A",  
    special_tokens=[
        ("[seq]", tokenizer.token_to_id("[seq]"))#, 
        #("[eos]", tokenizer.token_to_id("[eos]"))
    ]
)

# Save the tokenizer to disk
tokenizer.save(tokenizer_name)
logging.info("Training ended, tokenizer has been saved.")




####old version

# Initialize a tokenizer object
#tokenizer = Tokenizer(models.BPE())

# Setup normalization pipeline
#tokenizer.normalizer = normalizers.Sequence([Strip()])

# Setup pre-tokenizer
#tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Digits(individual_digits=True)])

# Initialize a trainer
#trainer = trainers.BpeTrainer(
#    vocab_size=50000,
#    min_frequency=5,
#    special_tokens=["<|endoftext|>", "<|pad|>"]
#)

#logging.info("Training tokenizer starts.")
# Train the tokenizer
#tokenizer.train_from_iterator(raw_texts, trainer)

# Define post-processing for the tokenizer
#tokenizer.post_processor = TemplateProcessing(
#    single="$A <|endoftext|>",  # Append the EOS token to a single sequence
    #pair="$A <|endoftext|> $B <|endoftext|>",   # Handle pair sequences (if applicable)
#    special_tokens=[
#        ("<|endoftext|>", tokenizer.token_to_id("<|endoftext|>"))
#    ]
#)

# Save the tokenizer to disk
#tokenizer.save(tokenizer_name)
#logging.info("Training ended, tokenizer has been saved.")
