# Train a tokenizer

Currently, this is just a script which trains a tokenizer using texts which are coming from a generator. At the moment all text sources:

* Form 10K filings
* Form 8K filings
* Earning call transcripts
* TR News

The vocabulary size is rather large (50,000), digits are regarded as individual tokens, and the wordpiece algorithm is used. The first 3855 tokens (in the json file from the saved tokenizer) look pretty odd. I guess this may be related to encoding issues. 

At a later stage, we may want to experiment here with aspects regarding the vocab size, preprocessing and the algorithm. 