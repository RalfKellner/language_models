# Chunk text into sequences of comparable length

The chunk size which is specified in this script corresponds to the number of words. For instance with 200 words, we usually get around 220 up to 300 number of tokens with the wordpiece tokenizer. This can drastically change when shifting to the BPE tokenizer which usually produces more tokens. 

The script chunks documents into sequences with up to 200 words, hereby, it adds sentences which are identified by .!?

For the 10K filings and for earning call transcripts I leave out a specified number of the first and last sentences, because those usually include formal beginning and ending text which is rather repetitive and does not include "regular" language.

It is important to know that once the database with sequences is created, each table gets shuffled. This is a measure of pre-caution because I want to avoid systematic patterns. For instance if a batch would be trained with text documents from the year 2008, it likely would include more negative polarity than in other years. 