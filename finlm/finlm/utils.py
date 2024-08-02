import re
import string
import torch

def split_text_into_chunks_by_words(text, max_words_per_chunk, ignore_first_sentences = None, ignore_last_sentences = None):
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


def count_numbers_and_punctuation(s):
    # Counters for digits and punctuation
    num_digits = sum(c.isdigit() for c in s)
    num_punctuation = sum(c in string.punctuation for c in s)
    len_string = len(s)
    
    if len_string == 0:
        return 0.0
    else:
        return (num_digits + num_punctuation) / len_string


def mask_tokens(inputs, mlm_probability, mask_token_id, special_token_ids, n_tokens, ignore_index = -100, hard_masking = False):
    device = inputs.device
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, mlm_probability, device=device)
    # create special_token_mask, first set all entries to false
    special_tokens_mask = torch.full(labels.shape, False, dtype = torch.bool, device = device)
    # flag all special tokens as true
    for sp_id in special_token_ids:
        special_tokens_mask = special_tokens_mask | (inputs == sp_id)

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    if ignore_index:
       labels[~masked_indices] = ignore_index  # We only compute loss on masked tokens

    if hard_masking:
        inputs[masked_indices] = mask_token_id
    else:
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device = device)).bool() & masked_indices
        inputs[indices_replaced] = mask_token_id 

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5, device = device)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(n_tokens, labels.shape, dtype=torch.long, device = device)
        inputs[indices_random] = random_words[indices_random]

    return inputs, labels


def replace_masked_tokens_randomly(inputs, mlm_probability, mask_token_id, special_token_ids, n_tokens, hard_masking = True):

    device = inputs.device

    masked_inputs, original_inputs = mask_tokens(
        inputs = inputs,
        mlm_probability = mlm_probability,
        mask_token_id = mask_token_id,
        special_token_ids = special_token_ids,
        n_tokens = n_tokens,
        ignore_index = None,
        hard_masking = hard_masking
        )
    
    masked_indices = torch.where(masked_inputs == mask_token_id, True, False)
    random_words = torch.randint(n_tokens, original_inputs.shape, dtype=torch.long, device = device)
    corrupted_inputs = original_inputs.clone()
    corrupted_inputs[masked_indices] = random_words[masked_indices]
    labels = torch.full(corrupted_inputs.shape, False, dtype=torch.bool, device=device)
    labels[masked_indices] = original_inputs[masked_indices] != corrupted_inputs[masked_indices]

    return corrupted_inputs, labels.float()


def replace_masked_tokens_from_generator(masked_inputs, original_inputs, logits, special_mask_id, discriminator_sampling = "multinomial"):
    
    device = masked_inputs.device
    discriminator_inputs = masked_inputs.clone()
    mask_indices = masked_inputs == special_mask_id

    if discriminator_sampling == "aggressive":
        sampled_ids = logits[mask_indices].argmax(-1)
    elif discriminator_sampling == "gumbel_softmax":
        sampled_ids = torch.nn.functional.gumbel_softmax(logits[mask_indices], hard = False).argmax(-1)
    else:
        sampled_ids = torch.multinomial(torch.nn.functional.softmax(logits[mask_indices], dim = -1), 1).squeeze()

    discriminator_inputs[mask_indices] = sampled_ids
    # initialize discriminator labels with False
    discriminator_labels = torch.full(masked_inputs.shape, False, dtype=torch.bool, device=device)
    # replace False with True if an id is sampled and not the same as the original one
    discriminator_labels[mask_indices] = discriminator_inputs[mask_indices] != original_inputs[mask_indices]
    # convert to float 
    discriminator_labels = discriminator_labels.float()

    return discriminator_inputs, discriminator_labels