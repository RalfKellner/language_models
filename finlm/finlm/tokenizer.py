from transformers import PreTrainedTokenizerFast


class FinLMTokenizer(PreTrainedTokenizerFast):

    """
    A tokenizer using the fast tokenizer wrapper class from transformers. This transformer comes with all functionalities as present for
    official models.
    """

    def __init__(
        self,
        tokenizer_file: str = None,
        bos_token: str = "[seq]",
        #eos_token: str = "[/seq]",
        unk_token: str = "[unk]",
        pad_token: str = "[pad]",
        mask_token: str = "[mask]",
        add_prefix_space: bool = False,
        trim_offsets: bool = True,
    ):
        super().__init__(
            tokenizer_file=tokenizer_file,
            bos_token=bos_token,
            #eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            add_prefix_space=add_prefix_space,
            trim_offsets=trim_offsets
        )



class FinLMGptTokenizer(PreTrainedTokenizerFast):

    """
    A tokenizer using the fast tokenizer wrapper class from transformers. This transformer comes with all functionalities as present for
    official models.
    """

    def __init__(
        self,
        tokenizer_file: str = None,
        eos_token: str = "<|endoftext|>",
        pad_token: str = "<|pad|>",
        add_prefix_space: bool = False,
        trim_offsets: bool = True,
    ):
        super().__init__(
            tokenizer_file=tokenizer_file,
            eos_token=eos_token,
            pad_token=pad_token,
            add_prefix_space=add_prefix_space,
            trim_offsets=trim_offsets
        )

    def __call__(self, *args, **kwargs):
        # Call the parent class's tokenization method
        encoded = super().__call__(*args, **kwargs)
        # Remove token_type_ids from the output
        encoded.pop("token_type_ids", None)
        return encoded