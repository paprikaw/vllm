from ast import pattern
from typing import Tuple

from pyparsing import rest_of_line
from benchmark_dataset import BenchmarkDataset, SampleRequest
from transformers import PreTrainedTokenizerBase
import numpy as np
import logging

logger = logging.getLogger(__name__)

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))


class PatternDataset(BenchmarkDataset):
    # Default values copied from benchmark_serving.py for the random dataset.
    DEFAULT_PREFIX_LEN = 0
    DEFAULT_RANGE_RATIO = 0.0
    DEFAULT_INPUT_OUTPUT_LEN = [(1024, 128)]

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
    def pattern_sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: list[int],
        prefix_len: int = DEFAULT_PREFIX_LEN,
        range_ratio: float = DEFAULT_RANGE_RATIO,
        input_output_len: list[Tuple[int,int]] = DEFAULT_INPUT_OUTPUT_LEN,
        **kwargs,
    ) -> list[SampleRequest]:

        # 我们根据request rate和total request，和change interval来计算
        # 每一个change interval中需要放多少个request
        assert len(num_requests) == len(input_output_len)
        requests: list[SampleRequest] = []
        for i, num_cur_request in enumerate(num_requests):
            assert num_cur_request > 0
            input_len, output_len = input_output_len[i]

            # Enforce range_ratio < 1
            assert range_ratio < 1.0, (
                "random_range_ratio must be < 1.0 to ensure a valid sampling range"
            )
            vocab_size = tokenizer.vocab_size
            num_special_tokens = tokenizer.num_special_tokens_to_add()
            real_input_len = input_len - num_special_tokens

            prefix_token_ids = (
                np.random.randint(0, vocab_size, size=prefix_len).tolist()
                if prefix_len > 0
                else []
            )

            # New sampling logic: [X * (1 - b), X * (1 + b)]
            input_low = int(real_input_len * (1 - range_ratio))
            input_high = int(real_input_len * (1 + range_ratio))
            output_low = int(output_len * (1 - range_ratio))
            output_high = int(output_len * (1 + range_ratio))

            # Add logging for debugging
            logger.info("Sampling input_len from [%s, %s]", input_low, input_high)
            logger.info("Sampling output_len from [%s, %s]", output_low, output_high)

            input_lens = np.random.randint(input_low, input_high + 1, size=num_cur_request)
            output_lens = np.random.randint(output_low, output_high + 1, size=num_cur_request)
            offsets = np.random.randint(0, vocab_size, size=num_cur_request)
            for i in range(num_cur_request):
                inner_seq = (
                    (offsets[i] + i + np.arange(input_lens[i])) % vocab_size
                ).tolist()
                token_sequence = prefix_token_ids + inner_seq
                prompt = tokenizer.decode(token_sequence)
                # After decoding the prompt we have to encode and decode it again.
                # This is done because in some cases N consecutive tokens
                # give a string tokenized into != N number of tokens.
                # For example for GPT2Tokenizer:
                # [6880, 6881] -> ['Ġcalls', 'here'] ->
                # [1650, 939, 486] -> ['Ġcall', 'sh', 'ere']
                # To avoid uncontrolled change of the prompt length,
                # the encoded sequence is truncated before being decode again.
                re_encoded_sequence = tokenizer.encode(prompt, add_special_tokens=False)[
                    : input_lens[i]
                ]
                prompt = tokenizer.decode(re_encoded_sequence)
                total_input_len = prefix_len + int(input_lens[i])
                requests.append(
                    SampleRequest(
                        prompt=prompt,
                        prompt_len=total_input_len,
                        expected_output_len=int(output_lens[i]),
                    )
                )
        return requests

    def sample(
        self, tokenizer: PreTrainedTokenizerBase, num_requests: int
    ) -> list[SampleRequest]:
        """
        Abstract method to generate sample requests from the dataset.

        Subclasses must override this method to implement dataset-specific logic
        for generating a list of SampleRequest objects.

        Args:
            tokenizer (PreTrainedTokenizerBase): The tokenizer to be used
             for processing the dataset's text.
            num_requests (int): The number of sample requests to generate.

        Returns:
            list[SampleRequest]: A list of sample requests generated from the
            dataset.
        """
        raise NotImplementedError("sample must be implemented in subclasses.")


def split_list(lst: list, n)->list[list]:
    """将 lst 尽量均衡分成 n 份"""
    length = len(lst)
    k, r = divmod(length, n)  # k是基础大小，r是多出来的个数
    result = []
    start = 0
    for i in range(n):
        end = start + k + (1 if i < r else 0)
        result.append(lst[start:end])
        start = end
    return result