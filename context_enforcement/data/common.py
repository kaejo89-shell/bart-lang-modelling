import abc
import datetime
import functools
import os
import re
from dataclasses import field, dataclass
import random
from typing import List, Dict, Optional, Union

import datasets
import numpy as np
import torch
import transformers
from basics.base import Base
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize


class ModelPreTrainingTask(Base):
    def __init__(self, tokenizer, max_seq_len=800, name=None):
        super().__init__(pybase_logger_name=name)
        self.tokenizer = tokenizer

        self.perturbations = [
            self.rotate_document_task,
            self.permutate_sentences_task,
            self.token_filling_task,
            self.token_masking_task,
            self.token_deletion_task,
        ]

        self.perturbations_text_domain = [
            self.rotate_document_task, self.permutate_sentences_task,
        ]

        self.perturbations_token_domain = [
            self.token_filling_task,
            self.token_masking_task, self.token_deletion_task,
        ]
        self.max_seq_len = max_seq_len

    @staticmethod
    def token_deletion_task(sequence: torch.Tensor,
                            mask_token_prob: float = 0.15,
                            ):
        """


        :param sequence:
        :param mask_token_prob:
        :return:
        """
        num_tokens = len(sequence)

        delete_mask = torch.rand(num_tokens) < mask_token_prob
        return sequence[~delete_mask]

    @staticmethod
    def rotate_document_task(document: str,
                             lang='english'):
        """

        :param document:
        :param lang:
        :return:
        """
        words = word_tokenize(document, language=lang)
        rotation_index = random.randint(0, len(words) - 1)
        rotated_doc = words[rotation_index:] + words[:rotation_index]
        return " ".join(rotated_doc)

    @staticmethod
    def permutate_sentences_task(document: str) -> str:
        """

        :param document:
        :return:
        """
        sentences = sent_tokenize(document)
        rand_order = torch.randperm(len(sentences))

        reordered_text = []
        for idx in rand_order:
            sent = sentences[idx].strip()
            if len(sent) == 0:
                continue
            reordered_text.append(sent)
        return " ".join(reordered_text)

    @staticmethod
    def token_masking_task(sequence: torch.Tensor,
                           mask_token_id: int,
                           mask_token_prob: float = 0.15,
                           special_token_list: Optional[List] = None,
                           ):
        """


        :param mask_token_id:
        :param sequence:
        :param mask_token_prob:
        :param special_token_list:
        :return:
        """

        if special_token_list is None:
            special_token_list = []
        num_tokens = len(sequence)
        for idx in range(num_tokens):
            mask_prob = torch.rand(1)
            if mask_prob < mask_token_prob:
                token = sequence[idx]
                if token in special_token_list:
                    continue
                sequence[idx] = mask_token_id
        return sequence

    @staticmethod
    def token_filling_task(sequence: torch.Tensor,
                           mask_token_id: int,
                           mask_token_prob: float = 0.15,
                           special_token_list: Optional[List] = None,
                           ):
        """
        This function/method works on tokenized sequence
        A number of text spans are sampled, with span lengths drawn from a Poisson distribution (λ = 3).
        Each span is replaced with a single [MASK] token. 0-length spans correspond to the insertion of
        [MASK] tokens. Text infilling is inspired by SpanBERT (Joshi et al., 2019), but SpanBERT samples
        span lengths from a different (clamped geometric) distribution, and replaces each span with a
        sequence of [MASK] tokens of exactly the same length. Text infilling teaches the model to predict
        how many tokens are missing from a span.

        :param sequence:
        :param mask_token_id:
        :param mask_token_prob:
        :param special_token_list:
        :return:
        """
        num_seq = len(sequence)
        span_length = int(torch.poisson(torch.tensor([3.0])))
        perturbed_ids = torch.empty(0, dtype=torch.long)
        if span_length > 0:
            for i in range(0, num_seq, span_length):
                if torch.rand(1) < mask_token_prob:
                    # check if the span does not contain special tokens
                    if not any(token in special_token_list for token in sequence[i: i + span_length]):
                        perturbed_ids = torch.cat(
                            (perturbed_ids, torch.tensor([mask_token_id], dtype=torch.long))
                        )
                else:
                    perturbed_ids = torch.cat(
                        (perturbed_ids, sequence[i: i + span_length])
                    )
        else:
            perturbed_ids = sequence  # if the span length is 0, the text is not perturbed
        return perturbed_ids

    def collate_fn(self, raw_text_list):
        """
        Collate function to be used in the dataloader.
        It applies the perturbations to the examples and returns the batch.
        TODO: improve efficiency
        :param raw_text_list: list of examples
        :return: batch ready to be fed to the model
        """

        original_texts = raw_text_list
        MAX_POSITION_EMBEDDINGS: int = self.max_seq_len
        input_ids = None
        for text in original_texts:
            perturbation_function = random.choice(self.perturbations)
            if perturbation_function in self.perturbations_text_domain:
                # need to truncate the text to 1024 tokens
                t_text = self.tokenizer(text, truncation=True, max_length=1024)
                text_truncated = self.tokenizer.decode(t_text["input_ids"], skip_special_tokens=True)
                perturbed_text = perturbation_function(text_truncated)
                perturbed_input_ids = self.tokenizer(
                    perturbed_text, return_tensors="pt", padding="max_length", truncation=True,
                    max_length=MAX_POSITION_EMBEDDINGS
                )["input_ids"][0]
            else:
                original_input_ids = self.tokenizer(
                    text, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_POSITION_EMBEDDINGS
                )["input_ids"][0]
                perturbed_input_ids = perturbation_function(
                    tokenized_sequence=original_input_ids,
                    mask_token_id=self.tokenizer.mask_token_id,
                    mask_probability=0.15,
                    list_special_tokens=self.tokenizer.all_special_ids,
                )
                if perturbed_input_ids.shape[-1] < MAX_POSITION_EMBEDDINGS:  # apply padding
                    perturbed_input_ids = torch.cat(
                        (perturbed_input_ids, torch.full((MAX_POSITION_EMBEDDINGS - perturbed_input_ids.shape[-1],),
                                                         self.tokenizer.pad_token_id,
                                                         dtype=torch.long)))
                perturbed_input_ids = torch.squeeze(perturbed_input_ids, dim=0)

            if input_ids is None:
                input_ids = perturbed_input_ids.unsqueeze(0)
            else:
                input_ids = torch.cat((input_ids, perturbed_input_ids.unsqueeze(0)), dim=0)

        tokenized_examples = {"input_ids": input_ids}
        # update the tokenized examples with the perturbed input ids and convert to tensors
        # update the attention mask
        tokenized_examples["attention_mask"] = [
            [1 if token_id != self.tokenizer.pad_token_id else 0 for token_id in input_ids]
            for input_ids in tokenized_examples["input_ids"]
        ]
        tokenized_examples["attention_mask"] = torch.tensor(tokenized_examples["attention_mask"])

        tokenized_examples["labels"] = self.tokenizer(
            original_texts, padding="max_length", truncation=True, max_length=MAX_POSITION_EMBEDDINGS,
            return_tensors="pt"
        )["input_ids"]

        return tokenized_examples


def fill_blanks(sentence, tag, options):
    assert tag in sentence, f'Error {tag} not found in {sentence}'
    tag_options = {tag: options}
    extended1 = [
        functools.reduce(lambda a,
                                kv: a.replace(*kv),
                         tag_options.items(),
                         re.sub('\s+', ' ',
                                ss.strip().replace('\n', ' '))) for ss in [sentence]][0]
    return extended1


def read_sentences(file, lower=False) -> List[str]:
    with open(file, 'r', encoding="utf-8") as o_file:
        sentences = []
        for s in o_file.readlines():
            ss = s.strip().lower() if lower else s.strip()
            sentences.append(ss)
    return sentences


def write_to_file(content, filename):
    fil = filename + '.txt'
    if os.path.exists(fil):
        os.remove(fil)
    with open(fil, 'x') as fwrite:
        fwrite.writelines("%s\n" % s for s in content)
    print('Done')
    return


def round_to_n(n, p=1):
    dec, integ = np.modf(n)
    val = integ + np.round(dec, p)
    return val


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round(elapsed))))


def normalize_whitespace(string):
    return re.sub(r'(\s)\1+', r'\1', string)


def create_text_tokenizer(model_base_name,
                          additional_tokens=None,
                          special_tokens=None,
                          ):
    """
    Creates a text tokenizer based on the specified models-base-name

    :param model_base_name:
    :param additional_tokens:
    :param special_tokens:
    :return:
    """
    if special_tokens is None:
        special_tokens = []
    if additional_tokens is None:
        additional_tokens = []
    tokenizer = AutoTokenizer.from_pretrained(model_base_name)
    tokenizer.add_tokens(additional_tokens)
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    return tokenizer


def pad_seq(
        seq: Union[np.ndarray, torch.Tensor, List], max_batch_len: int, pad_value: int
) -> List[int]:
    if len(seq) > max_batch_len:
        seq = seq.to(torch.long).unsqueeze(0)[:, :max_batch_len]
        return seq
    pads = torch.from_numpy(np.array([pad_value] * (max_batch_len - len(seq))))
    out = torch.concat([seq, pads], -1).to(torch.long).unsqueeze(0)
    return out


@dataclass
class Features:
    input_ids: Union[np.ndarray, torch.Tensor]
    attention_mask: Union[np.ndarray, torch.Tensor]
    labels: Optional[List[int]] = field(default_factory=list)
    decoder_attention_mask: Optional[List[int]] = field(default_factory=list)


class SmartCollator():
    def __init__(self,
                 pad_token_id: int,
                 context_max_len: int,
                 context_sampling_bounds: tuple,
                 label_pad_token_id: int = -100,
                 max_len: int = 512,
                 is_inference: bool = False):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id
        self.max_len = max_len
        self.is_inference = is_inference
        self.context_max_len = context_max_len
        self.context_sampling_bounds = context_sampling_bounds

    def __call__(self, batch: List[Features]) -> Dict[str, torch.Tensor]:
        batch_inputs: List = list()
        batch_attention_masks: List = list()
        decoder_attention_mask: List = list()
        labels: List = list()
        max_size = min([max([len(ex.input_ids)
                             for ex in batch]), self.max_len])

        max_size_output = min(
            [max([len(ex.labels) for ex in batch]), self.max_len])  # type: ignore

        for item in batch:
            batch_inputs += [pad_seq(item.input_ids,
                                     max_size, self.pad_token_id)]
            batch_attention_masks += [
                pad_seq(item.attention_mask, max_size, 0)]

            if not self.is_inference:
                labels += [pad_seq(item.labels, max_size_output,
                                   self.label_pad_token_id)]
                decoder_attention_mask += [
                    pad_seq(item.decoder_attention_mask, max_size_output, 0)
                ]

        input_ids = torch.concat(batch_inputs, 0)
        attention_mask = torch.concat(batch_attention_masks, 0)

        labels = torch.concat(labels, 0)
        decoder_attention_mask = torch.concat(decoder_attention_mask, 0)

        # Compute the context bounds for this batch
        # boundary = 0.45
        # boundary_start = int(input_ids.shape[-1] * boundary)
        # boundary_end = boundary_start + self.context_max_len

        context_boundary = compute_context_boundary(input_ids.shape[-1],
                                                    context_max_len=self.context_max_len,
                                                    context_sampling_bounds=self.context_sampling_bounds)

        if not self.is_inference:
            return dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                decoder_attention_mask=decoder_attention_mask,
                context_boundary=context_boundary
            )
        else:
            return dict(
                input_ids=torch.concat(batch_inputs, 0),
                attention_mask=torch.concat(batch_attention_masks, 0),
                context_boundary=context_boundary
            )


class DatasetProcessor(Dataset, metaclass=abc.ABCMeta):
    """
    Class handles the creation pytorch dataset object.

    """

    def __init__(self, tokenizer, data, use_special_token=True):
        self.tokenizer = tokenizer
        self.data = data
        self.use_special_token = use_special_token

    def __len__(self, ):
        return len(self.data)

    def __getitem__(self, idx):
        return self._process_data(self.data[idx])

    @abc.abstractmethod
    def _process_data(self, param):
        raise NotImplemented("Function not implemented")


"""
The class below is from: https://github.com/MorenoLaQuatra/bart-it/blob/main/data/pretraining_dataset.py
"""


class PretrainingDataset(torch.utils.data.Dataset):
    """
    This class is intended for sequence classification tasks.
    :param source_text: List of source text that is used as input to the model.
    :param stream_dataset: IterableDataset that is used as input to the model.
    :param tokenizer: The tokenizer to be used for tokenizing the texts. It can be an instance of the transformers AutoTokenizer class or a custom tokenizer.
    :param max_input_length: The maximum length of the tokenized input text.
    :param max_output_length: The maximum length of the tokenized output text.
    :param padding: The padding strategy to be used. Available options are available in the transformers library.
    :param truncation: Whether to truncate the text or not.
    :param is_streaming: Whether the dataset is a stream dataset or not.
    """

    def __init__(
            self,
            source_text: List[str] = None,
            stream_dataset: datasets.IterableDataset = None,
            tokenizer: transformers.PreTrainedTokenizer = None,
            max_input_length: int = 1024,
            max_output_length: int = 1024,
            padding: str = "max_length",
            truncation: bool = True,
            is_streaming: bool = False,
            mask_token: Optional[str] = None
    ):

        self._mask_token = mask_token
        if self._mask_token is None:
            self._mask_token = "[MASK] "

        if is_streaming:
            self._check_none(
                [
                    ("source_text", source_text),
                ]
            )

            self._check_not_none(
                [
                    ("tokenizer", tokenizer),
                    ("stream_dataset", stream_dataset),
                ]
            )
            self.stream_dataset = stream_dataset
        else:
            self._check_none(
                [
                    ("stream_dataset", stream_dataset),
                ]
            )
            self._check_not_none(
                [
                    ("tokenizer", tokenizer),
                    ("source_text", source_text),
                ]
            )
            self.source_text = source_text

        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.tokenizer = tokenizer
        self.padding = padding
        self.truncation = truncation
        self.is_streaming = is_streaming

    def _check_not_none(self, list_values: List):
        """
        This function checks if any of the arguments is None.
        :param args: The arguments to be checked.
        """
        for arg_name, arg in list_values:
            if arg is None:
                raise ValueError(f"The argument {arg_name} cannot be None.")

    def _check_none(self, list_values: List):
        """
        This function checks if any of the arguments is not None.
        :param args: The arguments to be checked.
        """
        for arg_name, arg in list_values:
            if arg is not None:
                raise ValueError(f"The argument {arg_name} must be None.")

    def sentence_permutation(self, document: str) -> str:
        """
        A document is divided into sentences based on full stops, and these sentences are shuffled in a random order.
        :param sentence: The sentence to be permuted.
        :return: The permuted sentence.
        """
        sentences = sent_tokenize(document)
        permuted_sentences = torch.randperm(len(sentences))
        permuted_sentence = []
        for idx in permuted_sentences:
            permuted_sentence.append(sentences[idx])
        return ' '.join(permuted_sentence)

    def text_infilling(self, text: str) -> str:
        """
        A number of text spans are sampled, with span lengths drawn from a Poisson distribution (λ = 3).
        Each span is replaced with a single [MASK] token. 0-length spans correspond to the insertion of
        [MASK] tokens. Text infilling is inspired by SpanBERT (Joshi et al., 2019), but SpanBERT samples
        span lengths from a different (clamped geometric) distribution, and replaces each span with a
        sequence of [MASK] tokens of exactly the same length. Text infilling teaches the model to predict
        how many tokens are missing from a span
        :param text: The text to be infilled.
        :return: The infilled text.
        """
        text = word_tokenize(text)
        text_length = len(text)
        infilled_text = ""
        for i in range(text_length):
            if torch.rand(1) < 0.15:
                if torch.rand(1) < 0.8:
                    infilled_text += self._mask_token
                else:
                    if torch.rand(1) < 0.5:
                        infilled_text += text[i] + " "
                    else:
                        infilled_text += self._mask_token
            else:
                infilled_text += text[i] + " "
        return infilled_text

    def token_masking(self, text: str) -> str:
        """
        Random tokens are replaced with the [MASK] token. This task trains the model to predict the original value of the masked tokens.
        :param text: The text to be masked.
        :return: The masked text.
        """
        text = text.split(" ")
        text_length = len(text)
        masked_text = ""
        for i in range(text_length):
            if torch.rand(1) < 0.15:
                masked_text += self._mask_token
            else:
                masked_text += text[i] + " "
        return masked_text

    def token_deletion(self, text: str) -> str:
        """
        Random tokens are deleted from the input. In contrast to token masking,
        the model must decide which positions are missing inputs.
        :param text: The text to be token deleted.
        :return: The token deleted text.
        """
        text = text.split(" ")
        text_length = len(text)
        deleted_text = ""
        for i in range(text_length):
            if torch.rand(1) < 0.15:
                deleted_text += ""
            else:
                deleted_text += text[i] + " "
        return deleted_text

    def document_rotation(self, document: str) -> str:
        """
        A token is chosen uniformly at random, and the document is rotated so that it begins with that token.
        This task trains the model to identify the start of the document.
        :param document: The document to be rotated.
        :return: The rotated document.
        """
        document = document.split(" ")
        document_length = len(document)
        random_index = torch.randint(0, document_length, (1,)).item()
        rotated_document = ""
        for i in range(document_length):
            rotated_document += document[(i + random_index) % document_length] + " "
        return rotated_document

    def __getitem__(self, idx):
        """
        This function is called to get the tokenized source and target text for a given index.
        :param idx: The index of the text and label to be returned.
        :return: A dictionary containing the tokenized source (`input_ids`) with attention mask (`attention_mask`) and the tokenized target (`labels`).
        """
        if self.is_streaming:
            text = list(self.stream_dataset.skip(idx).take(1))[0]["text"]
        else:
            text = self.source_text[idx]

        print("TEXT: ", text)

        # this function applies perturbation to the text that is passed to the model as input
        perturbation_function = random.choice(
            [
                self.sentence_permutation,
                self.text_infilling,
                self.token_masking,
                self.token_deletion,
                self.document_rotation,
            ]
        )
        print("\n\n*********\nPERTURBATION FUNCTION: ", perturbation_function)

        input_text = perturbation_function(text)

        print("INPUT TEXT: ", input_text)

        # the input of the model is the perturbed text
        input = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="pt",
        )

        # the output of the model is the correct text that is passed to the model as target
        output = self.tokenizer(
            text_target=text,
            max_length=self.max_output_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="pt",
        )

        item = {
            "input_ids": input["input_ids"].squeeze(),
            "attention_mask": input["attention_mask"].squeeze(),
            "labels": output["input_ids"].squeeze(),
        }

        return item

    def __len__(self):
        """
        This function is called to get the length of the dataset.
        :return: The length of the dataset.
        """
        return len(self.source_text)
