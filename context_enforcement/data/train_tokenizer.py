import argparse
import os
import pathlib
import shutil
from pathlib import Path
from typing import List, Union, Optional

from transformers import AutoTokenizer, BartTokenizer
from datasets import load_dataset, concatenate_datasets
from basics.base import Base
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


class TokenizerTrainer(Base):
    def __init__(self,
                 special_tokens: List[str] = None,
                 name=None, ):
        super().__init__(pybase_logger_name=name)
        if special_tokens is None:
            special_tokens = ["<pad>",
                              "<s>",
                              "</s>",
                              "<mask>",
                              "<unk>",
                              "[CLS]",
                              "[SEP]",
                              " "
                              ]

        self._tokenizer = Tokenizer(BPE())
        self._tokenizer.pre_tokenizer = Whitespace()
        self._trainer = BpeTrainer(special_tokens=special_tokens)

    def train_from_files(self, file_list: List[Union[str, Path]], output_path: str,
                         target_tokenizer_class: Optional[callable] = None):
        """

        :param file_list:
        :param output_path:
        :return:
        """
        self._tokenizer.train(file_list, trainer=self._trainer, )
        self._save_tokenizer(output_path, target_tokenizer_class=target_tokenizer_class)

    def train_from_iterator(self, sentence_list,
                            output_path: str,
                            num_examples: Optional[int] = None,
                            target_tokenizer_class: Optional[callable] = None):
        """


        :param target_tokenizer_class:
        :param sentence_list:
        :param output_path:
        :return:
        """

        if isinstance(sentence_list, list):
            num_examples = len(sentence_list)
        self._tokenizer.train_from_iterator(iterator=sentence_list, length=num_examples, trainer=self._trainer)
        self._save_tokenizer(output_path, target_tokenizer_class)

    def _save_tokenizer(self,
                        output_path,
                        target_tokenizer_class: Optional[AutoTokenizer] = None):
        """

        :param output_path:
        :return:
        """

        if target_tokenizer_class is None:
            self._log.info(f"Setting the target tokenizer class to {BartTokenizer.__name__}")
            target_tokenizer_class = BartTokenizer

        self._log.info("Saving trained tokenizer")
        os.makedirs('tokenizer-tmp', exist_ok=True)
        self._tokenizer.model.save('tokenizer-tmp')
        target_tokenizer = target_tokenizer_class(vocab_file='tokenizer-tmp/vocab.json',
                                                  merges_file='tokenizer-tmp/merges.txt')
        target_tokenizer.save_pretrained(output_path)

        # Remove the temp folder
        shutil.rmtree('tokenizer-tmp')

        self._log.info(f"Tokenizer trained and stored @ {output_path}. "
                       f"It can be loaded via `AutoTokenizer.from_pretrained('{output_path}')`")


# def train_tokenizer_from_datasets(dataset_name_list,
#                                   output_path: str,
#                                   target_tokenizer_class: Optional[callable] = None):
#     dataset_list = []
#     for ds_name in dataset_name_list:
#         ds = load_dataset(ds_name,
#                           split='train', streaming=False)
#         dataset_list.append(ds)
#     all_datasets = concatenate_datasets(dataset_list,
#                                         split='train')
#
#     def iters():
#         for example in all_datasets:
#             yield example['summary']
#
#     self.train_from_iterator(sentence_list=iters(), output_path=output_path, num_examples=len(all_datasets['summary']),
#                              target_tokenizer_class=target_tokenizer_class)


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path',
                        type=str,
                        required=True,
                        help="Location to save the tokenization output files"
                        )
    parser.add_argument('--input-file-list',
                        type=str,
                        required=False,
                        nargs='+',
                        metavar='N',
                        help="Location of the files containing the raw text"
                        )

    parser.add_argument('--dataset-name-list',
                        type=str,
                        required=False,
                        nargs='+',
                        metavar='N',
                        help="Name or id of the datasets to load with load_dataset function"
                        )

    return parser


if __name__ == '__main__':
    parser = add_arguments()
    args = parser.parse_args()
    run_config = vars(args)

    input_file_list = run_config['input_file_list']
    output_path = run_config['output_path']
    file_extension = run_config['file_extension']
    dataset_name_list = run_config['dataset_name_list']

    tokenizer_trainer = TokenizerTrainer()
    tokenizer_trainer.train_from_files(input_file_list, output_path=output_path, )
