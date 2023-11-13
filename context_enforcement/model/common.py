import argparse
import os

from transformers import TrainingArguments


def create_training_args(parser=None):
    parser = argparse.ArgumentParser() if parser is None else parser

    parser.add_argument(
        "--tokenizer-path",
        "-tokenizer-path",
        required=True,
        help="The location of the trained tokenizer",
    )
    parser.add_argument(
        "--model-base",
        "-mb",
        required=True,
        help="The type of transformer architecture",
    )
    parser.add_argument(
        "--output-dir",
        "-output-dir",
        default="trained_context_enforcers/",
        help="Location of where the trained models is saved",
    )
    parser.add_argument(
        "--task-type",
        required=False,
        help="Type of task: XSum, CNN-SUM",
    )
    parser.add_argument(
        "--run-id",
        "-run-id",
        type=str,
        default="",
        help="Id for the running"
    )
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", "-lr",
                        type=float, default=5e-5)

    parser.add_argument("--max-seq-len", default=1024, type=int)

    parser.add_argument('--evaluation-strategy', default='epoch')
    parser.add_argument('--save-strategy', default='epoch')

    parser.add_argument("--seed", default=10, type=int, help="Random seed")
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--weight-decay",
                        type=float, default=0.0)
    parser.add_argument("--warmup-ratio",
                        required=False,
                        type=float, default=0.0)
    parser.add_argument("--num-train-epochs", type=int, default=10)

    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument(
        "--per-device-train-batch-size",
        "-per-device-tbz",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        "-per-device-ebz",
        type=int,
        default=16,
    )
    # gradient_accumulation_steps
    parser.add_argument('--gradient-accumulation-steps',
                        type=int, default=1,
                        help="Gradient accumulation steps"
                        )
    parser.add_argument('--fp16', '-fp16', action="store_true")
    parser.add_argument("--verbose", "-verbose", action="store_true")
    parser.add_argument('--logging-dir',
                        default='../logging-dir',
                        required=False
                        )
    # warmup_ratio save_total_limit per_device_eval_batch_size

    return parser


def get_training_arguments(
        output_dir,
        num_train_epochs,
        learning_rate,
        lr_scheduler_type,
        warmup_ratio,
        weight_decay,

        save_total_limit,
        save_strategy,
        evaluation_strategy,
        eval_steps,
        per_device_train_batch_size,
        logging_dir,
        warmup_steps=0,
        greater_is_better=False,
        remove_unused_columns=False,
        verbose=False,
        load_best_model_at_end: bool = True,
        gradient_accumulation_steps=1,
        fp16=True,
        metric_for_best_model='eval_loss',
        **unused_args,
):

    return TrainingArguments(
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=fp16,
        overwrite_output_dir=True,
        adafactor=False,
        load_best_model_at_end=load_best_model_at_end,
        greater_is_better=greater_is_better,
        remove_unused_columns=remove_unused_columns,
        output_dir=output_dir,
        evaluation_strategy=evaluation_strategy,  # "epoch",
        save_strategy=save_strategy,  # 'epoch',
        lr_scheduler_type=lr_scheduler_type,
        learning_rate=learning_rate,
        save_total_limit=save_total_limit,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        warmup_steps=warmup_steps,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        disable_tqdm=not verbose,
        eval_steps=eval_steps,
        save_steps=eval_steps,
        logging_dir=logging_dir,
        metric_for_best_model=metric_for_best_model,
        **unused_args
    )
