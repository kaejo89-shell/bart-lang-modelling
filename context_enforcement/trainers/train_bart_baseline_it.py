import os
import comet_ml
import transformers
from basics.logging import get_logger
from datasets import load_dataset
from transformers import BartTokenizer
from context_enforcement.data.common import ModelPreTrainingTask
from context_enforcement.model.bart_baseline import create_bart_baseline
from context_enforcement.model.common import get_training_arguments, create_training_args

module_logger = get_logger(os.path.basename(__file__))

comet_ml.init(project_name='bart-contextual-base')
# 1. Enable logging of model checkpoints
os.environ["COMET_LOG_ASSETS"] = "True"

if __name__ == '__main__':
    parser = create_training_args()
    args = parser.parse_args()
    run_config = vars(args)

    tokenizer_path = run_config['tokenizer_path']
    tokenizer = BartTokenizer.from_pretrained(tokenizer_path)
    vocab_size = tokenizer.vocab_size
    max_seq_len = run_config['max_seq_len']

    model_pre_training_task = ModelPreTrainingTask(tokenizer=tokenizer, max_seq_len=max_seq_len)

    # Base version of Bart
    model = create_bart_baseline(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    train_streaming_dataset = load_dataset(
        "gsarti/clean_mc4_it",
        "full", split="train", streaming=True
    ).with_format(type="torch")
    eval_streaming_dataset = load_dataset(
        "gsarti/clean_mc4_it",
        "full",
        split="validation",
        streaming=True
    ).with_format(type="torch")

    # perturbation in string: document_rotation, sentence_permutation
    # perturbation in token : token_infilling, token_masking, token_deletion

    # total_steps (1 epoch, see it5) = 103_000_000 / 64 = 1_609_375 -- 1_700_000
    # warmup_steps = 1_700_000 * 0.01 = 17_000

    # Prepare training arguments
    output_dir = os.path.join(run_config['output_dir'], run_config['run_id'])
    training_args = get_training_arguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        warmup_steps=17_000,
        weight_decay=0.01,
        warmup_ratio=0.0,
        save_strategy="steps",
        evaluation_strategy="steps",
        max_steps=1_700_000,
        logging_dir=run_config['logging_dir'],
        logging_steps=100,
        eval_steps=10000,
        save_steps=10000,
        save_total_limit=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        fp16=True,
        dataloader_num_workers=24,
        learning_rate=1e-4,
        lr_scheduler_type="linear",

    )

    # Initialize the trainer
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_streaming_dataset,
        eval_dataset=eval_streaming_dataset,
        data_collator=model_pre_training_task.collate_fn,
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    module_logger.info(trainer.evaluate(eval_streaming_dataset))

    # Save the model
    trainer.save_model(output_dir)
    
