from datetime import datetime
import os

import wandb
import torch
from torch.utils.data.dataloader import DataLoader

from modules.datasets import ABCDataset, TransformerTrainDataset
from modules.tokenizers import CharTokenizer
from modules.models import Transformer
from modules.trainers import Trainer
from modules.callbacks import SaveCheckpointCallback, WandbLoggingCallback
from modules.configs import TrainerConfig, CharTokenizerConfig, TransformerConfig
from modules.utils import get_device


RUNS_PATH = "./runs"

IS_WANDB_ON = True


def get_run_id():
    return f"{datetime.now().isoformat(sep='-', timespec='seconds')}".replace(":", "-")


def main():

    if IS_WANDB_ON:
        wandb.login()

    run_id = get_run_id()
    run_path = RUNS_PATH + "/" + run_id + "/"
    os.mkdir(run_path)

    ckpt_path = run_path + "checkpoints/"
    os.mkdir(ckpt_path)
    
    # dataset = ABCDataset('./data/processed/merged.abc')
    dataset = ABCDataset("data/processed/notthingam_database.abc")
    corpus = dataset.corpus
    
    tokenizer_config = CharTokenizerConfig.from_yaml("config/tokenizer_config.yaml")
    tokenizer = CharTokenizer.from_corpus(corpus, tokenizer_config.SOS, tokenizer_config.EOS)
    
    trainer_config = TrainerConfig.from_yaml("config/trainer_config.yaml")

    transformer_config = TransformerConfig.from_yaml("config/model_config.yaml")
    transformer_config.vocab_size = tokenizer.vocab_size


    tokenized_train = tokenizer.encode(corpus)
    torch_train_dataset = TransformerTrainDataset(tokenized_train, transformer_config.context_size)
    train_dataloader = DataLoader(torch_train_dataset, batch_size=trainer_config.batch_size, shuffle=True, num_workers=trainer_config.num_workers)

    device = get_device()

    model = Transformer(transformer_config)
    callbacks = [SaveCheckpointCallback(ckpt_path=ckpt_path, period_epochs=1)]
    trainer = Trainer(trainer_config, callbacks=callbacks, device=device)

    tokenizer.config.to_yaml(run_path + "tokenizer_config.yaml")
    trainer.config.to_yaml(run_path + "trainer_config.yaml")
    model.config.to_yaml(run_path + "model_config.yaml")

    wandb_config = {
        "tokenizer": tokenizer.config.to_dict(),
        "model": model.config.to_dict(),
        "trainer": trainer.config.to_dict()
    }
    if IS_WANDB_ON:
        with wandb.init(project="ABCTransformer", config=wandb_config):
            trainer.add_callback(WandbLoggingCallback())
            trainer.fit(model, train_dataloader)
    else:
        trainer.fit(model, train_dataloader)


if __name__ == "__main__":
    main()