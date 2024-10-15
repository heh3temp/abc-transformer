import argparse

import torch
import muspy

from modules.tokenizers import CharTokenizer
from modules.models import Transformer
from modules.configs import CharTokenizerConfig, TransformerConfig
from modules.generation.strategies import TopKSampling, GreedySearch, AncestralSampling
from modules.utils import get_device


RUN_PATH = "assets/"
OUT_PATH = "tmp/"


def main():

    parser = argparse.ArgumentParser(
        description="generate melody using trained model"
    )

    parser.add_argument("-s", "--source", default=RUN_PATH, help="path to directory containing model configs and checkpoint")
    parser.add_argument("-o", "--out", default=OUT_PATH, help="path to directory in which the tune will be saved")
    args = parser.parse_args()
    
    tokenizer_config = CharTokenizerConfig.from_yaml(args.source + "tokenizer_config.yaml")
    tokenizer = CharTokenizer(tokenizer_config)
    
    transformer_config = TransformerConfig.from_yaml(args.source + "model_config.yaml")
    model = Transformer(transformer_config)

    device = get_device()

    model.load_checkpoint(args.source + "checkpoints/final.pt", device=device)
    model = model.to(device)

    strategy = TopKSampling(temperature=1.0, k=8)

    context = torch.tensor([tokenizer.EOS]).to(device)
    generated = model.generate(context, max_len=10000, EOS=tokenizer.EOS, strategy=strategy)
    abc_tune = tokenizer.decode(generated[1:])
    print(abc_tune)

    audio_tune = muspy.read_abc_string(abc_tune)

    muspy.write_audio(args.out + "tune.wav", audio_tune, audio_format="wav", gain=1.5)
    with open(args.out + "tune.abc", "w") as f:
        f.write(abc_tune)

if __name__ == "__main__":
    main()