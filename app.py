import os
import glob

import torch
import muspy
import gradio as gr

from modules.tokenizers import CharTokenizer
from modules.models import Transformer, AutoregressiveModel
from modules.configs import CharTokenizerConfig, TransformerConfig
from modules.generation.strategies import TopKSampling, Strategy
from modules.utils import get_device


ASSETS_PATH = "assets/"
MODEL_CKPT_PATH = ASSETS_PATH + "checkpoints/final.pt"
WAVE_PATH = "tmp/tune.vaw"



class Composer:
    
    def __init__(self, tokenizer: CharTokenizer, model: AutoregressiveModel, strategy: Strategy, device: torch.device) -> None:
        self.device = device
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.model = model

    def generate_tune(self) -> str:

        is_valid = False
        while not is_valid:
            try:
                context = torch.tensor([self.tokenizer.EOS]).to(self.device)
                generated = self.model.generate(context, max_len=10000, EOS=self.tokenizer.EOS, strategy=self.strategy)
                abc_tune = self.tokenizer.decode(generated[1:])
                audio_tune = muspy.read_abc_string(abc_tune)

            except ValueError:
                pass

            else:
                is_valid = True

        muspy.write_audio(WAVE_PATH, audio_tune, audio_format="wav", gain=1.5)
        
        return abc_tune, WAVE_PATH


def main():

    muspy.download_musescore_soundfont()

    device = get_device()
    tokenizer_config = CharTokenizerConfig.from_yaml(ASSETS_PATH + "tokenizer_config.yaml")
    tokenizer = CharTokenizer(tokenizer_config)

    transformer_config = TransformerConfig.from_yaml(ASSETS_PATH + "model_config.yaml")
    model = Transformer(transformer_config).to(device)
    model.load_checkpoint(MODEL_CKPT_PATH, device=device)

    strategy = TopKSampling(temperature=1.0, k=8)

    backed = Composer(tokenizer, model, strategy, device)

    interface = gr.Interface(
        fn=backed.generate_tune,
        inputs=None,
        outputs=["text", "audio"],
        theme=gr.themes.Soft(),
        title="AI Composer",
        description="This web application can be used to compose tunes using Transformer based neural network",
    )

    try:
        interface.launch(server_name="localhost", server_port=7860)

    except KeyboardInterrupt:
        interface.close()

    except Exception as e:
        print(e)
        interface.close()


if __name__ == "__main__":
    main()