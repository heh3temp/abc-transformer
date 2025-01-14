# ABCTransformer

This library enables training decoder-only transformer models to generate music in ABC notation. It also includes scripts for running inference and a simple web application interface.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Examples](#examples)
- [Setup](#setup)
- [Usage](#usage)
- [License](#license)


## Introduction
ABCTransformer is designed to facilitate the training of transformer-based models for music generation. It supports music composition in ABC notation, providing tools for model training, inference, and a web-based interface for interaction.

## Features
- Training transformer models for music generation in ABC notation
- Running inference to generate music samples in ABC as well as synthesized WAV files
- Web application interface developed with Gradio allows for easy interaction with the model


## Examples
Here are some examples of music generated by the model trained on The Nottingham Music Database:


https://github.com/user-attachments/assets/b560000d-5065-48c8-a82a-a15f596ae672


https://github.com/user-attachments/assets/6c3a7aa1-c99d-4c12-ac46-42928660489e


https://github.com/user-attachments/assets/ff856e9c-fc46-4e06-9e8c-4e8a18973cd8


https://github.com/user-attachments/assets/febfcf4e-039a-45bb-b530-1f2b0a3d7a03


https://github.com/user-attachments/assets/fb534211-b8c3-413a-a08e-d43e5aed44ad


https://github.com/user-attachments/assets/9342cd60-3c65-4396-8a77-4600ac6913d8



## Setup

1. **Install Python 3.10**
   Ensure you have Python 3.10 installed. You can use `pyenv` for managing Python versions.
   ```
   pyenv install 3.10.0
   pyenv local 3.10.0
   ```

2. **Install Fluidsynth**
   Fluidsynth is required for audio synthesis.
   ```
   sudo apt install fluidsynth
   ```

3. **Install Requirements**
   Install the necessary Python packages.
   ```
   pip install -r requirements.txt
   ```


## Usage

### Data preprocessing
If you want to use as custom dataset in ABC format you need to preprocess it using `preprocess_datasets.py` script.

### Configs
Model, Tokenizer and Trainer config files are located inside the `config` directory

### Training the Model
To train the model, run:
```
python train.py
```

Run logs and checkpoints are saved inside the `runs` directory

### Autoregressive Decoding Stratedies
If you want to experiment with different decoding strategies and parameters you must modify this line:
```
strategy = TopKSampling(temperature=1.0, k=8)
```
 inside the `generate.py script`. Currently implemented strategies are: `TopKSampling`, `GreedySearch` and `AncestralSampling`


### Generating Samples
To generate a sample, run:
```
python generate.py
```

### Launching the Web Application
To launch the web application, run:
```
python app.py
```

<p align="center">
  <img src="assets/web_app.gif" alt="" width="100%">
</p>


## License
This project is licensed under the **GNU General Public License v3.0**. See the [LICENSE](LICENSE) file for details.

---
