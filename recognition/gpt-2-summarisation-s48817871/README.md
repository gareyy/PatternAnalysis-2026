# Layman Summarisation of BioLaySumm Medical Reports using GPT-2
> Author: Gareth Guce, 48817871

To allow for patient autonomy and reduction of healthcare anxiety, patients need to read and understand their own clinical radiology reports. This is hampered by the fact that these reports contain medical jargon and are too complex to understand by the layperson, preventing what we want. My solution here is a bespoke LLM that translates complex medical reports to reports readable by the average person.

## Algorithms/Models Used
### The Model
GPT-2 (Generative Pretrained Transformer 2) is a Large Language Model architecture based on the decoder part of the transformer architecture developed by OpenAI.

The decoder works by first decomposing the input (which is a sequence of both user input and model output so far) into tokens. These tokens are fed into the embedding feed forward network (FFN), and token positions are injected into these embeddings through positional encoding. These embeddings are then fed through a sequence of transformer blocks, which include a multi head self attention (MHSA) block, and a fully connected network. The multi head self attention block contains weight matrices that transform embeddings into query, key, and value matrices (note that in MHSA, there are usually multiple query, key, and value weight matrices, like channels in convolutions) through multiplying the input embeddings. The query matrices and key matrices are multiplied (and normalised through a division by the square root of the key dimension and softmax) to make a context vector representing similarities between all pairs of tokens. This multiplication makes it so that for each input token, it 'attends' to every other token in the sequence. This context vector is then multiplied with the value matrix to encode what should be the next token in the sequence. This is then fed through another fully connected neural network (FCNN) for further understanding before being passed to the next transformer block or the end. The end components of the GPT take the current output so far through another FCNN and through a softmax function, mapped to the set of all possible output tokens. These output tokens represent the next possible token in the sequence.

Transformers and multi head self attention have the advantage of deeper understanding of variable length sequences through every token attending to both itself and all other tokens in a sequence. This leads to greater semantic understanding of the input text. 

To prevent MHSA from having tokens attend to future tokens during training, attention masking is used, where a triangular matrix is applied to the product of the query and key matrices before the softmax function and multiplication with the value matrix.

The GPT architecture also uses other features such as dropout layers for preventing overfitting, layer norms for easier training and consistent behaviour, and skip connections for information preservation and better gradient flow for faster training convergence.

![GPT 1's Architecture, Image from Wikimedia Commons](https://upload.wikimedia.org/wikipedia/commons/9/91/Full_GPT_architecture.png)

The image above features GPT-1's architecture. GPT-2 improves upon GPT-1 through a larger parameter count and a larger training corpus, leading to emergent instruction following behaviour such as summarisation, language translation, and the ability to answer questions about a text.

GPT-2 comes in four sizes (according to their huggingface page), small (124M parameters), medium (355M parameters), large (774M parameters), and extra large (1.5B parameters)

## Fine Tuning
Fine tuning is the process of training an existing set of pretrained model weights on performing better in a specific task, such as coding, chatting, roleplay, translation, or summarisation. This can be done in multiple ways, like supervised fine tuning (where the model is trained to predict the next token for domain specific training data) and reinforcement learning (where the model is trained to maximise a reward, such as human feedback or outcomes of generated code). 

In this project, I did instruction tuning through supervised fine tuning, where trained the pretrained GPT-2 model on a specific input format that instructs it to summarise complex reports. This input format detailed the complex report, and the layperson summary. The model was trained to follow this input format and return a summarised version of the report.

## Low-Rank Adaptation
Low-Rank Adaptation (LoRA) is a finetuning method that freezes pretrained model weights and instead injects low rank matrices onto the weight matrix of the model, which greatly reduces the number of parameters to train. This parameter reduction leads to faster training times and less memory usage.

Say our weight matrix $W$ is $n \times m$, we want to add to it an $n \times m$ matrix $T$. The matrix we want to add is constructed from the multiplication of two matrices, $P$ and $K$. $P$ is $n \times r$ and $K$ is $r \times m$, where $r$ is our rank. $PK = T$, and we use $T+W$ in model inference. Training involves modifying the parameters in $P$ and $K$.

![Image Illustrating Lora, Drawn by me](./imagery/lora.png)

In this project, I investigated the viability of LoRA finetuning for medical report summarisation in GPT-2.

## Dataset
The dataset used in this project is the [BioLaySumm](https://huggingface.co/datasets/BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track) dataset, a collection of medical reports, their summarisations, and associated images. In this project, the images were not used for training or inference. The data came pre split into training rows (150k rows), validation rows (10k rows), and test rows (10.5k rows). 

The test rows came with no layperson summaries, and so for benchmarking in this project, I split the TODO

Splits:
- Training
- Validation
- Test

### Data Augmentation

## How to use and reproduce
The only major dependencies here is [uv](https://docs.astral.sh/uv/) and python 3.14. Running any script in this repository (i.e using `uv run train.py --help`) will immediately download and install all other dependencies, including but not limited to:
- pytorch
- huggingface
- huggingface transformers
- matplotlib
- peft
- huggingface datasets

This project consists of the following scripts:
- `modules.py`: 
- `dataset.py`: 
- `train.py`: 
- `predict.py`: 
- `generate_plots.py`: 

## Evaluation
In evaluating the outputs of the models I trained

## Training Runs

## Bibliography
- TODO: can use zotero -> create bibliography from item
- openAI gpt 2
- huggingface transformers, peft, dataset
- biolaysumm dataset
- https://towardsdatascience.com/meet-gpt-the-decoder-only-transformer-12f4a7918b36/

Putra, M. A. (2025, January 6). Meet GPT, The Decoder-Only Transformer. Towards Data Science. https://towardsdatascience.com/meet-gpt-the-decoder-only-transformer-12f4a7918b36/
