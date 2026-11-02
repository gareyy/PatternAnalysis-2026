# Layman Summarisation of BioLaySumm Medical Reports using GPT-2
> Author: Gareth Guce, 48817871

To allow for patient autonomy and reduction of healthcare anxiety, patients need to read and understand their own clinical radiology reports. This is hampered by the fact that these reports contain medical jargon and are too complex to understand by the layperson, preventing what we want. My solution here is a bespoke LLM that translates complex medical reports to reports readable by the average person.

## Algorithms/Models Used
### The Model
GPT-2 (Generative Pretrained Transformer 2) is a Large Language Model architecture based on the decoder part of transformer architecture developed by OpenAI.

GPT-2 has these following architectural components.
- Decoder only transformer architecture, where inputs are placed alongside model outputs when being fed to the model.
- Embedding of input tokens to semantic representative vectors through a fully connected neural network.
- Positional encoding of input sematic vectors.
- Transformer Blocks, consisting of
    - Multi Head Self Attention. A Masked Query vector and key matrix are multiplied together, and the result is multiplied by the Value matrix. This allows for deep understanding of input data.
    - Fully connected neural network, for further processing of attention outputs.
- Token prediction, where the next token in the sequence is predicted from the processed inputs (processed through transformer blocks and feed forward networks).

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

## Bibliography
- TODO: can use zotero -> create bibliography from item
- openAI gpt 2
- huggingface transformers,
- https://towardsdatascience.com/meet-gpt-the-decoder-only-transformer-12f4a7918b36/

Putra, M. A. (2025, January 6). Meet GPT, The Decoder-Only Transformer. Towards Data Science. https://towardsdatascience.com/meet-gpt-the-decoder-only-transformer-12f4a7918b36/
