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

The test rows came with no layperson summaries, and so for benchmarking in this project, I split the validation set (which had both medical and layperson summaries) 80-20, where 80% is used in validation for hyperparameter tuning and 20% is held out in evaluation. The result is \~8k rows for validation, and \~2k rows held out for evaluation, The code for this can be found in `dataset.py`, and is done by offsetting the base index of `__getitem__`.

Splits:
- Training: Used for training the model, finetuning it for medical report summarisation for laypersons.
- Validation: Used for evaluation during training and approximating generalisation loss during training. This set is used particularly for hyperparameter tuning, and to ensure the model is capable of generalisation.
    - NOTE: In training, due to the long training times (Roughly \~2+ hours per epoch, due to the large dataset), only the first 300 (50 validation steps * batch size of 6) validation rows are used in validation loss calculation. Evaluating the whole validation set just for validation loss takes 2 minutes, and having explicitly 250 steps (spread throughout training, every $\lfloor \frac{n}{250} \rfloor$ steps, where n is the total number of training steps) for validation checking, leads to an additional 500 minutes of time spent in the training loop not training, or 8 hours. So only using 50 steps to do validation takes \~5 seconds per validation step, leading to an additional \~21 minutes in training rather than 8 hours.
- Test AKA Hold out: Used for evaluation in `predict.py`. To prevent me from overfitting the hyperparameters on the validation set during training, I set out this test split for evaluating finetuned models.

### Data Augmentation and Preprocessing
To make the model follow summarisation instructions, medical and layperson reports are put into instruction formats. These instruction formats make the model respond to summarisation requests, allow for separation of model outputs to medical and simplified outputs, and allow for inputting medical and layperson reports in the same sequence to the decoder-only transformer GPT model.

For the main training process, this format was used, with special tokens for instructions, the medical report part, and the summary part.

```
<inst>Summarise the following for a 6th grade layperson reading level in 1 to 2 sentences</inst>
<report>"{MEDICAL SUMMARY HERE}"</report>
<begin>"{LAYPERSON SUMMARY HERE}"
```

For evaluation of the base GPT-2 model, this format was used, where no special tokens from the above format were used as the model was not trained to use them.

```
Summarise the following for a 6th grade layperson reading level in 1 to 2 sentences:
"{MEDICAL SUMMARY HERE}" 
Summary:
"{LAYPERSON SUMMARY HERE}"
```

In development, this format was used before special token addition was implemented. This format was not used for any full training run, and was instead used as part of smoke tests so that I could make sure my own code was working.

```
{MEDICAL SUMMARY HERE} TLDR; {LAYPERSON SUMMARY HERE}
```

After putting the medical and layperson reports into a specific format, the pytorch dataset class for the BioLaySumm dataset tokenises the formatted data, and pads it to the max sequence length using the end-of-sentence token. This allowed it to be passed directly to a model's forward function after being sent to GPU memory.

## How to use and reproduce results
The only major dependencies here are [uv](https://docs.astral.sh/uv/), the huggingface cli, and python 3.14. Running any script in this repository (i.e using `uv run train.py --help`) will immediately download and install all other dependencies, including but not limited to:
- pytorch
- huggingface
- huggingface transformers
- matplotlib
- peft
- huggingface datasets
A more comprehensive list of dependencies can be found in `pyproject.toml`

Minimum system requirements (or rather, the specs I was able to fit this is) are 16GB+ of RAM, and a graphics card with 8GB+ of VRAM. Storage should be large enough to fit the dataset (\~29.3MB) and the pretrained model (\~548MB). This project and its training was done on Arch Linux, so anything else like Windows or MacOS is untested.

This project consists of the following scripts, files, and folders:
- `modules.py`: Holds input formats and the classes for them, functions for getting models (pretrained from huggingface or weights saved locally) and their corresponding tokeniser, functions for appending LoRA parameters to those models, functions for calculating cross entropy loss and rouge score for a given split of the dataset (through an inputted PyTorch DataLoader), and summary generation function for a given medical report prompt and its ground truth summary.
- `dataset.py`: Holds the `BioLaySumm` dataset class, which inherits from the PyTorch dataset class. It's an adaptor for the huggingface dataset that allows it to be augmented and used with my code.
- `train.py`: Does the model training through finetuning model on training dataset. Outputs a folder holding model weights and configurations that can be used in inference (i.e `predict.py`). Run `uv run train.py --help` to see command line arguments, which controls training hyperparameters. Options include the following:
```
--output-folder OUTPUT_FOLDER
                    Where to store output of training (default is a folder in the current working directory called "modelout")
--batch-size BATCH_SIZE
                    Batch size in training (default is 6)
--save-every SAVE_EVERY
                    Save every n steps, also does evaluation of metrics (val loss, train loss, ROUGE). If negative, it saves every max-steps/n (default is 250)
--max-steps MAX_STEPS
                    Max number of steps to train for (independent of epoch amount, takes priority over epoch amount), -1 to train for n epochs * number of batches in
                    training dataset (default is -1)
--n-epochs N_EPOCHS   Number of epochs to train for (default is 3)
--model-name MODEL_NAME
                    Huggingface model to train from (default is 'openai-community/gpt2')
--num-rows NUM_ROWS   Maximum number of rows to use in training dataset (default is None)
--max-eval-steps MAX_EVAL_STEPS
                    Maximum number of steps to evaluate for when calculating ROUGE and validation loss. -1 to evaluate for all batches in validation dataset (default is
                    50)
--lora                Train a LoRA instead of a finetuned model (disabled by default)
--input-format INPUT_FORMAT
                    Format for inputting report and summary to model. Options include 'TLDRFormat', 'InstructionFormat', 'SpecialFreeInstruction' (default is
                    InstructionFormat)
```
- `predict.py`: Runs evaluate suite on input model (input through the `--input-folder` parameter, such as a folder like `./modelout` for example). Does cross entropy loss and rouge score calculation on entire dataset holdout split, and does summarisation of example test cases using the model. All of these are printed out. Run `uv run predict.py --help` to see command line arguments, which help specify which model and generation parameters to use. Options include the following:
```
--input-folder INPUT_FOLDER
                    Where to load pretrained model from (default is a folder in the current working directory called "modelout")
--model-name MODEL_NAME
                    Name of base model, used for getting tokeniser (default is "openai-community/gpt2")
--max-eval-steps MAX_EVAL_STEPS
                    Maximum number of steps to evaluate for when calculating ROUGE and validation loss. -1 to evaluate for all batches in validation dataset (default is
                    -1)
--input-format INPUT_FORMAT
                    Format for inputting report and summary to model. Options include 'TLDRFormat', 'InstructionFormat', 'SpecialFreeInstruction' (default is
                    InstructionFormat)
```
- `generate_plots.py`: Code that takes `to_plot.json` files in each folder named `gpt2-*` and generates training and validation cross entropy loss plots over the training run. Do `uv run generate_plots.py` to generate each of these plots to the `imagery` folder. This script is mainly to help me write the README.md file.
- `README.md`: Project report, you are reading this right now.
- `imagery/`: Folder containing images used in this project.

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
