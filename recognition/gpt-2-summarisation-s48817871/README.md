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

### Synthetic Data Generation
Some training runs used synthetically generated data. This synthetic data was generated by a larger LLM (Qwen 3.6 35B A3B, a 4-bit quantisation, running on my laptop through llama.cpp). My script `generate_synthetic.py` iterated through the first 2500 rows of the BioLaySumm training data and used the medical reports from them to generate new data through rewording and summarisation. For a given medical report , the large LLM was first fed this prompt:
```
Please rewrite the following in roughly the same length. Write nothing else. \"{MEDICAL SUMMARY HERE}\"
```

And then this prompt:
```
Please rewrite and summarise the following, keep it roughly the same length, reword it to a 6th grade reading level, keep the summary third person. Write nothing else. \"{MEDICAL SUMMARY HERE}\"
```

This resulted in 2500 newly generated report-summary pairs. Some samples include:
TODO

My justification behind this is that a larger dataset of report-summary pairs will improve the model's performance in both ROUGE scores and manual inspection, as the model has a larger diversity of text to be finetuned on. With rewritten reports and summaries of existing data, the model will also be forced to deeply understand the relations between words in reports rather than just memorising preexisting summaries for given reports, helping with generalisation. What I did here is also technically some form of distillation (where a smaller model is trained on a larger model's outputs, or token predictions), and the usage of this generated data from a smarter model will theoretically improve the performance of the small GPT-2 model.

## How to use and reproduce results
The only major dependencies here are [uv](https://docs.astral.sh/uv/), the huggingface cli, and python 3.14. Running any script in this repository (i.e using `uv run train.py --help`) will immediately download and install all other dependencies, including but not limited to:
- pytorch
- huggingface
- huggingface transformers
- matplotlib
- peft
- huggingface datasets
- openai API python library
A more comprehensive list of dependencies can be found in `pyproject.toml`

Minimum system requirements (or rather, the specs I was able to fit this is) are 16GB+ of RAM, and a graphics card with 8GB+ of VRAM. Storage should be large enough to fit the dataset (\~29.3MB) and the pretrained model (\~548MB). This project and its training was done on Arch Linux, so anything else like Windows or MacOS is untested.

This project consists of the following scripts, files, and folders:
- `modules.py`: Holds input formats and the classes for them, functions for getting models (pretrained from huggingface or weights saved locally) and their corresponding tokeniser, functions for appending LoRA parameters to those models, functions for calculating cross entropy loss and rouge score for a given split of the dataset (through an inputted PyTorch DataLoader), and summary generation function for a given medical report prompt and its ground truth summary.
- `dataset.py`: Holds the `BioLaySumm` dataset class, which inherits from the PyTorch dataset class. It's an adaptor for the huggingface dataset that allows it to be augmented and used with my code. It also holds the `BLSSynthetic` dataset class, which concatenates both the original BioLaySumm training data with synthetically generated data.
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
--synthetic           Enable usage of synthetic data during training
--synthetic-only      Only use synthetic data in training. Only works if synthetic data is being used
--synthetic-csv SYNTHETIC_CSV
                    CSV file of synthetic data (medical report and summary pairs). Only works if synthetic data is being used
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
- `generate_synthetic.py`: Uses an LLM from an OpenAI API compatible endpoint to iterate through the BioLaySumm dataset, generating synthetic data through reworded reports and new summarisations. Command line options (from `uv run generate_synthetic.py --help`) include the following:
```
--url URL             URL of OpenAI API compatible endpoint (Default is a local llama.cpp endpoint)
--model MODEL         Model from API that should be used (Default is a specialied Qwen model a local llama.cpp endpoint)
--key KEY             API key for OpenAI API compatible endpoint (Default is "key")
--limit-rows LIMIT_ROWS
                    Number of rows from BioLaySumm dataset to be rewritten. -1 for all rows.
--output-csv OUTPUT_CSV
                    Output CSV of rewritten data (Default is "synthetic.csv")
```
- `README.md`: Project report, you are reading this right now.
- `imagery/`: Folder containing images used in this project.

## Evaluation
In evaluating the outputs of the models I trained on the holdout split, I used the following metric scores:
| Metric Name | Description |
| -------------- | --------------- |
| Cross Entropy Loss | Loss used in training, measuring how well the model can predict the next token in a sequence of expected tokens. |
| ROUGE-1 | Measures ratio of unigrams (single words) that appear in both summaries. This is usually an F1 score, consisting of precision (number of unigrams in the predicted summary that also appear in the original summary) and recall (number of unigrams in the original summary that appear in the predicted summary). |
| ROUGE-2 | Like ROUGE-1 but instead of being an F1 ratio of unigrams that appear in both summaries, its an F1 ratio of bigrams (pairs of consecutive words) that appear in both summaries. |
| ROUGE-L | Longest Common Subsequence. Calculated through the number of words in the predicted summary that are in the same order as in the original summary, divided by the number of words in the original summary. |
| ROUGE-L-Sum | Still uses the longest common subsequence, but looks at the summaries sentence wise (i.e through full stops or newlines). Calculated by splitting both summaries into their sentences, then calculating ROUGE-L on each sentence pair. Then returns the average of the ROUGE-L scores. |

The problem with ROUGE scores is that they only consider the words and text itself, and not the deeper semantic meaning. ROUGE scores only look at the intersection of words common in both predicted and original summaries, and do not look at semantics. For example, the phrase `The newborn cat is overweight` can be summarised to `The kitten is fat`, and would have a low ROUGE score as the calculations for it do not take into account the semantic similarities between `overweight` and `fat`, or `newborn cat` and `kitten`. Because of its ability to not be able to understand semantics, it cannot be used reliably to measure for factuality, understandability, or hallucinations. For example, the phrase `You are diagnosed with cancer` can be summarised correctly into `You do have cancer` or incorrectly into `You don't have cancer`, and both summaries would have equal or similar ROUGE scores. Because patients need factual yet understandable information for their own autonomy and clinical safety, ROUGE as a metric is simply not enough to evaluate summary quality.

Because of the unreliability of ROUGE scores for this task, I also did manual evaluation on these following report-summary pairs, that I believed represent the entire dataset. I manually inspected the predicted output of these to evaluate unquantifiable metrics like hallucination rate, faithfulness to original medical report, removal of redundant findings, readability by a person at the 6th grade reading level, and coherence.

### REPORT 1
REPORT: 
> Confirmed COVID-19. No pulmonary consolidations are observed. No pleural effusion. Conclusion: Examination within normal limits.

SUMMARY: 
> The patient has tested positive for COVID-19. However, there are no signs of pneumonia in the lungs, and no fluid has built up around the lungs. The conclusion is that the examination results are normal.

This report has multiple things that need to be translated directly from medical jargon to understandable text, such as 'pulmonary consolidations' to 'liquid in lungs', 'Confirmed COVID-19' to 'tested positive for COVID-19', and 'pleural effusion' to 'liquid around the lungs'. This ground truth summary also gets the implication that no fluid in lungs means no signs of pneumonia. If models were to get this correctly in summaries, it found hint at a possible ability to reason some medical conclusions. This report and its summary also has some redundant information in the form of 'examination is normal'.
### REPORT 2
REPORT: 
> No significant findings.

SUMMARY: 
> There are no important issues found.

I used this report as a small portion of the rows in all of the splits report the absence of anything significant. This prompt also has the possibility of demonstrating the diversity of outputs for the same input.
### REPORT 3
REPORT: 
> Comparison with the previous study dated 06 08 2012 shows the appearance of a right basal pneumothorax. Subcutaneous emphysema in the right thoracic wall. Radiological worsening of diffuse bilateral alveolar involvement.

SUMMARY: 
> "Compared to the study from June 8, 2012, there is now a condition called a right basal pneumothorax, which means air has leaked into the space between the lungs and chest wall on the right side. There is also air trapped under the skin of the right chest wall. The overall picture of the lungs has gotten worse, showing more involvement of both lungs.

This prompt has a date in it, and it can be used to determine how well a model handles dates. This prompt also shows the need for understandability, as a `right basal pneumothorax` would be indecipherable to a layperson reading this, and the provided summary demonstrates that the model should be able to explain what it is.
### REPORT 4
REPORT: 
> Signs of COPD. Images are compatible with a healed fracture of the posterior arch of the 10th right rib.

SUMMARY:
> Signs of chronic obstructive pulmonary disease (COPD) are present. The images also show that there was a fracture in the back part of the 10th rib on the right side, which has now healed.

This prompt and summary pair demonstrates that models should be able to handle medical abbrieviations such as COPD, as some items in the dataset would have existing medical abbrieviations. This summary also directly refers to the image that this report is typically paired with, showing that the model should be able to handle reports with external references without hallucinating.
### REPORT 5
REPORT: 
> Chronic changes in the lung parenchyma. Biapical pleural thickening. Calcified aortic atheromatosis. Pseudonodular image in the left base likely related to the nipple. Hiatal hernia. Dorsal lumbar scoliosis.

SUMMARY:
> The lungs show long-term changes. Thickening is seen at the top of both sides of the lung coverings. There is a buildup of calcium in the fatty deposits of the main blood vessel in the abdomen. A small bump-like image on the left side is probably related to the nipple. A hiatal hernia, where the stomach bulges into the chest, is present. The lower back shows a sideways curve in the spine.

This report has a lot of body-specific language, referring to such places as lungs, the abdomen, the nipples, the chest, and the spine. Reports that refer to the body itself are common within the dataset, and the model should handle summarising these factually.
### REPORT 6
REPORT: 
> Right parahilar infiltrate and atelectasis. Increased retrocardiac density related to atelectasis and consolidation associated with right pleural effusion. Clinical data is important for correct radiological assessment.

SUMMARY: 
> There is a cloudiness near the right lung's airways and a part of the lung has collapsed. The area behind the heart is denser, which could be due to the collapsed lung and a possible lung infection along with fluid around the right lung. It's important to consider the patient's medical history for a proper understanding of the x-ray.

This report demonstrates the importance of translating medical jargon to layperson understandable information, as its corresponding summary makes sure to explain the underlying meaning behind each medical condition instead of purely just reducing the word count. This report and summary pair also has redundant information stating how data is important for radiological assessment.
## Training Runs

TODO

## Bibliography
- TODO: can use zotero -> create bibliography from item
- openAI gpt 2
- openAI API
- asyncio 
- huggingface transformers, peft, dataset
- biolaysumm dataset
- https://towardsdatascience.com/meet-gpt-the-decoder-only-transformer-12f4a7918b36/

Putra, M. A. (2025, January 6). Meet GPT, The Decoder-Only Transformer. Towards Data Science. https://towardsdatascience.com/meet-gpt-the-decoder-only-transformer-12f4a7918b36/

## Appendix A: Feasibility Report
I did my feasibility check off on October 16th. Some things wrote here may be outdated, or written to a less higher standard to the rest of this markdown file.

### Chosen Problem, Chosen Model, Dataset
- Chosen Problem: Clinical Language Models, Summarisation of complex clinical radiology reports into layperson accessible language
- Chosen Model: GPT-2 Medium (Hard Difficulty)
    - I believe this type of model would be the best for paragraph summarisation and simplification as GPT style models have proven their worth in many other fields, such as chat and thinking. GPT architecture for text summarisation has also been a proven application for a long while before ChatGPT (i.e Grammarly). 
    - The self attention focus of transformers lends well to text summarisation, as the relationship between every token is interrogated and used in prediction of the text token in a summary of an input.
    - What is notable is that the GPT family model is a decoder only transformer architecture, which means the inputs (medical paragraphs to be summarised) must share the same space as the output. This means I have to structure input and output data in some sort of context usable by a decoder only architecture (like a chat or json interface).
- Datasets
    - BioLaySumm LaymanRRG dataset
    - Possible usage of synthesised datasets (generated summarisations of existing biomedical paragraphs using stronger and newer models)

### User Need
- Patients need to read and understand their own clinical radiology reports to allow for patient autonomy and reduction of healthcare anxiety. Unfortunately, the complex nature and usage of medical language would lead patients being unable to understand their own condition, leading them to have less autonomy and more anxiety.

### Scope
- Comparison of full finetune of weights vs LoRA
- Possible extension of dataset through synthetic dataset generation (through using a smarter LLM to reword reports and summarise them to create new datapoints)
- Limit to only using GPT-2 Medium (360M)
- Use AdamW + Cosine Annealing LR
    - or Muon if needed?
- Learning rate should be something low as we are finetuning, such as `5e-5` (as suggested by https://huggingface.co/learn/llm-course/en/chapter3/4)
- Comparison of how data is formatted
    - i.e in the following formats
    ```
    Mild basal infiltration on the right.
    TLDR;
    A mild amount of fluid is present at the bottom of the right lung.
    ```

    ```
    Mild basal infiltration on the right.
    <|TLDR|>
    A mild amount of fluid is present at the bottom of the right lung.
    ```

    ```
    Mild basal infiltration on the right.
    Or in layman's terms,
    A mild amount of fluid is present at the bottom of the right lung.
    ```

    ```
    <|RADIOLOGY_REPORT|>
    Mild basal infiltration on the right.
    <|RADIOLOGY_REPORT_END|>
    <|LAYMAN_REPORT|>
    A mild amount of fluid is present at the bottom of the right lung.
    <|LAYMAN_REPORT_BEGIN|>
    ```

### Acceptance Criteria
- Should have high scores on metrics like ROUGE-L, ROUGE-Lsum, ROUGE-1, ROUGE-2
    - These metrics evaluate similarity between an example summarisation and output summarisation, however should be balanced with other metrics as it does not account for factualness and hallucinations.
- Criterions evaluated manually (by taking 5+ test cases and evaluating them by hand)
    - (Little to) no hallucinations. High faithfulness to the original text
    - Retain as many facts as needed (remove unnecessary or redundant findings)
    - Should be readable by a layperson (a 6th grade reading level)
    - Should be coherent and read like real language
    - All these metrics are things that ROUGE does not cover, and so will need evaluation by a person.
- Must run on a gaming laptop with 16GB of RAM, a 4060 with 8GB VRAM, and a Ryzen 7 7735HS
    - This is my personal gaming laptop that I have converted for use in my ML work, allowing me to quickly iterate on this project without waiting for or paying for compute
- Should have a reasonable token input/output speed
    - Users would not be happy if the model takes 10+ minutes to summarise a report
- ~~Should take a reasonable amount of time to train, such as 20 minutes per epoch~~
    - I only have a limited time to work on this project, the computer im using this for has other responsibilities (other assignments, gaming), and I am very inpatient
    - **FUTURE NOTE**: I initially made this criteria wrongly using the `karthiksagarn/astro_horoscope` dataset, which was way smaller than the larger BioLaySumm dataset. 

### Risks
- Finetuned model/LoRA may not converge to a low enough training loss, leading to low ROUGE scores and incoherentness
- Finetuned model/LoRA may overfit, i.e have high validation loss, low ROUGE metrics, incoherent or incorrect output, but low training loss
- I may waste time trying to tune one or more hyperparameter, i.e learning rate, epoch size, optimiser, etc
- Getting banned by Anthropic/OpenAI for distillation (I will wear it proudly like a badge) in synthetic data generation
- Any model (original weights, LoRA, or finetuned) having repeating outputs and no termination condition

### Preliminary Work Done & Further Work
- Have Done
    - Wrote and tested various GPT-2 sizes in both inference and training
        - used https://huggingface.co/learn/llm-course/en/chapter3/3 as inspiration
        - found that small (135M) and medium (360M) are trainable on my system
        - Used Trainer API
    - Did finetuning on a different dataset (`karthiksagarn/astro_horoscope`) in evaluation of model sizes
    - Looked at dataset
        - Weirdly, the test split does not have layman reports, I will need to evaluate it manually, and with no ROUGE score
    - Created dataset class for pytorch
- To do soon
    - Write full training loop without training API
    - Update dataset class for tokenisation and formatting
### Fallback
- Fallback Problem: Bounding Box detection of Lesions
- Chosen Model: YOLOv7 (Medium Difficulty)
    - Most popular bounding box detection architecture
- Dataset: ISIC Data
