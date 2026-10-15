# Feasibility Check Off

## Chosen Problem, Chosen Model, Dataset
- Chosen Problem: Clinical Language Models, Summarisation of complex clinical radiology reports into layperson accessible language
- Chosen Model: GPT-2 Medium (Hard Difficulty)
    - I believe this type of model would be the best for paragraph summarisation and simplification as GPT style models have proven their worth in many other fields, such as chat and thinking. GPT architecture for text summarisation has also been a proven application for a long while before ChatGPT (i.e Grammarly). 
    - The self attention focus of transformers lends well to text summarisation, as the relationship between every token is interrogated and used in prediction of the text token in a summary of an input.
    - What is notable is that the GPT family model is a decoder only transformer architecture, which means the inputs (medical paragraphs to be summarised) must share the same space as the output. This means I have to structure input and output data in some sort of context usable by a decoder only architecture (like a chat or json interface).
- Datasets
    - BioLaySumm LaymanRRG dataset
    - Possible usage of synthesised datasets (generated summarisations of existing biomedical paragraphs using stronger and newer models)

## User Need
- Patients need to read and understand their own clinical radiology reports to allow for patient autonomy and reduction of healthcare anxiety. Unfortunately, the complex nature and usage of medical language would lead patients being unable to understand their own condition, leading them to have less autonomy and more anxiety.

## Scope
- Comparison of full finetune of weights vs LoRA
- Possible extension of dataset through synthetic dataset generation (through using a smarter LLM to reword reports and summarise them to create new datapoints)
- Limit to only using GPT-2 Medium (360M)
- Use AdamW + Cosine Annealing LR
    - or Muon if needed?
- Learning rate should be something low as we are finetuning, such as `5e-5` (as suggested by https://huggingface.co/learn/llm-course/en/chapter3/4)

## Acceptance Criteria
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
- Should take a reasonable amount of time to train, such as 20 minutes per epoch
    - I only have a limited time to work on this project, the computer im using this for has other responsibilities (other assignments, gaming), and I am very inpatient

## Risks
- Finetuned model/LoRA may not converge to a low enough training loss, leading to low ROUGE scores and incoherentness
- Finetuned model/LoRA may overfit, i.e have high validation loss, low ROUGE metrics, incoherent or incorrect output, but low training loss
- 

## Preliminary Work Done & Further Work
- TODO: data audit, smoke test experiment, measured resource estimate
- TODO: detail what i'll do next

## Fallback
- Fallback Problem: Bounding Box detection of Lesions
- Chosen Model: YOLOv7 (Medium Difficulty)
    - Most popular bounding box detection architecture
- Dataset: ISIC Data

1. User Need, Scope & Acceptance Criteria: The intended user, prototype purpose, and 3–5 testable acceptance criteria addressing the domain’s open dilemma, including an operational/resource constraint (e.g., peak VRAM, runtime).
2. Model Choice & Course Concepts: The proposed dataset configuration, model architecture, chosen difficulty band, and how course concepts and literature support the choices.
3. Preliminary Feasibility Evidence: An initial data audit, baseline smoke-test experiment, or measured resource estimate. A fully trained final model is not required.
4. Risks, Budget & Fallback: Key technical risks, computing budget, the next planned experiment, and a workable fallback.
