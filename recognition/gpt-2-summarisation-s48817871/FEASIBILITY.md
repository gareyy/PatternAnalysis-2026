# Feasibility Check Off

## Chosen Problem, Chosen Model, Dataset
- Chosen Problem: Clinical Language Models, Summarisation of complex clinical radiology reports into layperson accessible language
- Chosen Model: GPT-2 (Hard Difficulty)
    - I believe this type of model would be the best for paragraph summarisation and simplification as GPT style models have proven their worth in many other fields, such as chat and thinking. GPT architecture for text summarisation has also been a proven application for a long while before ChatGPT (i.e Grammarly). 
    - The self attention focus of transformers lends well to text summarisation, as the relationship between every token is interrogated and used in prediction of the text token in a summary of an input.
    - What is notable is that the GPT family model is a decoder only transformer architecture, which means the inputs (medical paragraphs to be summarised) must share the same space as the output. This means I have to structure input and output data in some sort of context usable by a decoder only architecture (like a chat or json interface).
- Datasets
    - BioLaySumm LaymanRRG dataset
    - Possible usage of synthesised datasets (generated summarisations of existing biomedical paragraphs using stronger and newer models)
    - Possible extra biomedical paragraph data for additional medical intelligence

## Acceptance Criteria
- TODO: also add operational/resource cosntraints (VRAM, runtime)

- Resource Constraints
    - Must run on a gaming laptop with 16GB of RAM, a 4060 with 8GB VRAM, and a Ryzen 7 7735HS

## User Need, Scope

## Risks

## C

## Preliminary Work Done & Further Work
- TODO: data audit, smoke test experiment, measured resource estimate
- TODO: detail what i'll do next

## Fallback
- TODO: detail fallback experiment if this one doesnt go well


1. User Need, Scope & Acceptance Criteria: The intended user, prototype purpose, and 3–5 testable acceptance criteria addressing the domain’s open dilemma, including an operational/resource constraint (e.g., peak VRAM, runtime).
2. Model Choice & Course Concepts: The proposed dataset configuration, model architecture, chosen difficulty band, and how course concepts and literature support the choices.
3. Preliminary Feasibility Evidence: An initial data audit, baseline smoke-test experiment, or measured resource estimate. A fully trained final model is not required.
4. Risks, Budget & Fallback: Key technical risks, computing budget, the next planned experiment, and a workable fallback.
