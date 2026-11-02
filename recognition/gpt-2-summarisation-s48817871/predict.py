# TODO: like fix this up
import torch
from torch.utils.data import DataLoader
from dataset import BioLaySummDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, DataCollatorWithPadding, TokenizersBackend, PreTrainedModel
import argparse
import modules
device = 'cuda' if torch.cuda.is_available() else 'cpu'

prompts = {
    "Confirmed COVID-19. No pulmonary consolidations are observed. No pleural effusion. Conclusion: Examination within normal limits.": "The patient has tested positive for COVID-19. However, there are no signs of pneumonia in the lungs, and no fluid has built up around the lungs. The conclusion is that the examination results are normal.",
    "No significant findings.": "There are no important issues found.",
    "Comparison with the previous study dated 06 08 2012 shows the appearance of a right basal pneumothorax. Subcutaneous emphysema in the right thoracic wall. Radiological worsening of diffuse bilateral alveolar involvement.": "Compared to the study from June 8, 2012, there is now a condition called a right basal pneumothorax, which means air has leaked into the space between the lungs and chest wall on the right side. There is also air trapped under the skin of the right chest wall. The overall picture of the lungs has gotten worse, showing more involvement of both lungs.",
    "Signs of COPD. Images are compatible with a healed fracture of the posterior arch of the 10th right rib.": "Signs of chronic obstructive pulmonary disease (COPD) are present. The images also show that there was a fracture in the back part of the 10th rib on the right side, which has now healed.",
    "Chronic changes in the lung parenchyma. Biapical pleural thickening. Calcified aortic atheromatosis. Pseudonodular image in the left base likely related to the nipple. Hiatal hernia. Dorsal lumbar scoliosis.": "The lungs show long-term changes. Thickening is seen at the top of both sides of the lung coverings. There is a buildup of calcium in the fatty deposits of the main blood vessel in the abdomen. A small bump-like image on the left side is probably related to the nipple. A hiatal hernia, where the stomach bulges into the chest, is present. The lower back shows a sideways curve in the spine.",
    "Right parahilar infiltrate and atelectasis. Increased retrocardiac density related to atelectasis and consolidation associated with right pleural effusion. Clinical data is important for correct radiological assessment.": "There is a cloudiness near the right lung's airways and a part of the lung has collapsed. The area behind the heart is denser, which could be due to the collapsed lung and a possible lung infection along with fluid around the right lung. It's important to consider the patient's medical history for a proper understanding of the x-ray."
        }


parser = argparse.ArgumentParser(add_help=True)

parser.add_argument("--input-folder", default="./modelout", help='Where to load pretrained model from')
parser.add_argument("--model-name", default="openai-community/gpt2", help='Name of base model, used for getting tokeniser')
parser.add_argument("--max-eval-steps", default=-1, type=int, help="Maximum number of steps to evaluate for when calculating ROUGE and validation loss. -1 to evaluate for all batches in validation dataset")
parser.add_argument("--input-format", default="InstructionFormat", help=f"Format for inputting report and summary to model. Options include {str(list(modules.formats))[1:-1]}")

args = parser.parse_args()
inputfmt = modules.get_format(args.input_format)
tokeniser, model, = modules.get_model_and_tokeniser(args.model_name, args.input_folder, device, inputfmt)
data_collator = DataCollatorWithPadding(tokeniser)
torch.no_grad()
model.eval()
valid_dataset = BioLaySummDataset("test", tokeniser, inputfmt.fmt)
valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=16, collate_fn=data_collator)
model = model.to(device)
for p, s in prompts.items():
    modules.eval_on_prompt(p, s, inputfmt, tokeniser, model)
loss_mean, loss_stdev = modules.val_loss(model, valid_dataloader, device, max_eval_steps=args.max_eval_steps)
print(f"""--TEST SPLIT LOSSES:--
mean: {loss_mean:.5f}
stdev: {loss_stdev:.5f}""")
rouges = modules.rouge_score(model, tokeniser, valid_dataloader, device, inputfmt, max_eval_steps=args.max_eval_steps)
print(f"""--ROUGE SCORES:--
ROUGE-1: {rouges['rouge1']:.5f}
ROUGE-2: {rouges['rouge2']:.5f}
ROUGE-L: {rouges['rougeL']:.5f}
ROUGE-L-SUM: {rouges['rougeLsum']:.5f}""")
