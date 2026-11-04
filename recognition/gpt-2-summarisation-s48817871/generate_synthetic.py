import openai
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

MODEL = "Qwen 3.6 35B A3B IQ4_XS Speed"
BASE_URL = "http://localhost:9931/v1"
KEY = "key"
EFFORT = "none"
RADIOLOGY_REPORT = 'radiology_report'
LAYMAN_REPORT = 'layman_report'

LIMIT_ROWS = 10000

GENERATED_FILE = "synthetic.csv"

REPORT_REWRITE_FORMAT = "Can you please rewrite the following in roughly the same length? \"{}\""
REPORT_SUMMARISE_FORMAT = "Can you please rewrite and summarise the following in roughly the same length for a 6th grade reading level? \"{}\""

def rewrite_and_summarise(report: str, client: openai.OpenAI):
    response = client.responses.create(
        model=MODEL,
        reasoning={"effort":EFFORT},
        input=REPORT_REWRITE_FORMAT.format(report),
    )
    rewrite = response.output_text
    response = client.responses.create(
        model=MODEL,
        reasoning={"effort":EFFORT},
        input=REPORT_SUMMARISE_FORMAT.format(report),
    )
    summary = response.output_text
    return rewrite, summary

if __name__ == "__main__":
    client = openai.OpenAI(
            base_url=BASE_URL,
            api_key=KEY,
            )
    bls = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track", split="train")
    df = pd.DataFrame(columns=[RADIOLOGY_REPORT, LAYMAN_REPORT])
    LIMIT_ROWS = min(bls.num_rows, LIMIT_ROWS)
    for i in tqdm(range(LIMIT_ROWS)):
        radreport = str(bls[RADIOLOGY_REPORT][i])
        reword_report, new_summary = rewrite_and_summarise(radreport, client)
        df.loc[i] = {RADIOLOGY_REPORT: reword_report, LAYMAN_REPORT: new_summary}
    df.to_csv(GENERATED_FILE)
