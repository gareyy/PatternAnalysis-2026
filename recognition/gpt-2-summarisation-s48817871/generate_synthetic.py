import openai
from datasets import load_dataset
import pandas as pd
import asyncio
from tqdm.asyncio import tqdm_asyncio

MODEL = "Qwen 3.6 35B A3B IQ4_XS Speed"
BASE_URL = "http://localhost:9931/v1"
KEY = "key"
EFFORT = "none"
RADIOLOGY_REPORT = 'radiology_report'
LAYMAN_REPORT = 'layman_report'

LIMIT_ROWS = 10

GENERATED_FILE = "synthetic.csv"

REPORT_REWRITE_FORMAT = "Please rewrite the following in roughly the same length. \"{}\""
REPORT_SUMMARISE_FORMAT = "Please rewrite and summarise the following, keep it roughly the same length, reword it to a 6th grade reading level, keep the summary third person. \"{}\""

async def rewrite_and_summarise(report: str, client: openai.OpenAI, i: int, df: pd.DataFrame) -> None:
    response = await client.responses.create(
        model=MODEL,
        reasoning={"effort":EFFORT},
        input=REPORT_REWRITE_FORMAT.format(report),
    )
    rewrite = response.output_text
    response = await client.responses.create(
        model=MODEL,
        reasoning={"effort":EFFORT},
        input=REPORT_SUMMARISE_FORMAT.format(report),
    )
    summary = response.output_text
    df.loc[i] = {RADIOLOGY_REPORT: rewrite, LAYMAN_REPORT: summary}

async def main():
    client = openai.AsyncOpenAI(
            base_url=BASE_URL,
            api_key=KEY,
            )
    bls = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track", split="train")
    df = pd.DataFrame(columns=[RADIOLOGY_REPORT, LAYMAN_REPORT])
    limrows = min(bls.num_rows, LIMIT_ROWS)
    await tqdm_asyncio.gather(
        *(rewrite_and_summarise(str(bls[RADIOLOGY_REPORT][i]), client, i, df) for i in range(limrows))
            )
    df.to_csv(GENERATED_FILE)

if __name__ == "__main__":
    asyncio.run(main())
