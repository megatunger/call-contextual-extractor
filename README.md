# Call Contextual Extractor

## Dataset:

- Only very small subset dataset file is available in Repo
- To download the full dataset (12726 records), please contact

## Installation

Create the environment (first time only), then always **activate it** before installing or running Python so packages match [environment.yml](environment.yml):

```bash
conda env create -f environment.yml
conda activate ml-audio-llm
```

After pulling dependency changes, update the same env (still activated):

```bash
conda env update -f environment.yml --prune
```

```bash
jupyter lab
```

Use the **ml-audio-llm** kernel in notebooks so imports resolve to this env.

### Silero VAD dependencies

Aligned with the [Silero VAD wiki — Dependencies](https://github.com/snakers4/silero-vad/wiki/Examples-and-Dependencies#dependencies):

- **torch** (pinned in `environment.yml`)
- **torchaudio** (audio I/O; note: some formats may still be loaded via **librosa** / **ffmpeg** in notebooks)
- **silero-vad** (pip package with `load_silero_vad`, `get_speech_timestamps`, …)
- **ffmpeg** (conda) for `ffprobe` metadata and decoders
- **Optional:** `onnxruntime` if you switch Silero to the ONNX path (`onnx=True` in `load_silero_vad`)

## Pipeline

### Data preparation — [`01_PrepareDataCalls.ipynb`](01_PrepareDataCalls.ipynb)

Loads `dataset/staging_calls.csv`, explores fields, and downloads a **limited** set of recording URLs into `data/raw_audio/` (see `TEST_DOWNLOAD_LIMIT` in the notebook).

### VAD → ASR → transcript — [`02b_SileroVAD_Final.ipynb`](02b_SileroVAD_Final.ipynb)

Production notebook with **three separable stages** (run independently, stop/resume anytime):

| Stage           | Notebook flag       | Output                                                                        |
| --------------- | ------------------- | ----------------------------------------------------------------------------- |
| 1 — VAD segment | `RUN_STAGE_1_VAD`   | `data/segmented_audio/<call>/segments_manifest.jsonl`, `vad.done`, WAV chunks |
| 2 — ASR         | `RUN_STAGE_2_ASR`   | `asr_results.jsonl` (checkpointed per segment)                                |
| 3 — Merge       | `RUN_STAGE_3_MERGE` | `final_dialogue.json` (`Agent:` / `Customer:` lines)                          |

Implementation lives in [`pipeline/`](pipeline/) (`vad_segment.py`, `asr_recognize.py`, `transcript.py`). Set `MAX_CALLS = None` to process all files in `data/raw_audio/`.

Smoke test notebook: [`02_SileroVADTest.ipynb`](02_SileroVADTest.ipynb) (visual VAD inspection).

#### Stereo recordings and customer-focused context

For **contextual extraction** (what the **customer** said or needs), **splitting the legs matters**. A **mono downmix** blends agent and customer, which weakens attribution and makes it harder to **focus on the customer**.

**Default for this project:** **UCall** stereo recordings use **left = agent**, **right = customer** → **split channels** and run **VAD / ASR on the customer channel** (**channel index 1**, `CUSTOMER_CHANNEL_INDEX` in [`02_SileroVADTest.ipynb`](02_SileroVADTest.ipynb)). **Why 16 kHz:** Silero’s wideband model expects **16 kHz** (fixed chunk size vs sample rate); raw files are often **48 kHz**, so the notebook resamples before VAD.

Whole-call mono VAD is only reasonable for coarse “is anyone speaking?” checks, not for **customer-centric** pipelines.

### Step 2: Automatic Speech Recognition (ASR)

ASR runs on a deployed service. POST each VAD segment to the recognize endpoint, then merge segment-level transcripts by timestamp to reconstruct a full call conversation (`Agent: ...`, `Customer: ...`).

**Endpoint:** `http://103.140.249.39:8000/recognize/`

**Auth:** Send the API key in the `x-api-key` header. Store it in repo-root `.env` as `SPEECH_API_KEY=...` (or export `SPEECH_API_KEY` in your shell; the env var overrides `.env`).

**Form fields:**

| Field        | Value                                           |
| ------------ | ----------------------------------------------- |
| `audio_file` | WAV (or other supported) file upload            |
| `word_level` | `"1"` for word-level timestamps in the response |

Example (multipart upload):

```bash
curl --location 'http://103.140.249.39:8000/recognize/' \
  --header "x-api-key: ${SPEECH_API_KEY}" \
  --form 'audio_file=@path/to/segment.wav' \
  --form 'word_level="1"'
```

```python
import os
import requests

url = "http://103.140.249.39:8000/recognize/"
api_key = os.environ["SPEECH_API_KEY"]
headers = {"x-api-key": api_key}

with open("path/to/segment.wav", "rb") as file:
    files = {"audio_file": file}
    data = {"word_level": "1"}
    response = requests.post(url, headers=headers, files=files, data=data)
response.raise_for_status()
# Parse response JSON per contract below
```

Expected API response:

```json
{
  "success": true,
  "transcription": "thế sao tôi bận lắm tôi không đi được",
  "word_levels": [
    {
      "word": "thế",
      "start": 1.9000000000000001,
      "end": 2.0
    },
    {
      "word": "sao",
      "start": 2.06,
      "end": 2.2800000000000002
    },
    {
      "word": "tôi",
      "start": 2.52,
      "end": 2.64
    },
    {
      "word": "bận",
      "start": 2.74,
      "end": 2.84
    },
    {
      "word": "lắm",
      "start": 2.94,
      "end": 3.04
    },
    {
      "word": "tôi",
      "start": 3.14,
      "end": 3.24
    },
    {
      "word": "không",
      "start": 3.2800000000000002,
      "end": 3.4
    },
    {
      "word": "đi",
      "start": 3.42,
      "end": 3.5
    },
    {
      "word": "được",
      "start": 3.56,
      "end": 3.7
    }
  ]
}
```

Failure case:

```json
{
  "success": false
}
```

Final output format (after merging recognized segments):

```json
["Agent: Em có gì không?", "Customer: Dạ không chị ơi"]
```

Resume behavior in `02b_SileroVAD_Final.ipynb`:

- **Stage 1:** calls with `vad.done` are skipped on rerun (`skip_vad_if_done=True`).
- **Stage 2:** each segment append to `asr_results.jsonl`; successful rows are skipped on rerun (failed rows retried by default).
- **Stage 3:** rebuilds `final_dialogue.json` from the latest ASR checkpoint.

Progress bars (`tqdm`) show call-level progress in Stage 1 and segment-level progress in Stage 2.

### Step 3: Dataset Formatting & LLM Fine-Tuning

In this step, we construct a fine-tuning dataset by passing the transcripts (`final_dialogue.json`) to a teacher LLM (e.g., Gemini `gemini-3.1-flash-lite`) to extract structured customer information. The generated dataset is saved in JSONL format with `instruction`, `input`, and `response` pairs. We then use this dataset to fine-tune a smaller, on-device model (e.g., Unsloth Qwen 3.5 or other candidates) using LoRA.

**Target CRM Fields:**

- `customer_sector`: string
- `customer_needs`: a string (summarized)
- `customer_interested`: number (from scale 1 to 10)
- `customer_busy`: boolean (true / false)
- `customer_scheduled_at`: time string (or null if not scheduled)
- `customer_rating` (optional): number (from scale 1 to 10, sometimes we only have this)

**Scripts:**

- `pipeline/dataset_builder.py`: Calls the teacher LLM to format the dataset.
- `pipeline/fine_tuner.py`: Handles LoRA fine-tuning for the specified small models.

Preview and exploration are available in [`03_FineTuning.ipynb`](03_FineTuning.ipynb).

### Step 4: Model Evaluation, JSON Extraction & API Service

After fine-tuning (Step 3), compare candidate models on held-out transcripts, then serve the full **audio → CRM JSON** pipeline over HTTP.

#### Model evaluation — [`04_Evaluation.ipynb`](04_Evaluation.ipynb)

Runs [`pipeline/evaluator.py`](pipeline/evaluator.py) on a **fixed random sample** (20 rows, `seed=42`) from `data/finetuning_dataset.jsonl`, comparing each model’s structured output to the **Gemini teacher labels** in the `response` field.

**Metrics:**

| Metric | Meaning |
| ------ | ------- |
| `Valid_JSON_%` | Share of samples where the model output parses as JSON |
| `Busy_Accuracy_%` | Exact match on `customer_busy` (among valid JSON rows) |
| `Sector_Match_%` | Exact match on `customer_sector` (case-insensitive) |
| `Schedule_Match_%` | Exact match on `customer_scheduled_at` |
| `Interested_MAE` | Mean absolute error on `customer_interested` (1–10) |
| `Rating_MAE` | Mean absolute error on `customer_rating` when present |

Results are written to `data/evaluation_results.csv`; the notebook plots parsability, field-match accuracy, and MAE with seaborn.

**CLI (local or after Colab setup):**

```bash
python pipeline/evaluator.py --models data/models/Qwen3.5-2B_lora data/models/Qwen3.5-0.8B_lora
```

On Colab (GPU), the notebook installs `condacolab`, mounts Drive, and pulls Unsloth drivers (`unsloth[colab-new]`, `xformers`, etc.) before running the evaluator.

#### End-to-end API — [`api.py`](api.py) · demo [`05_API_Demo_Colab.ipynb`](05_API_Demo_Colab.ipynb)

[`api.py`](api.py) exposes a single FastAPI endpoint that chains the production pipeline:

1. **Upload** call audio (WAV or other supported format)
2. **VAD** — Silero segments on the customer channel ([`pipeline/vad_segment.py`](pipeline/vad_segment.py))
3. **ASR** — parallel requests to the recognize service ([`pipeline/asr_recognize.py`](pipeline/asr_recognize.py); requires `SPEECH_API_KEY` in `.env`)
4. **Merge** — `Agent:` / `Customer:` lines ([`pipeline/transcript.py`](pipeline/transcript.py))
5. **Extract** — fine-tuned LoRA inference ([`pipeline/inference.py`](pipeline/inference.py))

At startup the server loads Silero VAD and the LoRA adapter at `data/models/Qwen3.5-2B_lora` (override path in `api.py` if needed). Uploaded files and segment dirs are removed after each request.

**Run locally:**

```bash
conda activate ml-audio-llm
uvicorn api:app --host 0.0.0.0 --port 8000
```

**Request:** `POST /extract` — multipart form field `file` (audio upload).

**Success response:**

```json
{
  "call_id": "uuid",
  "transcript": "Agent: ...\nCustomer: ...",
  "extraction": {
    "customer_sector": "...",
    "customer_needs": "...",
    "customer_interested": 5,
    "customer_busy": false,
    "customer_scheduled_at": null,
    "customer_rating": null
  }
}
```

**Example:**

```bash
curl -X POST "http://127.0.0.1:8000/extract" \
  -F "file=@path/to/call.wav"
```

**Colab + ngrok:** [`05_API_Demo_Colab.ipynb`](05_API_Demo_Colab.ipynb) runs the same server on a Colab GPU, opens an [ngrok](https://ngrok.com/) tunnel on port 8000, and prints a public URL so you can hit `/extract` from Postman or your local machine. Set your ngrok auth token in the notebook (`ngrok.set_auth_token(...)`); do not commit tokens to the repo.

**Scripts:**

- [`pipeline/evaluator.py`](pipeline/evaluator.py): Multi-model benchmark → `data/evaluation_results.csv`
- [`pipeline/inference.py`](pipeline/inference.py): LoRA load + `generate_extraction()` (used by the API and CLI)
- [`api.py`](api.py): FastAPI app with `/extract`
