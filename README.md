# Call Contextual Extractor

## Dataset

- In repo: `dataset/staging_calls.demo.csv` (~20 rows)
- Full export (~12.7k recordings) — contact for access

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

Uses `dataset/staging_calls.csv` (not in repo — use `staging_calls.demo.csv` for a small run, or get the full CSV). Downloads recording URLs into `data/raw_audio/` (`TEST_DOWNLOAD_LIMIT` in notebook).

### VAD → ASR → transcript — [`02b_SileroVAD_Final.ipynb`](02b_SileroVAD_Final.ipynb)

Production notebook with **three separable stages** (run independently, stop/resume anytime):

| Stage           | Notebook flag       | Output                                                                        |
| --------------- | ------------------- | ----------------------------------------------------------------------------- |
| 1 — VAD segment | `RUN_STAGE_1_VAD`   | `data/segmented_audio/<call>/segments_manifest.jsonl`, `vad.done`, WAV chunks |
| 2 — ASR         | `RUN_STAGE_2_ASR`   | `asr_results.jsonl` (checkpointed per segment)                                |
| 3 — Merge       | `RUN_STAGE_3_MERGE` | `final_dialogue.json` (`Agent:` / `Customer:` lines)                          |

Implementation lives in [`pipeline/`](pipeline/) (`vad_segment.py`, `asr_recognize.py`, `transcript.py`). Set `MAX_CALLS = None` to process all files in `data/raw_audio/`.

Smoke test: [`02a_SileroVAD_Test.ipynb`](02a_SileroVAD_Test.ipynb) (visual VAD on customer channel).

#### Stereo (UCall)

Left = agent, right = customer (`CUSTOMER_CHANNEL_INDEX = 1` in [`02a_SileroVAD_Test.ipynb`](02a_SileroVAD_Test.ipynb)). Avoid mono downmix — you lose speaker attribution.

[`02b_SileroVAD_Final.ipynb`](02b_SileroVAD_Final.ipynb) and [`api.py`](api.py) VAD/ASR **both** Agent + Customer channels, then merge by time. Silero expects 16 kHz; pipeline resamples from 48 kHz when needed.

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

Gemini teacher (`gemini-3.1-flash-lite`, `GEMINI_API_KEY` in `.env`) labels `data/segmented_audio/*/final_dialogue.json` → `data/finetuning_dataset.jsonl` (full labeled pool). [`pipeline/dataset_builder.py`](pipeline/dataset_builder.py) then writes an **80/20 train/test split** (`finetuning_dataset_train.jsonl`, `finetuning_dataset_test.jsonl`, `seed=42`). LoRA fine-tune on the train split with Unsloth — [`03_FineTuning.ipynb`](03_FineTuning.ipynb).

Re-split only (no Gemini calls): `python pipeline/dataset_builder.py --split-only`

**CRM fields:** `customer_sector`, `customer_needs`, `customer_interested` (1–10), `customer_busy`, `customer_scheduled_at`, `customer_rating` (optional).

**Scripts:** [`pipeline/dataset_builder.py`](pipeline/dataset_builder.py), [`pipeline/fine_tuner.py`](pipeline/fine_tuner.py). Adapters under `data/models/`.

### Step 4: Model Evaluation, JSON Extraction & API Service

#### Evaluation — [`04_Evaluation.ipynb`](04_Evaluation.ipynb)

[`pipeline/evaluator.py`](pipeline/evaluator.py) runs on the **held-out test split** (`data/finetuning_dataset_test.jsonl`) against Gemini labels in `response`. Writes `data/evaluation_results.csv`; notebook has bar charts.

| Metric | Notes |
| ------ | ----- |
| `Valid_JSON_%` | Parses as JSON |
| `Busy_Accuracy_%` | `customer_busy` exact match |
| `Sector_Match_%` | `customer_sector`, case-insensitive |
| `Schedule_Match_%` | `customer_scheduled_at` exact match |
| `Interested_MAE` | `customer_interested` |
| `Rating_MAE` | `customer_rating` when set |

```bash
python pipeline/evaluator.py --models data/models/Qwen3.5-2B_lora data/models/Qwen3.5-0.8B_lora
```

Colab: condacolab + Drive mount + Unsloth pip deps (see notebook).

#### API — [`api.py`](api.py) · [`05_API_Demo_Colab.ipynb`](05_API_Demo_Colab.ipynb)

`POST /extract` — upload audio → VAD → ASR (`SPEECH_API_KEY`) → merge transcript → LoRA JSON (`data/models/Qwen3.5-2B_lora`). Temp files deleted per request.

```bash
conda activate ml-audio-llm
uvicorn api:app --host 0.0.0.0 --port 8000
```

```bash
curl -X POST "http://127.0.0.1:8000/extract" -F "file=@path/to/call.wav"
```

```json
{
  "call_id": "uuid",
  "transcript": "Agent: ...\nCustomer: ...",
  "extraction": { "customer_sector": "...", "customer_needs": "...", "customer_interested": 5, "customer_busy": false, "customer_scheduled_at": null, "customer_rating": null }
}
```

Colab demo: same `uvicorn` on GPU, ngrok on port 8000 — notebook prints the public URL for `/extract`. Add your ngrok token in the notebook cell; keep it out of git.
