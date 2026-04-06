# Call Contextual Extractor

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

### Voice activity detection — Silero — [`02_SileroVADTest.ipynb`](02_SileroVADTest.ipynb)

Smoke test: prints **per-file audio metadata** (codec, channels, sample rate, PCM subtype or N/A for compressed files) and runs **Silero VAD** on the **customer channel** when stereo. Tune `VAD_TEST_MAX_FILES` at the top of the notebook.

**Batch script (planned, not in repo yet):**

```bash
python 01_run_vad.py --input_dir data/raw_audio/ --output_dir data/segmented_audio/
```

That script should mirror the notebook defaults: **customer-only** segments for downstream context extraction.

#### Stereo recordings and customer-focused context

For **contextual extraction** (what the **customer** said or needs), **splitting the legs matters**. A **mono downmix** blends agent and customer, which weakens attribution and makes it harder to **focus on the customer**.

**Default for this project:** **UCall** stereo recordings use **left = agent**, **right = customer** → **split channels** and run **VAD / ASR on the customer channel** (**channel index 1**, `CUSTOMER_CHANNEL_INDEX` in [`02_SileroVADTest.ipynb`](02_SileroVADTest.ipynb)). **Why 16 kHz:** Silero’s wideband model expects **16 kHz** (fixed chunk size vs sample rate); raw files are often **48 kHz**, so the notebook resamples before VAD.

Whole-call mono VAD is only reasonable for coarse “is anyone speaking?” checks, not for **customer-centric** pipelines.

### Step 2: Automatic Speech Recognition (ASR)

ASR runs on a deployed service. Upload each audio file (e.g. VAD segments from the previous step) to the upload endpoint; the API returns transcripts (details depend on the service response).

**Endpoint:** `https://speechv2.ucall.vn/api/record/upload`

**Auth:** Send the API key in the `x-api-key` header (store it in an env var, not in source).

Example (multipart upload, field name `audio`):

```python
import os
import requests

url = "https://speechv2.ucall.vn/api/record/upload"
api_key = os.environ["UCALL_SPEECH_API_KEY"]
headers = {"x-api-key": api_key}

with open("path/to/segment.wav", "rb") as file:
    files = {"audio": file}
    response = requests.post(url, headers=headers, files=files)
response.raise_for_status()
# Parse response JSON / text per API contract
```

### Step 3: Dataset Formatting & LLM Fine-Tuning

_Placeholder — workflow TBD._

### Step 4: JSON Extraction & API Service

_Placeholder — workflow TBD._
