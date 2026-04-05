# Call Contextual Extractor

## Installation

```bash
conda env create -f environment.yml
conda activate ml-audio-llm
```

```bash
jupyter lab
```

## Pipeline

### Step 1: Voice Activity Detection (VAD) & Segmentation (01_PrepareDataCalls)

```bash
python 01_run_vad.py --input_dir data/raw_audio/ --output_dir data/segmented_audio/
```

Pulls a sample of 10000+ raw `.wav` sales call recordings from cloud storage and processes them using Silero VAD. This step removes silence and chunks the audio into manageable segments (<1ms per chunk processing time), optimizing the data for speech recognition.

### Step 2: Automatic Speech Recognition (ASR)

ASR runs on a deployed service. Upload each audio file (e.g. VAD segments from Step 1) to the upload endpoint; the API returns transcripts (details depend on the service response).

**Endpoint:** `https://speechv2.ucall.vn/api/record/upload`

**Auth:** Send the API key in the `x-api-key` header.

Example (multipart upload, field name `audio`):

```python
import requests

url = "https://speechv2.ucall.vn/api/record/upload"
api_key = "9a45f6c1-5c72-4c71-9311-5b6c049800e4"
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
