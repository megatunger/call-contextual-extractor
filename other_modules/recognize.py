import uuid
from time import sleep

import sentry_sdk
from fastapi import APIRouter, Depends, Form, UploadFile

from app.core.config import settings
from app.dependencies import get_api_key_security  # Import dependency
from app.utils import (decode_array, encode_array_nkey,
                       get_decoder_ngram_model, get_redis_db,
                       get_wav2vec_model, read_and_resample_audiofile)
from app.utils.gender import detect_gender, get_gender_model

router = APIRouter(
    prefix="/recognize",
    tags=["recognize"],
    dependencies=[Depends(get_api_key_security)]
)

# Initialize model and processor from pretrained model
wave2vec_model = get_wav2vec_model(model=False, processor=True)

# Initialize KenLM language model for decoding
language_model = get_decoder_ngram_model(
    wave2vec_model.processor.tokenizer, settings.LANGUAGE_MODEL)

# Initial Gender recognition boosting model
gender_model = get_gender_model()

# Initialize Redis connection to manage queue and store results
redis_db = get_redis_db()

# time_per_frame = model.config.inputs_to_logits_ratio / processor.feature_extractor.sampling_rate
# model.config.inputs_to_logits_ratio = 320
time_per_frame = 320 / wave2vec_model.processor.feature_extractor.sampling_rate


@router.post("/")
async def main(audio_file: UploadFile, word_level: str = Form(None), confidence_score: str = Form(None), gender_recog: str = Form(None)):
    # Initialize response dictionary with default failure status
    response = {"success": False}

    # Read and preprocess the uploaded audio file
    # Returns resampled audio data and sample rate
    audio = read_and_resample_audiofile(audio_file.file)

    # Generate unique identifier for this audio processing request
    audio_id = str(uuid.uuid4())

    # Use Redis pipeline to perform multiple operations atomically
    with redis_db.pipeline() as pipe:
        # Set expiring key to track audio processing status
        pipe.set(audio_id + '0', 1, ex=settings.REDIS_AUDIO_QUEUE_INPUT_EXPIRE)
        # Add encoded audio data to processing queue
        pipe.rpush(
            settings.REDIS_AUDIO_QUEUE,
            encode_array_nkey(audio_id, audio))
        # Execute all pipeline commands
        pipe.execute()

    if gender_recog:
        try:
            gender_prob = detect_gender(gender_model, audio_file)
            response.update({
                "gender_prob": gender_prob
            })
        except Exception as error:
            sentry_sdk.capture_exception(error)

    # Polling loop to wait for processing completion
    # Loop for maximum CLIENT_MAX_TRIES attempts
    num_tries = 0
    output = None
    for _ in range(settings.CLIENT_MAX_TRIES):
        num_tries += 1
        # Attempt to retrieve and delete the processed result
        output = redis_db.getdel(audio_id)
        # If result exists, decode and return transcription
        if output is not None:
            # Convert binary output to logits array
            logits = decode_array(output)

            # Apply language model to get final transcription
            beam_outputs = language_model.decode_beams(
                logits, beam_width=settings.BEEM_SEARCH)
            transcription = beam_outputs[0][0]

            # Update response with successful transcription
            response.update({
                "transcription": transcription,
                "success": True
            })

            if word_level:
                # Returns the transcription and word-level timing information.
                word_levels = []
                best_beam = beam_outputs[0]
                if len(best_beam) > 2 and best_beam[2] is not None:
                    for word, (start, end) in best_beam[2]:
                        word_levels.append({
                            "word": word,
                            "start": start * time_per_frame,
                            "end": end * time_per_frame,
                        })
                response.update({
                    "word_levels": word_levels,
                })

            # If confidence score is requested, add it to the response
            if confidence_score:
                best_beam = beam_outputs[0]
                if len(best_beam) > 1:
                    response.update({
                        "model_confidence_score": best_beam[-2],
                        "lm_confidence_score": best_beam[-1],
                    })

            break
        # Wait before next polling attempt
        sleep(settings.CLIENT_SLEEP)

    if output is None:
        sentry_sdk.capture_message("Time out waiting for Redis response")

    return response
