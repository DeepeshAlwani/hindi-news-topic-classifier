# Hindi News Topic Classifier — AI Pipeline

Classifies the news topic of a screenshot/image containing Hindi news content,
ignoring any advertisements or product branding present in the image.

## Setup

```bash
pip install openai
export OPENROUTER_API_KEY="sk-or-..."   # get one free at https://openrouter.ai/keys
```

## Usage

**Single image:**
```bash
python -c "from classifier import classify_image; import json; print(json.dumps(classify_image('sample.jpeg'), ensure_ascii=False, indent=2))"
```

**Batch (a folder of images):**
```bash
python batch_classify.py ./test_images --out results
```
Produces `results.json` (full detail per image, including transcribed text and reasoning)
and `results.csv` (summary table).

## Architecture

Single-stage pipeline: the image is sent directly to a vision-language model
(VLM) capable of Hindi OCR, with a system prompt that asks it to:

1. Transcribe the actual news content (headline/ticker/banner text), explicitly
   ignoring ads, product packaging, and brand logos present in the image.
2. Classify the story into one of a fixed taxonomy.
3. Return structured JSON with the transcription, translation, category,
   confidence, and one-line reasoning.

**Why single-stage instead of VLM→separate LLM:** a two-stage pipeline (OCR
model → text classifier) adds a failure point where OCR errors silently
propagate downstream, and requires two model calls per image. A single
strong multilingual VLM does the reading and reasoning in one pass. The
`extracted_text` field is still returned in every response, so if a
classification looks wrong it's possible to tell whether it was an OCR
failure or a reasoning failure without re-running anything.

**Model used:** `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` on
OpenRouter, chosen after testing several free vision models against sample
images — it correctly read dense Devanagari text and ignored the ad content
in every test image. The model is set as a single config variable in
`classifier.py` (`MODEL = ...`) and can be swapped to compare alternatives,
e.g. `google/gemma-3-27b-it:free` or `qwen/qwen2.5-vl-32b-instruct:free`.

## Taxonomy

```
Politics, Sports, Business/Economy, Entertainment, Crime, International,
Education/Exams, Technology, Health, Other
```

`Education/Exams` was added after initial testing showed govt exam/recruitment
news (a very common Indian news category — exam dates, cancellations, paper
leaks) was being forced into `Other`. Taxonomy is easy to extend by editing
the `TAXONOMY` list in `classifier.py` — no other code changes needed.

## Handling ads/branding in the image

Handled entirely through prompting rather than image preprocessing: the
system prompt explicitly instructs the model to ignore any advertisements,
product packaging, or brand logos and to focus only on the actual news
content. Tested against the sample images (which all include an R-pure
spice-brand ad panel) — the ad content did not affect classification in any
test run.

## Reliability

- `temperature=0` for reproducible output.
- Each request is retried up to 3 times (linear backoff) on rate limits,
  timeouts, or malformed JSON responses — free-tier models can be flaky
  under load.
- Failed classifications (after retries) are recorded with `_parse_error: true`
  and the raw error, rather than silently dropped, so failures are visible
  in the batch output.

## Known limitations / next steps

- Evaluated qualitatively on ~5 sample images due to time constraints; no
  labeled test set was available to compute precision/recall per category.
- Free-tier OpenRouter models are rate-limited and availability can change;
  for production use this should move to a paid tier or a self-hosted model
  for throughput and uptime guarantees.
- Taxonomy was built by inspecting sample images and is not exhaustive —
  a larger labeled dataset would help validate category coverage
  (e.g. whether "Other" is being used appropriately, not as a catch-all
  for anything the model finds ambiguous).
- Currently classifies one dominant story per image; images with multiple
  distinct news items are collapsed into the most prominent one.