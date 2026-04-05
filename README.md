# ru-review-summarization

**Authors:** Arefiev T., Konyukhova V., Egorova A.

Generative summarization of Russian movie reviews using knowledge distillation from LLM and fine-tuning of compact encoder-decoder models.

We implement and compare three seq2seq architectures for abstractive summarization:

- **ruT5-base** – universal text-to-text model, optimal balance of quality and speed
- **ruBART-large** – denoising autoencoder, strong contextual understanding
- **PEGASUS** – specialized for summarization, but English version failed on Russian

Experiments are conducted on **Kinopoisk reviews** dataset with 20,450 train/validation/test pairs.

---

## 📊 Datasets

| Dataset | Source | Domain | Classes (sentiment) | Train/Val/Test |
|---------|--------|--------|---------------------|----------------|
| Kinopoisk Reviews | Glepka/kinopoisk_classification | Movie reviews (Russian) | pos / neu / neg | 57,000 / 1,500 / 0 |

**Key challenge:** No large Russian summarization datasets exist. We solved this via **knowledge distillation** – generating gold summaries using Qwen/Qwen2.5-1.5B-Instruct.

### Dataset Structure

| Split | Size | Avg review length | Avg summary length |
|-------|------|-------------------|--------------------|
| Train | 16,400 | 1,247 chars | 187 chars |
| Validation | 2,050 | 1,238 chars | 185 chars |
| Test | 2,050 | 1,251 chars | 186 chars |

**HF Dataset:** [Auttar/KinopoiskReviewsSummarization](https://huggingface.co/datasets/Auttar/KinopoiskReviewsSummarization)

---

## 🧠 Models

- **Qwen/Qwen2.5-1.5B-Instruct** – used for generating gold summaries (knowledge distillation).  
- **ruT5-base** (220M params) – model for fine-tuning on the synthesized dataset.
- **ruBART-large** (406M params) – model for fine-tuning on the synthesized dataset.
- **google/pegasus-x-base** – model for fine-tuning on the synthesized dataset. English model, failed on Russian (output garbage)

---

## Methods Implemented

### 1. Baseline (TF-IDF Extractive)
Simple extractive summarization:
- Split text into sentences
- Vectorize with TF-IDF
- Rank sentences by word importance
- Select top-3 sentences in original order

**Pros:** Fast, interpretable. **Cons:** "Ragged" summaries, no paraphrasing.

### 2. sberbank-ai/ruT5-base (Fine-tuned)
Encoder-decoder with unified "text-to-text" approach:
- No prefix required
- Max input length: 512 tokens
- Max output length: 128 tokens
- Learning rate: 5e-5, batch size: 4, epochs: 3

### 3. sn4kebyt3/ru-bart-large (Fine-tuned)
Denoising autoencoder with bidirectional context:
- Same training parameters as T5

### 3. google/pegasus-x-base (Fine-tuned)
Encoder-decoder with unified "text-to-text" approach:
- Same training parameters as T5

### 4. Metrics
We evaluate with three complementary metrics:
- **ROUGE (1,2,L)** – n-gram overlap (standard)
- **BERTScore** – semantic similarity via BERT embeddings
- **SBERT Similarity** – cosine similarity of sentence embeddings

---

## 📈 Results

### Quantitative Comparison (Test set, 100 samples)

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore | SBERT Sim |
|-------|---------|---------|---------|-----------|-----------|
| Baseline (TF-IDF) | 0.0331 | 0.0029 | 0.0291 | 0.6233 | 0.5612 |
| ruT5-base | 0.1497 | 0.0600 | 0.1497 | 0.7227 | 	0.6786 |
| ruBART-large | 0.0580 | 0.0267 | 0.0580 | 0.7442 | 0.7540 |
| ruPEGASUS (English) | 0.1323 | 0.0483 | 0.1323 | 0.5890 | 0.2032 |

### Key Findings

**ruT5-base** achieved the best results on ROUGE metrics:
- **+352% improvement** over baseline in ROUGE-1 (from 0.0331 to 0.1497)
- **+16% improvement** in BERTScore (from 0.6233 to 0.7227)
- **+21% improvement** in SBERT similarity (from 0.5612 to 0.6786)
- Model learned to extract structure (pluses/minuses)
- Generates coherent, fluent summaries

**ruBART-large** shows an interesting pattern:
- Lower ROUGE scores (0.0580) but **highest semantic metrics** (BERTScore 0.7442, SBERT 0.7540)
- Suggests ruBART generates paraphrased content with different vocabulary, capturing meaning without exact n-gram overlap
- Trade-off: lexical precision (ROUGE) vs semantic fidelity (BERTScore/SBERT)

**PEGASUS (English) failed on Russian** – generated garbage output due to tokenizer incompatibility with Cyrillic

---

## Conclusions

### What worked

1. **Knowledge distillation** successfully created a Russian summarization dataset without manual labeling
2. **ruT5-base** is the optimal balance for this task:
   - 220M parameters → fast inference
   - Good structure extraction (pluses/minuses)
   - Balanced performance across all metrics

### What we learned

1. **ruBART-large** prioritizes semantic similarity over lexical overlap:
   - Lower ROUGE but higher BERTScore/SBERT
   - May be preferable for applications where meaning > exact wording
   - Could benefit from different decoding strategies (e.g., lower temperature)

2. **English PEGASUS** cannot handle Russian – must use multilingual or Russian-specific models

## How to Reproduce

### Inference

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

model = T5ForConditionalGeneration.from_pretrained("Auttar/RuT5SentimentSummarization")
tokenizer = T5Tokenizer.from_pretrained("Auttar/RuT5SentimentSummarization")

def summarize(review):
    input_text = f"summarize: {review}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs.input_ids, max_length=128, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Evaluation

```python
from rouge_score import rouge_scorer
from bert_score import BERTScorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
bert_scorer = BERTScorer(lang="ru", rescale_with_baseline=False)

# Calculate metrics
rouge_scores = scorer.score(reference, prediction)
bert_score = bert_scorer.score([prediction], [reference])
```

---

## 🔗 Links

- **Dataset:** https://huggingface.co/datasets/Auttar/KinopoiskReviewsSummarization
- **Models:** https://huggingface.co/Auttar/RuT5SentimentSummarization, https://huggingface.co/Auttar/kinopoisk-pegasus, https://huggingface.co/Auttar/kinopoisk-rubart
- **GitHub:** [https://github.com/Auttar/ru-review-summarization](https://github.com/AuttarYT/Kinopoisk_Summarization/tree/main)

---

## 📝 Acknowledgments

- Original Kinopoisk reviews dataset: [Glepka/kinopoisk_classification](https://huggingface.co/datasets/Glepka/kinopoisk_classification)
- Knowledge distillation with Qwen/Qwen2.5-1.5B-Instruct
- Hugging Face Transformers library
