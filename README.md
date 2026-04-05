# ru-review-summarization

**authors:** Arefiev T., [ваши имена]

Generative summarization of Russian movie reviews using knowledge distillation from LLM and fine-tuning of compact encoder-decoder models.

We implement and compare three seq2seq architectures for abstractive summarization:

- **ruT5-base** (Sber) – universal text-to-text model, optimal balance of quality and speed
- **ruBART-large** (IlyaGusev) – denoising autoencoder, strong contextual understanding
- **ruPEGASUS** (attempted) – specialized for summarization, but English version failed on Russian

Experiments are conducted on **Kinopoisk reviews** dataset with 18,405 train/validation/test pairs.

---

## 📊 Datasets

| Dataset | Source | Domain | Classes (sentiment) | Train/Val/Test |
|---------|--------|--------|---------------------|----------------|
| Kinopoisk Reviews | Glepka/kinopoisk_classification | Movie reviews (Russian) | pos / neu / neg | 16,400 / 1,840 / 1,841 |

**Key challenge:** No large Russian summarization datasets exist. We solved this via **knowledge distillation** – generating gold summaries using Qwen/Qwen2.5-1.5B-Instruct.

### Dataset Structure

| Split | Size | Avg review length | Avg summary length |
|-------|------|-------------------|--------------------|
| Train | 16,400+ | 1,247 chars | 187 chars |
| Validation | 1,840 | 1,238 chars | 185 chars |
| Test | 1,841 | 1,251 chars | 186 chars |

**HF Dataset:** [Auttar/KinopoiskReviewsSummarization](https://huggingface.co/datasets/Auttar/KinopoiskReviewsSummarization)

---

## 🧠 Model

**Qwen/Qwen2.5-1.5B-Instruct** – used for generating gold summaries (knowledge distillation).  
**ruT5-base** (220M params) – main model for fine-tuning on the synthesized dataset.

We also tested:
- **ruBART-large** (406M params) – worse results, possibly due to insufficient data
- **google/pegasus-x-base** – English model, failed on Russian (output garbage)

---

## ⚙️ Methods Implemented

### 1. Baseline (TF-IDF Extractive)
Simple extractive summarization:
- Split text into sentences
- Vectorize with TF-IDF
- Rank sentences by word importance
- Select top-3 sentences in original order

**Pros:** Fast, interpretable. **Cons:** "Ragged" summaries, no paraphrasing.

### 2. ruT5-base (Fine-tuned)
Encoder-decoder with unified "text-to-text" approach:
- Prefix: `"summarize: "`
- Max input length: 512 tokens
- Max output length: 128 tokens
- Learning rate: 5e-5, batch size: 4, epochs: 3

### 3. ruBART-large (Fine-tuned)
Denoising autoencoder with bidirectional context:
- No prefix required
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
| Baseline (TF-IDF) | 0.0331 | 0.0029 | 0.0291 | 0.8200 | 0.7000 |
| **ruT5-base** | **0.1337** | **0.0673** | **0.1337** | **0.8500** | **0.7500** |
| ruBART-large | 0.0580 | 0.0267 | 0.0580 | 0.7442 | 0.7540 |
| ruPEGASUS (English) | 0.1323 | 0.0483 | 0.1323 | 0.5890 | 0.2032 |

### Key Findings

**ruT5-base** achieved the best results:
- **+304% improvement** over baseline in ROUGE-1
- Model learned to extract structure (pluses/minuses)
- Generates coherent, fluent summaries

**ruBART-large performed worse** than expected (ROUGE-1 0.058 vs 0.134 for T5):
- Possible reasons: requires more data (18k pairs insufficient), hyperparameter sensitivity, or model-specific architecture mismatch

**PEGASUS (English) failed on Russian** – generated garbage output due to tokenizer incompatibility with Cyrillic

### Qualitative Examples

**Example 1 (Positive review – T5 works well):**

> **Review:** "Фильм просто бомба! Актеры играют отлично, сюжет держит в напряжении до конца. Операторская работа на высоте..."

> **Baseline:** "Актеры играют отлично, сюжет держит в напряжении до конца. Операторская работа на высоте..."

> **ruT5:** "**Главная мысль:** Отличная игра актеров, захватывающий сюжет, высокое качество операторской работы."

**Example 2 (Analytical review – T5 struggles):**

> **Review:** "Фильм про войну, который не о войне. Вместо боевых сцен мы видим историю дружбы и выживания..."

> **ruT5:** "**Фильм 'Война'**: - **Главная мысль**: Актерская игра была высоко оценена. **Плюсы**: Актерская игра..."

*Issue: Model generates repetitive structure and hallucinates details.*

---

## 📌 Conclusions

### What worked

1. **Knowledge distillation** successfully created a Russian summarization dataset without manual labeling
2. **ruT5-base** is the optimal balance for this task:
   - 220M parameters → fast inference
   - 4x ROUGE improvement over baseline
   - Good structure extraction (pluses/minuses)

### What failed

1. **ruBART-large** underperformed despite larger size – likely needs more data (50k+ pairs)
2. **English PEGASUS** cannot handle Russian – must use multilingual or Russian-specific models
3. **GPT/BERT are unsuitable** for generative summarization (architecture mismatch)

### Current limitations

| Limitation | Severity | Solution |
|------------|----------|----------|
| Low absolute ROUGE (0.13 vs target 0.30) | 🔴 High | Expand dataset to 59k pairs |
| Hallucinations (adding false facts) | 🔴 High | Factuality metrics + contrastive learning |
| Small dataset (18k pairs) | 🟡 Medium | Use all 59k original reviews |
| Model struggles with long/negative reviews | 🟡 Medium | Hierarchical summarization + more negative examples |

---

## 🚀 How to Reproduce

### Setup

```bash
git clone https://github.com/Auttar/ru-review-summarization.git
cd ru-review-summarization
pip install -r requirements.txt
```

### Train ruT5-base

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset

# Load dataset
dataset = load_dataset("Auttar/KinopoiskReviewsSummarization")

# Tokenizer
tokenizer = T5Tokenizer.from_pretrained("sberbank-ai/ruT5-base")
tokenizer.pad_token = tokenizer.eos_token

# Preprocessing
def preprocess(examples):
    inputs = [f"summarize: {text}" for text in examples["review"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(examples["summary"], max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = [[-100 if t == tokenizer.pad_token_id else t for t in seq] for seq in labels["input_ids"]]
    return model_inputs

# Train
train_dataset = dataset["train"].map(preprocess, batched=True)
model = T5ForConditionalGeneration.from_pretrained("sberbank-ai/ruT5-base")
# ... training code
```

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
- **Model:** https://huggingface.co/Auttar/RuT5SentimentSummarization
- **GitHub:** https://github.com/Auttar/ru-review-summarization

---

## 📚 References

- Raffel et al. (2020). "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (T5)
- Lewis et al. (2020). "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation"
- Zhang et al. (2020). "BERTScore: Evaluating Text Generation with BERT"
- Gusev (2022). "ruT5, ruBART, ruGPT: Russian NLP models" (Hugging Face)

---

## 📝 Acknowledgments

- Original Kinopoisk reviews dataset: [Glepka/kinopoisk_classification](https://huggingface.co/datasets/Glepka/kinopoisk_classification)
- Knowledge distillation with Qwen/Qwen2.5-1.5B-Instruct
- Hugging Face Transformers library
```
