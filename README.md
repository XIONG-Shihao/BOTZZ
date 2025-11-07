## Project Overview

This project investigates how **Large Language Model (LLM) confidence** correlates with **answer correctness**, and proposes a lightweight mechanism for **adaptive inference** — where retrieval augmentation or further reasoning is only triggered when the model is uncertain.

The goal is to design an **offline-capable educational chatbot** that can assist students in science learning environments without relying on cloud-based APIs or large compute resources. In this fine-tuned model, we applied **LoRA(Low-Rank Adaptation of Large Language Models)** and **RAG(Retrieval-Augmented Generation)**

---
## Acknowledgement

This project uses the **ScienceQA dataset** released by  
**Lu et al., 2022 – "Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering"**.

The dataset (`problems.json`, `pid_splits.json`, and `captions.json`) are adapted from the official [ScienceQA GitHub repository](https://github.com/lupantech/ScienceQA) for experimental and educational purposes under their academic use license.

We gratefully acknowledge the original authors for their contribution to open research datasets that make this project possible.

---

## Background and Motivation

Most educational chatbots depend on internet-connected, high-parameter models such $as GPT-4. These systems are powerful but costly and inaccessible in offline or low-resource environments.$  

Our motivation was to explore whether smaller, quantized, or fine-tuned LLMs can remain useful by introducing a **confidence-driven control mechanism**:

> The chatbot should “trust itself” when confident,  
> and “look up external knowledge” when uncertain.

To test this idea empirically, we first needed to measure how accurately a model’s *self-reported confidence* aligns with its *actual correctness*.  
This repository implements that analysis.

---
## Research Objective

The central research question:

> **Does an LLM’s output confidence correlate with the correctness of its answers on scientific reasoning tasks?**

If so, we can use confidence as a decision signal to design an *adaptive retrieval-augmented* chatbot — one that dynamically chooses between direct answering and external information lookup.

---
## Repository Contents
scienceqa-confidence-analysis/
│
├── data/
│   └── 4o_wrong_indexes.json        # Indices of ScienceQA questions GPT-4o-mini got wrong
│
├── notebooks/
│   ├── version_2.ipynb              # Core experiment notebook (confidence–accuracy analysis)
│   └── index.ipynb                  # Helper notebook to load and inspect the difficult subset
│
├── requirements.txt                 # Minimal dependencies
└── README.md                        # This file

---
## 📚 Referenced Papers and Techniques

This project builds upon several foundational works in neural networks, interpretability, and retrieval-augmented generation.  
The following studies and frameworks informed our methods and inspired key experimental design choices:

- **Lu et al. (2022), ScienceQA: A Dataset for Multi-Modal Science Question Answering**  
  → Provided the dataset benchmark used in all experiments. The dataset combines image and text reasoning, allowing us to analyze model confidence in multi-modal contexts.

- **Hu et al. (2022), LoRA: Low-Rank Adaptation of Large Language Models**  
  → Inspired the team’s later stage experiments on lightweight fine-tuning for the “hard subset” of questions identified from the confidence analysis.

- **Lewis et al. (2020), Retrieval-Augmented Generation (RAG)**  
  → Provided the conceptual foundation for BOTZZ’s “adaptive retrieval” stage, where uncertain predictions trigger a retrieval process to consult external context.

- **Touvron et al. (2023), LLaMA: Open and Efficient Foundation Language Models**  
  → Referenced during comparison with smaller on-device LLMs (Gemma, LLaMA, DeepSeek) evaluated for feasibility of offline deployment.

These works collectively shaped the hybrid strategy of combining **confidence estimation**, **adaptive retrieval**, and **parameter-efficient fine-tuning** as seen in the later architecture design.

---
##  Experimental Setup

### Dataset  
- **ScienceQA (Lu et al., 2022)**  
  A multimodal question–answering dataset containing 21,208 examples from science textbooks and lectures across primary to high school levels.

### Model  
- **GPT-4o-mini (OpenAI)**  
  A small, efficient model supporting `logprobs` output for token-level probability extraction.

### Method  
1. Each question in ScienceQA’s test split was queried to GPT-4o-mini.  
2. The model’s **predicted answer** and **token log probabilities** were retrieved via API.  
3. Log-probabilities were converted to linear **confidence scores**.  
4. Results were evaluated for:
   - Accuracy (correct vs. incorrect)
   - Confidence distribution
   - Conditional probabilities between confidence and correctness
5. Indices of incorrect answers were saved as `4o_wrong_indexes.json` for later re-evaluation.

---

## Key Results

The experiments revealed a strong statistical correlation between model confidence and correctness:

| Conditional Probability | Value | Interpretation |
|--------------------------|--------|----------------|
| P(confident \| correct) | 0.8868 | 88.7% of correct answers were high-confidence |
| P(confident \| incorrect) | 0.2505 | Only 25% of wrong answers were falsely confident |
| P(correct \| confident) | **0.9640** | When confident, GPT-4o-mini was 96.4% accurate |
| P(correct \| unconfident) | 0.5333 | When unconfident, accuracy dropped sharply |

These results show that **confidence is a reliable predictor of correctness**, validating its use as a control variable for adaptive computation.

#### **Table 2. Method Accuracy Comparison (Full Test Set)**

| Model Setup | Accuracy (%) |
|--------------|--------------|
| URL Fine-tuned (quantized)* | 71.52 |
| Zero-shot (quantized) | 73.21 |
| RAG (quantized) | 65.64 |
| RAG (URL fine-tuned + quantized) | 63.36 |
| Mixed method ∼ Full test set | **73.52** |

*The URL fine-tuned model used URL image prompts instead of base64 encoding, resulting in a **5.72% accuracy gain** over the baseline fine-tuned model.  
Overall fine-tuning produced a **6.3% improvement**, showing that even with image handling challenges, quantized models benefited from targeted optimization.

---

#### **Table 3. Model Accuracy Comparison (Difficult Subset)**

| Model | Accuracy (%) |
|--------|---------------|
| GPT-4o Zero-shot | 49.28 |
| Gemini-2.0-Flash Zero-shot | 55.21 |
| Gemini-2.0-Flash with RAG (#1) | **60.82** |
| Gemini-2.0-Flash with RAG (#2) | 57.96 |

These results were computed on the **“difficult subset”** extracted from `4o_wrong_indexes.json`, containing the examples GPT-4o-mini got wrong.  
They demonstrate that **RAG improves accuracy by up to 11.5%** on complex questions requiring external retrieval.

---

#### **Table 4. Quantized Model Comparison (Full Test Set)**

| Model | Accuracy (%) |
|--------|---------------|
| Gemma3-4B Zero-Shot Quantized | 49.28 |
| Gemma3-4B Zero-Shot (Fine-tuned + Quantized) | **55.21** |

Even with substantially fewer parameters than GPT-4o or Gemini-Flash, the **Gemma 3-4B** model achieved competitive accuracy on the difficult subset.  
This confirms the feasibility of running **quantized, fine-tuned models** efficiently on consumer hardware (e.g., Apple M3 chips), aligning with BOTZZ’s design goals for **low-resource educational chatbots**.

---

## Visual Findings

The following visualizations were generated from `fine_tune.ipynb`:

- **Histogram:** Ratio of correct vs. incorrect answers across confidence bins.  
- **Conditional Accuracy Curve:** Sharp rise in accuracy near 100% confidence.  
- **Grade-Level Accuracy:** Stable across grades, slightly higher in senior years.  
- **Image vs. Text Questions:** Slightly lower performance on image-dependent questions.  
- **Subject-Level Accuracy:** Consistently high across Natural, Social, and Language Science.

Together, these demonstrate that the model’s uncertainty signal generalizes well across difficulty, topic, and modality.

---

## Difficult Subset Extraction

After analysis, 12% of ScienceQA test questions were answered incorrectly.  
Their indices were exported to `data/4o_wrong_indexes.json`.

This subset serves as the **“hard question set”** used in later project stages (fine-tuning, adaptive RAG, and quantization).  
You can reload and inspect this subset with the helper notebook:

```python
# in notebooks/index.ipynb
import json
from datasets import load_dataset

dataset = load_dataset("science_qa", "science_qa")
with open("../data/4o_wrong_indexes.json") as f:
    wrong = json.load(f)

subset = dataset["test"].select(wrong)
print(subset[0])
```

## How to Use
### 1. Install Dependencies
To ensure compatibility with the original experiment, install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

If you don’t have a requirements.txt file yet, create one with the following minimal dependencies:
```
openai
numpy
pandas
matplotlib
tqdm
datasets
```

### 2. Run the Main Analysis
Open the main Jupyter notebook to reproduce the confidence–accuracy experiment:
```
jupyter notebook notebooks/fine_tune.ipynb
```

Then execute the cells in order to:
	•	Load or query the ScienceQA dataset.
	•	Evaluate GPT-4o-mini responses with log-probabilities.
	•	Compute confidence scores and correctness rates.
	•	Generate all relevant figures and statistical outputs.
	•	Export the set of incorrect predictions to data/4o_wrong_indexes.json.

This notebook reproduces the numerical and visual results presented in the final project report, including:
	•	Accuracy vs. confidence correlation plots.
	•	Grade-level and subject-level breakdowns.
	•	Conditional probability tables.

### 3. Load and Inspect the “Difficult Subset”
The file 4o_wrong_indexes.json contains the indices of test questions GPT-4o-mini answered incorrectly.

You can reload and inspect those examples using the helper notebook:

```
jupyter notebook notebooks/index.ipynb
```

Inside that notebook, the following Python snippet demonstrates how to use the index file:

```python
import json
from datasets import load_dataset

# Load ScienceQA dataset
dataset = load_dataset("science_qa", "science_qa")

# Load the indices of wrong predictions
with open("../data/4o_wrong_indexes.json") as f:
    wrong = json.load(f)

# Select the corresponding questions
subset = dataset["test"].select(wrong)

# Display one example
print(subset[0])
```

This subset corresponds to the “hard questions” later used in fine-tuning and RAG experiments described in the final report.

### 4. Reproducing the Report’s Results

Running the notebook end-to-end will output:
	•	Confidence-conditioned accuracy values
	•	Visualization figures comparable to Figures 1–6 from the final report
	•	Exported list of incorrect answer indices
	•	Mean and variance of model confidence across the dataset

These reproduce the quantitative results reported in Section V (Results Analysis) of the final COMP9444 project report.