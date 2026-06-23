# AI Engineering Intern - Week 03: Deep Learning & NLP Benchmarking
**Repository:** `ai-intern-week03-deep-learning`

This repository details the empirical evaluations, architecture comparisons, and custom fine-tuning tasks executed during Week 03 of the AI Engineering Internship.

---

## 1. Fashion-MNIST: Feed-Forward Neural Network Grid Experiments

### Objective
To build a modular Multi-Layer Perceptron (MLP/FFNN) using PyTorch and evaluate the intersectional performance of hidden layer activation functions (ReLU vs. Sigmoid) and optimization algorithms (SGD vs. Adam) under distinct Learning Rates ($LR$).

### Experimental Grid Matrix
All configurations utilize a hidden layer topology of `[128]` hidden units, trained over 5 epochs with a batch size of 64.

| Config | Activation | Optimizer | Learning Rate ($LR$) | Final Test Accuracy | Behavior & Convergence Insights |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **#1** | **ReLU** | **SGD** | `0.01` | **83.76%** | **Best for high LR.** Steady, stable gradient flow without saturation. |
| **#2** | ReLU | Adam | `0.01` | 83.43% | Fast initial drop in Loss, but sub-optimal due to unstable steps ($LR$ too high for Adam). |
| **#3** | Sigmoid | SGD | `0.01` | 78.24% | Sluggish convergence caused by standard vanishing/saturating gradient effects. |
| **#4** | Sigmoid | Adam | `0.01` | 82.27% | Adam's adaptive momentum helps counteract Sigmoid's gradient flattening. |
| **#5** | ReLU | SGD | `0.001` | 76.24% | Critically underfit. The $LR$ is too small for standard SGD momentum. |
| **#6** | **ReLU** | **Adam** | `0.001` | **86.57%** | **Best Configuration.** Optimal step sizing via adaptive weight tracking. |
| **#7** | Sigmoid | SGD | `0.001` | 66.73% | Complete failure to converge. Severe gradient vanishing combined with weak steps. |
| **#8** | Sigmoid | Adam | `0.001` | 86.10% | Highly impressive recovery. Adam entirely resolves Sigmoid's standard training stagnation. |

### Architectural Takeaways
* **The Activation Shift:** `ReLU` wakes up nodes immediately because its derivative is always 1 for positive inputs, whereas `Sigmoid` squashes values, leading to vanishing gradients.
* **The Optimizer-LR Co-dependence:** Adam performs poorly at high $LR$ (`0.01`) but dominates at standard low ranges (`0.001`), thanks to individual parameters having dedicated, dynamically scaled learning schedules.

*Artifact Saved:* Best model checkpoint exported to disk as `model.pt`.

---

## 2. Zero-Shot Sentiment Pipeline & Context Limits

### Objective
Testing edge-case semantic extraction using the `distilbert-base-uncased-finetuned-sst-2-english` pipeline on declarative sentences versus complex sarcastic statements.

### Execution Results
1. *"I am very excited to join you all here."* $\rightarrow$ **POSITIVE** (99.98% Confidence)
2. *"Yesterday's debug session gave me an awful headache."* $\rightarrow$ **NEGATIVE** (99.96% Confidence)
3. *"I just love searching for a syntax error in my code ..."* $\rightarrow$ **NEGATIVE** (82.00% Confidence)

### Analytical Insight
The model successfully flags the sarcastic entry as `NEGATIVE` with 82% confidence. Because Transformers utilize bidirectional attention matrices rather than scanning text linearly, the token `"love"` is heavily contextualized and weighed down by the downstream contextual string `"searching for a syntax error"`, flipping the categorical output score.

---

## 3. News Summarization Architecture Constraints

### Objective
Scraping production data using `newspaper3k` and overriding the standard Hugging Face pipeline wrappers to deploy `facebook/bart-large-cnn` through manual Seq2Seq tensor decoding.

### Benchmark Analysis: Short vs. Long Configurations

* **Short Summary (Article 1):** Configured via constrained `max_new_tokens=60` and length suppression (`length_penalty=0.5`).
  * *Result:* Condenses the entire heatwave article into a clean, standalone sentence summary.
* **Long Summary (Articles 2 & 3):** Configured via forced bounds (`min_new_tokens=90`, `max_new_tokens=200`, `length_penalty=2.0`).
  * *Result:* Forces the decoder to expand its generation loop.

---

## 4. Subword Tokenization Demo

### Objective
Inspecting the core processing mechanics of the structural tokenizer wrapper `bert-base-uncased`.

### Ingestion Output
* **Input Text:** `"Tokenization is fast and easy!"`
* **Subword Tokens:** `['token', '##ization', 'is', 'fast', 'and', 'easy', '!']`
* **Numerical Input IDs:** `[101, 19204, 3989, 2003, 3435, 1998, 3733, 999, 102]`
* **Attention Mask Array:** `[1, 1, 1, 1, 1, 1, 1, 1, 1]`

### Technical Takeaways
1. **The Out-Of-Vocabulary (OOV) Solution:** The word `"Tokenization"` was broken down into `['token', '##ization']`. The prefix `##` marks a subword suffix link, allowing the system to comprehend unseen words dynamically without breaking the vocabulary cap.
2. **Special Tokens Framing:** `101` maps to the `[CLS]` token (denoting sequence classification initialization), and `102` maps to the `[SEP]` boundary token.

---

## 5. Mini-Project: IMDb Movie Review Sentiment Classifier Fine-Tuning

### Objective
Mounting an operational binary classification head onto raw `distilbert-base-uncased` checkpoints using Hugging Face's native `Trainer` API.

### Pipeline Customizations
* **Data Limits:** Downsampled to 1,000 training records and 200 validation records to enable fast CPU iterations.
* **Truncation Boundaries:** Tokenized sequence limits hard-bounded to `max_length=64` to dramatically reduce matrix processing overhead in RAM.

### Model Evaluation Performance
* **Total Training Epochs:** 2
* **Loss Optimization Track:** Steady decline over training cycles.
* **Final Out-Of-Sample Validation Accuracy:** **75.00%**

### Post-Fine-Tuning Out-Of-Sample Tests
The fine-tuned model was immediately exposed to fresh, un-tokenized test reviews via a custom evaluation pipeline:

* *"This movie was an absolute masterpiece, the acting was phenomenal!"* $\rightarrow$ **POSITIVE** (88.70% Confidence)
* *"I fell asleep after 20 minutes. Total waste of time and money."* $\rightarrow$ **NEGATIVE** (89.62% Confidence)
* *"The visual effects were stunning, but the story was incredibly boring and predictable."* $\rightarrow$ **NEGATIVE** (88.79% Confidence)

### Core Analysis
The third review showcases true structural comprehension. Despite the heavily weighted target adjective `"stunning"` at the start, the model successfully registers the contextual pivot introducing the negation (`"but the story was incredibly boring..."`), classifying the entire string as `NEGATIVE`.

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/ShahJahan-del/ai-intern-week03-deep-learning.git](https://github.com/ShahJahan-del/ai-intern-week03-deep-learning.git)
   cd ai-intern-week03-deep-learning
   jupyter notebook