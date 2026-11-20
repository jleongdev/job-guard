# 🛡️ Job Guard: Explainable Fake Job Posting Detector

> Paste in a job ad and get a verdict, a risk score, and the reasons behind the decision, so you can spot a scam **before** you share your IC number, bank details or a "processing fee".

Job Guard is an explainable AI system that flags suspicious job postings. It uses a **Naive Bayes text classifier built from scratch**, combines it with **red-flag signals** in a weighted risk model, and tunes that model with **hill climbing and random restarts**. It was trained and evaluated on **15,787 real-world job postings**.

Built for **FIT1061 Introduction to Artificial Intelligence** at Monash University Malaysia (High Distinction project).

---

## ✨ Highlights

- **Hand-coded ML, no ML libraries.** Multinomial Naive Bayes with Laplace smoothing and log-probabilities, plus my own precision, recall, F1 and confusion-matrix functions, each unit-tested against examples worked out by hand.
- **Explainable by design.** Every flag shows the top 5 words that pushed the posting towards "fake", plus the red flags it triggered.
- **Hybrid risk model.** A weighted score that combines P(fake) with three red flags: no company logo, no screening questions, and payment words.
- **Optimisation without gradients.** F1 changes in jumps, so I tuned the weights and threshold with hill climbing and 20 random restarts.
- **Fairness checked.** The system wrongly flags honest employers with and without a logo at almost the same rate, a **0.8-point gap** compared with **100 points** for a simple no-logo rule.
- **Rigorous evaluation.** 2,093 duplicates removed to prevent leakage, a stratified 70/15/15 split, and the test set kept untouched until the final run.

---

## 🔍 Demo

```python
check_posting(
    "URGENT!!! DATA ENTRY WORK FROM HOME",
    "Earn money fast! No experience needed. Weekly pay $2500. "
    "Contact hr.recruit2024@gmail.com or 555-018-7742 to start immediately!!!",
    has_logo=False, has_questions=False,
)
```

```
Verdict: ⚠️ NEEDS REVIEW
Risk score: 1.18 (threshold 0.49) | P(fake) from text: 90.9%
Signals found:
  - No company logo
  - No screening questions
  - Email address in the text (moves contact off the job site)
  - Phone number in the text (moves contact off the job site)
  - Most suspicious words: ['earn', 'entry', 'phonetoken', 'immediately', 'pay']
```

A normal software engineering posting comes back as **✅ Looks genuine**, with P(fake) = 0.0%.

---

## 🧠 How it works

```
Job posting (title + description + metadata)
        │
        ▼
 ┌──────────────────┐   lowercase, keep letters,
 │    Tokeniser     │   links/emails/phones → urltoken / emailtoken / phonetoken
 └──────────────────┘
        │
        ▼
 ┌──────────────────┐   P(fake | words), Laplace smoothing,
 │   Naive Bayes    │   log-probabilities, length normalisation
 └──────────────────┘
        │ P(fake)
        ▼
 ┌──────────────────┐   risk = w1·P(fake) + w2·no_logo
 │ Weighted risk    │          + w3·no_questions + w4·payment_term
 │     score        │   flag if risk ≥ threshold
 └──────────────────┘
        ▲
        │ weights + threshold
 ┌──────────────────┐
 │  Hill climbing   │   maximise validation F1, 20 random restarts
 └──────────────────┘
        │
        ▼
 Verdict + risk score + top 5 suspicious words + red flags
```

### 1. Text features
Titles and descriptions are tokenised into words over a **47,169-word vocabulary** built from the training set only. The dataset anonymises links, emails and phone numbers, so I turn each one into a single token. They turned out to be useful signals:

| Signal | Fake postings | Genuine postings |
|---|---|---|
| Link in text | 26.3% | 14.4% |
| Email in text | 5.5% | 2.8% |
| Phone number in text | 3.8% | 1.4% |

This fits a common scam pattern of moving applicants off the job site.

### 2. Naive Bayes, from scratch
`train_nb` learns P(word | fake) and P(word | genuine) with Laplace smoothing. `prob_fake` adds log-probabilities so long postings don't underflow. **Length normalisation** divides the evidence by the number of known words, so P(fake) reflects the average evidence per word.

### 3. Red flags

| Red flag | Fake postings | Genuine postings |
|---|---|---|
| No company logo | 67.3% | 18.3% |
| No screening questions | 69.9% | 50.5% |
| Mentions a payment word | 8.3% | 3.9% |

### 4. Hill climbing
A state is `(w1, w2, w3, w4, b)`. Each step tries ±0.05 on every value and moves to the best neighbour only if F1 improves. **20 random restarts** escape poor local maxima. One run from equal weights reached F1 0.494, and the best restart reached **0.515**.

---

## 📊 Results (untouched test set: 2,368 postings, 101 scams)

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Majority class (always genuine) | 0.000 | 0.000 | 0.000 |
| Keyword rule (15 scam phrases) | 0.125 | 0.020 | 0.034 |
| No-logo rule | 0.148 | 0.713 | 0.246 |
| Naive Bayes (threshold 0.5) | 0.523 | 0.228 | 0.317 |
| **Job Guard (official system)** | **0.500** | **0.238** | **0.322** |
| **Job Guard + length normalisation** | **0.427** | **0.347** | **0.383** |

- **Best F1 of every model**, beating all three rule-based baselines.
- **Length normalisation raised F1 by 19%** (0.322 → 0.383) and caught 46% more scams (24 → 35).
- **About 1 false alarm per 100 genuine postings**, compared with 413 false alarms for the no-logo rule.
- **Fairness:** the official system's false-alarm gap between employers with and without a logo is 0.8 points. The length-normalised version stays within the 10-point target at 6.1 points.

> Why not accuracy? Only about 4% of postings are fake, so a model that never flags anything still scores 95.7% accuracy while catching zero scams. Job Guard is judged on **recall, precision and F1** for the fake class.

---

## 🧪 Experiments

| # | Question | Result | Decision |
|---|---|---|---|
| 1 | Does removing rare words (min count 3 or 5) help? | Recall ↑ but F1 ↓ (0.497 → 0.366) | Keep the full vocabulary: rare words help identify genuine ads |
| 2 | Do random restarts beat a single hill-climbing run? | F1 0.494 → **0.515** | Adopted |
| 3 | Does length normalisation fix Naive Bayes overconfidence? | Postings with P(fake) near 0 or 1: 98.6% → **0%**. Validation F1 0.515 → **0.536** | Adopted |

All decision rules were set **before** running each experiment, and all tuning was done on the validation set only.

---

## 💡 Key insights

- **Explainability catches problems that metrics miss.** The top-word explanations showed the model was partly learning place names, which shaped the experiments that followed.
- **Naive Bayes is overconfident on long text.** Treating every word as independent evidence pushes probabilities to 0 or 1, and length normalisation fixes this.
- **Simple rules are easy to beat but hard to make fair.** The no-logo rule catches many scams but flags every honest employer without a logo.
- **There is a real trade-off between catching scams and treating small employers fairly**, and it needs to be measured, not assumed.

---

## 🗂️ Project structure

```
job-guard/
├── H2_comprehensive_project_implementation.ipynb   # full pipeline, experiments, report
├── images/                                         # figures and demo screenshots for this README
│   ├── demo.png
│   ├── test_comparison.png
│   └── hill_climbing.png
├── requirements.txt
└── README.md
```

---

## 🚀 Getting started

1. **Clone the repo**
   ```bash
   git clone https://github.com/<your-username>/job-guard.git
   cd job-guard
   ```
2. **Install dependencies**
   ```bash
   pip install numpy pandas matplotlib jupyter
   ```
   Tested with numpy 2.5.1, pandas 3.1.0 and matplotlib 3.11.1.
3. **Get the data.** Download `fake_job_postings.csv` from the [Real / Fake Job Posting Prediction dataset on Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) and place it in the project folder.
4. **Run the notebook**
   ```bash
   jupyter notebook H2_comprehensive_project_implementation.ipynb
   ```
   A fixed random seed (42) makes every result reproducible.

---

## 🛣️ Roadmap

- Remove place and company names before training, so the model learns scam patterns rather than sources
- Better text cleaning for joined words from bullet-point formatting
- K-fold cross-validation for more stable estimates
- Logistic regression trained with gradient descent, so words and red flags are learned together
- Testing on current Malaysian job ads

---

## ⚠️ Responsible use

Job Guard marks postings as **"needs review"**. It is not a final judgement. It was trained on an older, mostly US dataset, and scammers keep changing their wording, so always verify a company independently before sharing personal details or money.

---

## 📚 Dataset & acknowledgements

- **Data:** Real / Fake Job Posting Prediction (Kaggle, Shivam Bansal), based on the Employment Scam Aegean Dataset (EMSCAD) from the University of the Aegean's Laboratory of Information & Communication Systems Security. It contains 17,880 postings, reduced to 15,787 after removing duplicates.
- **Course:** FIT1061 Introduction to Artificial Intelligence, Monash University Malaysia.

---
