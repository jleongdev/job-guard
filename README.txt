Job Guard - FIT1061 H2 Comprehensive Project (extra material)
Student: Jerry Leong Jun Fai (37116541)

CONTENTS
- fake_job_postings.csv : the dataset used by the notebook
- README.txt            : this file

DATASET
Real/Fake Job Posting Prediction (Kaggle), based on the Employment Scam
Aegean Dataset (EMSCAD). 17,880 job postings, 866 labelled fake.
Licence: CC0 1.0 Universal (Public Domain)
Source: https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction

HOW TO RUN
1. Put fake_job_postings.csv in the same folder as
   H2_comprehensive_project_implementation.ipynb
2. Open the notebook in Jupyter.
3. Choose Kernel > Restart & Run All.

REQUIREMENTS
- Python 3 with numpy, pandas and matplotlib
  (exact versions are printed in the notebook's Setup cell)
- No other libraries are needed. Naive Bayes, the metrics, the weighted
  risk score and hill climbing are all hand-coded.

REPRODUCIBILITY
- The random seed is fixed at 42 in the Setup cell, and reset at the start
  of Experiment 2 so the random restarts give the same results every run.
- The 70/15/15 split is stratified and uses the same seed.
- The test set was used once for the official result. A second, clearly
  labelled test run (Experiment 3) appears after it.