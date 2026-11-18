# Support Ticket Classifier for WanderWays Travel

Machine learning project that classifies customer support tickets into predefined categories so they can be routed to the right support team faster.

## Status
- [x] Task 1: Baseline logistic regression
- [x] Task 2: Feature engineering and model comparison
- [ ] Task 3: FastAPI deployment on PythonAnywhere

## Dataset
Bitext Gen AI Chatbot Customer Support Dataset from Kaggle (26,872 tickets, 11 categories; 24,635 after dropping duplicates). Place the CSV at `data/tickets.csv` (not committed). Text column: `instruction`; label column: `category`.

## Approach
1. Load data with pandas, drop missing/duplicate rows
2. Clean text (lowercase, strip URLs/punctuation)
3. 80/20 stratified train/test split
4. TF-IDF (1-2 grams) + Logistic Regression pipeline
5. Evaluate accuracy and save with pickle
6. Task 2: compare LogisticRegression, LinearSVC, MultinomialNB, RandomForest and a char n-gram variant with 5-fold CV (macro-F1), and save the best

## Results
Test accuracy: **99.57%** (majority-class baseline: 22.08%).

**Leakage / memorisation check:** The TF-IDF vectorizer is inside a Pipeline, so it is fit on training text only. I also measured how similar each test ticket is to its nearest training ticket. Only 8.5% of test tickets are 0.95+ similar (median similarity 0.80). Accuracy on tickets below 0.7 similarity is about 99.0% (95.3% for the 85 least similar), so the score is not mainly driven by near-duplicates.

**Confidently wrong example:** "Thanks, great service!" was predicted as CONTACT (61%) instead of FEEDBACK. None of the 24,635 cleaned tickets in the dataset contain "thanks", "great" or "thank", so only "service" reached the classifier. Of the 533 tickets containing "service", 343 are CONTACT and 188 are FEEDBACK, so the model leans CONTACT. This is a data coverage problem (no praise wording in the data), not a class-balance problem (CONTACT is 8.1%, ACCOUNT is the largest at 22.1%). Several correct predictions also had low confidence (32-47%). Ideas for Task 2: character n-grams, and testing with tickets worded differently from the training set.

**Errors:** 21 of 4,927 test tickets were misclassified, mostly typos ("pyments", "onlinepayment") and very short tickets. Character n-grams should help in Task 2.

**Caveat:** The dataset is synthetic and templated, so real-world accuracy is likely lower. A grouped split and cross-validation will follow in Task 2.

## Task 2: Model comparison (5-fold CV, macro-F1)

Models were compared with 5-fold stratified cross-validation on the training split only. Each pipeline has its own TF-IDF vectorizer, so every fold fits it on its own training part (no leakage). The test split was used once, for the chosen model.

| Model | Macro-F1 (mean) | Std |
|---|---|---|
| LinearSVC + char n-grams | 0.9999 | 0.0001 |
| LinearSVC | 0.9968 | 0.0004 |
| MultinomialNB | 0.9961 | 0.0005 |
| LogisticRegression (Task 1 baseline) | 0.9959 | 0.0007 |
| RandomForest | 0.9943 | 0.0007 |

**Best model:** LinearSVC with word + character n-gram TF-IDF, fitted on the full training split and saved as `best_classifier.pkl`. On the held-out test split it scores 99.96% accuracy and 0.9995 macro-F1. The confusion matrix is saved as `confusion_matrix.png`.

**Did character n-grams help?** Yes: about +0.003 macro-F1 over plain LinearSVC, more than the fold-to-fold std. They fixed all 4 of my hand-written typo tickets ("pyment", "reimbursment", "chnage my adress", "unsubcribe"); the baseline got 1 of 4.

**Remaining errors:** 2 of 4,927 test tickets, both predicted as DELIVERY (an ORDER ticket about the ETA of an order, and a PAYMENT ticket about a checkout error). Both were also baseline errors and are arguably ambiguous. "Thanks, great service!" is still predicted CONTACT instead of FEEDBACK because the data has no praise wording.

**Caveat:** the dataset is synthetic and templated (including its typos), so 99.96% is near the ceiling and some of the gain may come from near-duplicate matching. Real tickets will likely score lower. A grouped split by near-duplicate cluster is not done.

## Run it
```bash
pip install -r requirements.txt
jupyter notebook 01_baseline.ipynb
jupyter notebook 02_model_comparison.ipynb
```

## Files
- `01_baseline.ipynb` : Task 1 baseline notebook
- `02_model_comparison.ipynb` : Task 2 model comparison notebook
- `ticket_classifier.pkl` : Task 1 baseline model
- `best_classifier.pkl` : Task 2 best model
- `confusion_matrix.png` : confusion matrix of the best model
- `model_comparison.csv` / `model_comparison.png` : CV results table and chart
- `requirements.txt`

## Author
Azhar Aziz
