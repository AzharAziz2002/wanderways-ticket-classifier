# Support Ticket Classifier for WanderWays Travel

Machine learning project that classifies customer support tickets into predefined categories so they can be routed to the right support team faster.

## Status
- [x] Task 1: Baseline logistic regression (this repo state)
- [ ] Task 2: Feature engineering and model comparison
- [ ] Task 3: FastAPI deployment on PythonAnywhere

## Dataset
Bitext Gen AI Chatbot Customer Support Dataset from Kaggle (26,872 tickets, 11 categories; 24,635 after dropping duplicates). Place the CSV at `data/tickets.csv` (not committed). Text column: `instruction`; label column: `category`.

## Approach
1. Load data with pandas, drop missing/duplicate rows
2. Clean text (lowercase, strip URLs/punctuation)
3. 80/20 stratified train/test split
4. TF-IDF (1-2 grams) + Logistic Regression pipeline
5. Evaluate accuracy and save with pickle

## Results
Test accuracy: **99.57%** (majority-class baseline: 22.08%).

**Leakage / memorisation check:** The TF-IDF vectorizer is inside a Pipeline, so it is fit on training text only. I also measured how similar each test ticket is to its nearest training ticket. Only 8.5% of test tickets are 0.95+ similar (median similarity 0.80). Accuracy on tickets below 0.7 similarity is about 99.0% (95.3% for the 85 least similar), so the score is not mainly driven by near-duplicates.

**Confidently wrong example:** "Thanks, great service!" was predicted as CONTACT (61%) instead of FEEDBACK. None of the 24,635 cleaned tickets in the dataset contain "thanks", "great" or "thank", so only "service" reached the classifier. Of the 533 tickets containing "service", 343 are CONTACT and 188 are FEEDBACK, so the model leans CONTACT. This is a data coverage problem (no praise wording in the data), not a class-balance problem (CONTACT is 8.1%, ACCOUNT is the largest at 22.1%). Several correct predictions also had low confidence (32-47%). Ideas for Task 2: character n-grams, and testing with tickets worded differently from the training set.

**Errors:** 21 of 4,927 test tickets were misclassified, mostly typos ("pyments", "onlinepayment") and very short tickets. Character n-grams should help in Task 2.

**Caveat:** The dataset is synthetic and templated, so real-world accuracy is likely lower. A grouped split and cross-validation will follow in Task 2.

## Run it
```bash
pip install -r requirements.txt
jupyter notebook 01_baseline.ipynb
```

## Files
- `01_baseline.ipynb` : notebook
- `ticket_classifier.pkl` : trained model
- `requirements.txt`

## Author
Azhar Aziz
