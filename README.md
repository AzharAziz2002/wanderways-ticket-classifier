# Support Ticket Classifier for WanderWays Travel

Machine learning project that classifies customer support tickets into predefined categories so they can be routed to the right support team faster.

## Status
- [x] Task 1: Baseline logistic regression (this repo state)
- [ ] Task 2: Feature engineering and model comparison
- [ ] Task 3: FastAPI deployment on PythonAnywhere

## Dataset
Public support ticket dataset from Kaggle (place the CSV in `data/`, not committed if large). Text column: `instruction`; label column: `category`.

## Approach
1. Load data with pandas, drop missing/duplicate rows
2. Clean text (lowercase, strip URLs/punctuation)
3. 80/20 stratified train/test split
4. TF-IDF (1-2 grams) + Logistic Regression pipeline
5. Evaluate accuracy and save with pickle

## Results
Test accuracy: **99.57%**

The dataset is synthetic and templated, so real-world accuracy is likely lower. This will be tested in Task 2 with cross-validation.

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
