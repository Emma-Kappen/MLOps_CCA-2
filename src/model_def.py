"""Single source of truth for the claim-classification model (TF-IDF text features -> gradient boosting).
Kept in its own module so train/evaluate/deploy all load the same pickled pipeline."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

def to_dense(X):
    return X.toarray()  # vocabulary is small (field-tagged tokens), so dense is cheap; HistGradientBoosting needs it

def build_pipeline():
    return Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 1), min_df=5)),
        ('dense', FunctionTransformer(to_dense, accept_sparse=True)),
        ('clf', HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=20,
            l2_regularization=1.0, class_weight='balanced',
            early_stopping=True, n_iter_no_change=20, random_state=0)),
    ])