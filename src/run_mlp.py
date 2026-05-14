"""Run MLP CV and generate submission — Task 9."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from models import load_data, cross_validate, build_mlp, build_feature_matrix, make_submission

print('Loading data...')
train_df, test_df = load_data()
print(f'Train: {train_df.shape}  Test: {test_df.shape}')

print('\n=== MLP Neural Network (5-fold CV) ===')
cv_mlp = cross_validate(build_mlp(), train_df)

print('\n=== Building full feature matrix ===')
X_train, X_test, y_train = build_feature_matrix(train_df, test_df)
print(f'X_train: {X_train.shape}  X_test: {X_test.shape}')

print('\n=== Saving submission ===')
make_submission(build_mlp(), X_train, y_train, X_test, test_df['id'], 'mlp_v1.csv')
print('Done.')
