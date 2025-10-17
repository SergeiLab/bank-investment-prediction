import pandas as pd
import numpy as np
import pickle
import json
import glob
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Load model
model_file = sorted(glob.glob('investment_model_seed_*.pkl'))[-1]
seed = int(model_file.split('_')[-1].replace('.pkl', ''))
print(f"Loading model with SEED {seed}")

model = pickle.load(open(model_file, 'rb'))
encoders = pickle.load(open(f'label_encoders_seed_{seed}.pkl', 'rb'))
meta = json.load(open(f'model_metadata_seed_{seed}.json', 'r'))

features = meta.get('features', meta.get('feature_columns', []))
best_thr = meta.get('best_threshold', 0.5)
val_f1 = meta.get('val_f1', meta.get('validation_f1', 0))
print(f"F1 on validation: {val_f1:.4f}, Threshold: {best_thr:.3f}\n")

# Load test
test = pd.read_csv('invest_test_public.csv')
print(f"Test: {test.shape}")
ids = test['customer_id'].copy()

# Features
def create_features(df):
    df = df.copy()
    df['age_balance_ratio'] = df['age'] / (df['balance'] + 1)
    df['balance_per_age'] = df['balance'] / (df['age'] + 1)
    df['balance_log'] = np.log1p(df['balance'])
    df['high_balance'] = (df['balance'] > df['balance'].median()).astype(int)
    df['balance_squared'] = df['balance'] ** 2
    df['age_squared'] = df['age'] ** 2
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 100], labels=['young', 'middle', 'senior', 'elderly'])
    df['offer_log'] = np.log1p(df['offer_amount'])
    df['offer_to_balance_ratio'] = df['offer_amount'] / (df['balance'] + 1)
    df['can_afford'] = (df['balance'] >= df['offer_amount']).astype(int)
    df['offer_percentage_of_balance'] = (df['offer_amount'] / (df['balance'] + 1)) * 100
    df['prev_inv_and_responded'] = df['previous_investments'] * df['responded_before']
    df['experience_score'] = df['previous_investments'] + df['responded_before']
    
    risk_map = {'low': 0, 'medium': 1, 'high': 2}
    df['risk_numeric'] = df['risk_profile'].map(risk_map)
    df['high_risk_high_offer'] = ((df['risk_profile'] == 'high') & (df['offer_amount'] > df['offer_amount'].median())).astype(int)
    df['is_personal_channel'] = df['marketing_channel'].isin(['phone', 'in_branch']).astype(int)
    df['is_digital_channel'] = df['marketing_channel'].isin(['email', 'sms']).astype(int)
    
    tier_map = {'standard': 0, 'gold': 1, 'platinum': 2}
    df['tier_numeric'] = df['membership_tier'].map(tier_map)
    df['premium_client'] = (df['membership_tier'] == 'platinum').astype(int)
    df['wealth_score'] = (df['balance'] * df['tier_numeric']) / (df['age'] + 1)
    df['engagement_score'] = (df['previous_investments'] * 2 + df['responded_before'] + df['tier_numeric'])
    df['young_high_balance'] = ((df['age'] < 35) & (df['balance'] > df['balance'].median())).astype(int)
    df['senior_platinum'] = ((df['age'] > 55) & (df['membership_tier'] == 'platinum')).astype(int)
    return df

test = create_features(test)

# Encoding
cat_features = meta.get('cat_features', meta.get('categorical_features', []))
for col in cat_features:
    if col in encoders:
        # Проверяем какое окончание использовалось
        if col + '_enc' in features:
            test[col + '_enc'] = encoders[col].transform(test[col].astype(str))
        elif col + '_encoded' in features:
            test[col + '_encoded'] = encoders[col].transform(test[col].astype(str))
        else:
            test[col + '_enc'] = encoders[col].transform(test[col].astype(str))

# Predict
X_test = test[features]
proba = model.predict_proba(X_test)[:, 1]
pred = (proba > best_thr).astype(int)

print(f"\nPredictions: {pd.Series(pred).value_counts().to_dict()}")

# Save
sub = pd.DataFrame({'customer_id': ids, 'accepted': pred})
out = f'result_seed_{seed}.csv'
sub.to_csv(out, index=False)

print(f"\nSaved: {out}")
print(f"Accepted: {(pred == 1).sum()} ({(pred == 1).sum() / len(pred) * 100:.1f}%)")
print(f"Rejected: {(pred == 0).sum()} ({(pred == 0).sum() / len(pred) * 100:.1f}%)")