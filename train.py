import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report, roc_auc_score
import lightgbm as lgbm
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

SEED = np.random.randint(1, 10000)
print(f"SEED: {SEED}\n")
np.random.seed(SEED)

# Загрузка
train = pd.read_csv('invest_train.csv')
print(f"Train: {train.shape}")
print(f"Target distribution:\n{train['accepted'].value_counts(normalize=True)}\n")

# Feature engineering
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

train = create_features(train)

# Encoding
cat_cols = ['risk_profile', 'marketing_channel', 'membership_tier', 'age_group']
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    train[col + '_enc'] = le.fit_transform(train[col].astype(str))
    encoders[col] = le

# Features
features = ['age', 'balance', 'offer_amount', 'previous_investments', 'responded_before',
            'age_balance_ratio', 'balance_per_age', 'balance_log', 'high_balance', 'balance_squared',
            'age_squared', 'offer_log', 'offer_to_balance_ratio', 'can_afford', 'offer_percentage_of_balance',
            'prev_inv_and_responded', 'experience_score', 'risk_numeric', 'high_risk_high_offer',
            'is_personal_channel', 'is_digital_channel', 'tier_numeric', 'premium_client',
            'wealth_score', 'engagement_score', 'young_high_balance', 'senior_platinum',
            'risk_profile_enc', 'marketing_channel_enc', 'membership_tier_enc', 'age_group_enc']

X = train[features]
y = train['accepted']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
print(f"Train: {X_train.shape}, Val: {X_val.shape}\n")

# Model
model = LGBMClassifier(n_estimators=1000, learning_rate=0.05, max_depth=7, num_leaves=31,
                       min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                       reg_alpha=0.1, reg_lambda=0.1, random_state=SEED, n_jobs=-1, verbose=-1)

model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
          callbacks=[lgbm.early_stopping(50), lgbm.log_evaluation(100)])

# Threshold optimization
y_pred_proba = model.predict_proba(X_val)[:, 1]
best_f1, best_thr = 0, 0.5
for thr in np.arange(0.3, 0.7, 0.01):
    f1 = f1_score(y_val, (y_pred_proba > thr).astype(int))
    if f1 > best_f1:
        best_f1, best_thr = f1, thr

y_pred = (y_pred_proba > best_thr).astype(int)
print(f"\nBest threshold: {best_thr:.3f}")
print(f"F1 Score: {best_f1:.4f}")
print(f"ROC AUC: {roc_auc_score(y_val, y_pred_proba):.4f}\n")
print(classification_report(y_val, y_pred))

# CV
cv = cross_val_score(model, X, y, cv=5, scoring='f1', n_jobs=-1)
print(f"\nCV F1: {cv.mean():.4f} ± {cv.std():.4f}\n")

# Save
pickle.dump(model, open(f'investment_model_seed_{SEED}.pkl', 'wb'))
pickle.dump(encoders, open(f'label_encoders_seed_{SEED}.pkl', 'wb'))

meta = {'seed': SEED, 'features': features, 'cat_features': cat_cols, 'best_threshold': best_thr,
        'val_f1': best_f1, 'cv_f1': cv.mean(), 'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
json.dump(meta, open(f'model_metadata_seed_{SEED}.json', 'w'), indent=2)

print(f"Model saved with SEED {SEED}")
print(f"Files: investment_model_seed_{SEED}.pkl, label_encoders_seed_{SEED}.pkl, model_metadata_seed_{SEED}.json")