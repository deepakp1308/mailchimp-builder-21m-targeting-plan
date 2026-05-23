"""
Phase 3b: Propensity models using HistGradientBoosting + Optuna HPO.
5 models: P(completion), P(free->paid), P(upgrade), P(churn), P(activation)
Falls back to sklearn HistGradientBoosting if LightGBM is unavailable.
"""
import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression
from sklearn.inspection import permutation_importance
import warnings, os, sys, json
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "customers_engineered.parquet")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("PHASE 3b: Propensity Models (HistGBM + Optuna)")
print("=" * 60)

df = pd.read_parquet(DATA_PATH)
print(f"Loaded {len(df):,} customers")

# --- Define targets ---
df['target_completion'] = (df['email_completion_rate'] >= 0.5).astype(int)
df['target_upgrade'] = (df['mrr_status'] == 'expansion').astype(int)
df['target_churn'] = df['churned'] if 'churned' in df.columns else (df['mrr_status'] == 'churn').astype(int)
df['target_activation'] = ((df['email_creates_90d'] > 0) & (df['email_publishes_90d'] > 0)).astype(int)
df['target_free_to_paid'] = 0

# --- Feature columns ---
FEATURE_COLS = [
    'avg_mrr', 'plan_amount', 'tenure_months', 'list_size', 'subscribed_size',
    'list_count', 'email_creates_90d', 'email_publishes_90d', 'email_tests_90d',
    'ca_template_creates_90d', 'ai_content_creates_90d',
    'sms_creates_90d', 'automation_creates_90d',
    'builder_active_days_90d', 'total_builder_events_90d',
    'email_creates_30d', 'email_publishes_30d', 'total_builder_events_30d',
    'creates_trend', 'publishes_trend', 'events_trend',
    'avg_open_rate', 'avg_click_rate', 'avg_bounce_rate',
    'avg_health_score', 'avg_engagement_score', 'avg_deliverability_score',
    'friction_score', 'email_completion_rate', 'email_abandonment_rate',
    'total_delivered_3mo', 'total_revenue_3mo', 'total_orders_3mo',
]

available_features = [c for c in FEATURE_COLS if c in df.columns]

# Time-based split: sort by tenure (newer = test)
df_sorted = df.sort_values('tenure_months', ascending=False).reset_index(drop=True)
X_all = df_sorted[available_features].fillna(0).values
split_idx = int(len(df_sorted) * 0.7)
X_train, X_test = X_all[:split_idx], X_all[split_idx:]

MODELS_CONFIG = {
    'completion': {'target': 'target_completion', 'desc': 'P(email completion)'},
    'upgrade': {'target': 'target_upgrade', 'desc': 'P(upgrade/expansion)'},
    'churn': {'target': 'target_churn', 'desc': 'P(churn)'},
    'activation': {'target': 'target_activation', 'desc': 'P(builder activation)'},
    'free_to_paid': {'target': 'target_free_to_paid', 'desc': 'P(free→paid)'},
}

N_TRIALS = 50
model_results = {}
all_scores = {}
feature_importances = {}


def objective(trial, X_tr, y_tr, X_val, y_val):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 15, 127),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 100),
        'l2_regularization': trial.suggest_float('l2_regularization', 1e-8, 10.0, log=True),
        'max_iter': trial.suggest_int('max_iter', 100, 500),
        'random_state': 42,
        'early_stopping': True,
        'validation_fraction': 0.15,
        'n_iter_no_change': 15,
    }

    model = HistGradientBoostingClassifier(**params)
    model.fit(X_tr, y_tr)
    preds = model.predict_proba(X_val)[:, 1]

    if len(np.unique(y_val)) < 2:
        return 0.5

    return roc_auc_score(y_val, preds)


for model_name, config in MODELS_CONFIG.items():
    target_col = config['target']
    print(f"\n{'─' * 50}")
    print(f"Training: {config['desc']} ({model_name})")
    print(f"{'─' * 50}")

    y_all = df_sorted[target_col].fillna(0).astype(int).values
    y_train, y_test = y_all[:split_idx], y_all[split_idx:]

    pos_rate = y_train.mean()
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,} | Positive rate: {pos_rate:.3%}")

    if pos_rate < 0.001 or pos_rate > 0.999:
        print(f"  SKIPPING — target is near-constant (pos_rate={pos_rate:.4f})")
        all_scores[f'p_{model_name}'] = np.full(len(df_sorted), pos_rate)
        model_results[model_name] = {'auc': 0.5, 'status': 'skipped', 'positive_rate': round(float(pos_rate), 4)}
        continue

    # Optuna HPO
    X_tr_opt, X_val_opt, y_tr_opt, y_val_opt = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        lambda trial: objective(trial, X_tr_opt, y_tr_opt, X_val_opt, y_val_opt),
        n_trials=N_TRIALS, show_progress_bar=False
    )

    best_params = study.best_params
    best_params['random_state'] = 42
    best_params['early_stopping'] = True
    best_params['validation_fraction'] = 0.15
    best_params['n_iter_no_change'] = 15
    print(f"  Best AUC (val): {study.best_value:.4f}")

    # Retrain on full training set
    final_model = HistGradientBoostingClassifier(**best_params)
    final_model.fit(X_train, y_train)

    train_preds = final_model.predict_proba(X_train)[:, 1]
    test_preds = final_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_preds)
    print(f"  Test AUC: {test_auc:.4f}")

    # Isotonic calibration
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    iso.fit(train_preds, y_train)
    calibrated_train = iso.predict(train_preds)
    calibrated_test = iso.predict(test_preds)

    full_preds = np.concatenate([calibrated_train, calibrated_test])
    all_scores[f'p_{model_name}'] = full_preds

    # Feature importance (permutation-based, on sample)
    sample_n = min(5000, len(X_test))
    perm_imp = permutation_importance(
        final_model, X_test[:sample_n], y_test[:sample_n],
        n_repeats=5, random_state=42, scoring='roc_auc'
    )
    imp_df = pd.DataFrame({
        'feature': available_features,
        'importance': perm_imp.importances_mean,
    }).sort_values('importance', ascending=False)
    feature_importances[model_name] = imp_df

    print(f"  Top 5 features (permutation importance):")
    for _, row in imp_df.head(5).iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")

    model_results[model_name] = {
        'auc': round(test_auc, 4),
        'positive_rate': round(float(pos_rate), 4),
        'status': 'trained',
    }

# --- Save scores ---
df_out = df_sorted.copy()
for col, vals in all_scores.items():
    df_out[col] = vals

df_out.to_parquet(DATA_PATH, index=False)
print(f"\nPropensity scores saved to parquet ({len(all_scores)} models)")

results_path = os.path.join(OUT_DIR, "model_results.json")
with open(results_path, 'w') as f:
    json.dump(model_results, f, indent=2)

for model_name, imp_df in feature_importances.items():
    imp_df.to_csv(os.path.join(OUT_DIR, f"shap_{model_name}.csv"), index=False)

print(f"\n{'Model Summary':^60}")
print("-" * 60)
for name, result in model_results.items():
    print(f"  {name:20s}: AUC={result['auc']:.4f}  ({result['status']})")
