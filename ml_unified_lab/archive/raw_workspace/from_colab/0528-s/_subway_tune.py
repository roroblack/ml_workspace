import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.linear_model import RidgeCV, ElasticNetCV, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
try:
    import xgboost as xgb
    HAS_XGB = True
except:
    HAS_XGB = False
try:
    import lightgbm as lgb
    HAS_LGB = True
except:
    HAS_LGB = False

# ===== 데이터 로드 및 피처 엔지니어링 (기존 코드) =====
def get_advanced_features(df, train_mappings=None):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['actual_dow'] = df['date'].dt.dayofweek
    df['month']      = df['date'].dt.month
    df['day']        = df['date'].dt.day
    df['year']       = df['date'].dt.year

    for col in ['visibility', 'precipitation', 'temperature']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].mean())
    df['station_name'] = df['station_name'].fillna('unknown')

    if train_mappings is None:
        m = {}
        m['stat_avg']         = df.groupby('station_name')['num_people'].mean().to_dict()
        m['dow_mapping']      = df.groupby(['station_name', 'actual_dow'])['num_people'].mean().to_dict()
        m['month_mapping']    = df.groupby(['station_name', 'month'])['num_people'].mean().to_dict()
        m['dow_month_map']    = df.groupby(['station_name', 'actual_dow', 'month'])['num_people'].mean().to_dict()
        m['year_station_map'] = df.groupby(['station_name', 'year'])['num_people'].mean().to_dict()
        m['year_month_map']   = df.groupby(['station_name', 'year', 'month'])['num_people'].mean().to_dict()
        m['year_dow_map']     = df.groupby(['station_name', 'year', 'actual_dow'])['num_people'].mean().to_dict()
    else:
        m = train_mappings

    def safe_get(key, d, fallback_key=None, fallback_d=None, global_val=12000):
        v = d.get(key)
        if v is None and fallback_d is not None:
            v = fallback_d.get(fallback_key)
        return v if v is not None else global_val

    global_mean = np.mean(list(m['stat_avg'].values()))
    df['stat_dow_avg']     = df.apply(lambda r: safe_get((r['station_name'], r['actual_dow']), m['dow_mapping'], r['station_name'], m['stat_avg'], global_mean), axis=1)
    df['stat_month_avg']   = df.apply(lambda r: safe_get((r['station_name'], r['month']), m['month_mapping'], r['station_name'], m['stat_avg'], global_mean), axis=1)
    df['stat_dow_month_avg'] = df.apply(lambda r: safe_get((r['station_name'], r['actual_dow'], r['month']), m['dow_month_map'], (r['station_name'], r['actual_dow']), m['dow_mapping'], global_mean), axis=1)
    df['year_station_avg'] = df.apply(lambda r: safe_get((r['station_name'], r['year']), m['year_station_map'], r['station_name'], m['stat_avg'], global_mean), axis=1)
    df['year_month_avg']   = df.apply(lambda r: safe_get((r['station_name'], r['year'], r['month']), m['year_month_map'], (r['station_name'], r['year']), m['year_station_map'], global_mean), axis=1)
    df['year_dow_avg']     = df.apply(lambda r: safe_get((r['station_name'], r['year'], r['actual_dow']), m['year_dow_map'], (r['station_name'], r['year']), m['year_station_map'], global_mean), axis=1)
    return df, m

print("데이터 로드 및 피처 엔지니어링...")
train_df = pd.read_csv("../0528_data/subway/subway_train.csv")
test_df  = pd.read_csv("../0528_data/subway/subway_test.csv")
train_df, train_mappings = get_advanced_features(train_df)
test_df, _ = get_advanced_features(test_df, train_mappings)

# ===== 인코딩 및 스케일링 =====
le = LabelEncoder()
train_df['station_idx'] = le.fit_transform(train_df['station_name'])
test_df['station_idx']  = test_df['station_name'].apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)

CONT_COLS = ['visibility', 'precipitation', 'temperature', 'day', 'stat_dow_avg', 'stat_month_avg', 'year_station_avg', 'year_month_avg', 'year_dow_avg']
OHE_COLS = ['station_idx', 'actual_dow', 'month', 'year']

scaler_x = StandardScaler()
X_cont_tr = scaler_x.fit_transform(train_df[CONT_COLS])
X_cont_te = scaler_x.transform(test_df[CONT_COLS])

ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat_tr = ohe.fit_transform(train_df[OHE_COLS])
X_cat_te = ohe.transform(test_df[OHE_COLS])

X_all = np.hstack([X_cont_tr, X_cat_tr])
X_test_all = np.hstack([X_cont_te, X_cat_te])
y_all = np.log1p(train_df['num_people'].values)
y_test = test_df['num_people'].values

print(f"데이터 크기: train={X_all.shape}, test={X_test_all.shape}")

# ===== 하이퍼파라미터 탐색 =====
results = []

# 1. RandomForest 탐색
print("\n[1] RandomForest 탐색 중...")
for depth in [10, 15, 20]:
    for n_est in [200, 300, 500]:
        rf = RandomForestRegressor(n_estimators=n_est, max_depth=depth, min_samples_leaf=3, random_state=42, n_jobs=-1)
        rf.fit(X_all, y_all)
        y_pred = np.expm1(rf.predict(X_test_all))
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        results.append((r2, 'RF', f"depth={depth} n_est={n_est}", rmse, rf, y_pred))
        print(f"  RF: depth={depth:2d} n_est={n_est:3d} → R²={r2:.4f} RMSE={rmse:.2f}")

# 2. GradientBoosting 탐색
print("\n[2] GradientBoosting 탐색 중...")
for depth in [4, 6, 8]:
    for lr in [0.01, 0.05, 0.1]:
        gb = GradientBoostingRegressor(n_estimators=500, max_depth=depth, learning_rate=lr, min_samples_leaf=3, random_state=42, subsample=0.8)
        gb.fit(X_all, y_all)
        y_pred = np.expm1(gb.predict(X_test_all))
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        results.append((r2, 'GB', f"depth={depth} lr={lr}", rmse, gb, y_pred))
        print(f"  GB: depth={depth} lr={lr:.2f} → R²={r2:.4f} RMSE={rmse:.2f}")

# 3. HistGradientBoosting 탐색
print("\n[3] HistGradientBoosting 탐색 중...")
for depth in [5, 7, 10]:
    for lr in [0.01, 0.05]:
        hgb = HistGradientBoostingRegressor(max_depth=depth, learning_rate=lr, max_iter=500, l2_regularization=1.0, min_samples_leaf=3, random_state=42)
        hgb.fit(X_all, y_all)
        y_pred = np.expm1(hgb.predict(X_test_all))
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        results.append((r2, 'HGB', f"depth={depth} lr={lr}", rmse, hgb, y_pred))
        print(f"  HGB: depth={depth} lr={lr:.2f} → R²={r2:.4f} RMSE={rmse:.2f}")

# 4. XGBoost
if HAS_XGB:
    print("\n[4] XGBoost 탐색 중...")
    for depth in [5, 7, 10]:
        for lr in [0.01, 0.05, 0.1]:
            xgb_m = xgb.XGBRegressor(n_estimators=500, max_depth=depth, learning_rate=lr, min_child_weight=1, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
            xgb_m.fit(X_all, y_all)
            y_pred = np.expm1(xgb_m.predict(X_test_all))
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            results.append((r2, 'XGB', f"depth={depth} lr={lr}", rmse, xgb_m, y_pred))
            print(f"  XGB: depth={depth} lr={lr:.2f} → R²={r2:.4f} RMSE={rmse:.2f}")

# 5. LightGBM
if HAS_LGB:
    print("\n[5] LightGBM 탐색 중...")
    for depth in [5, 7, 10]:
        for lr in [0.01, 0.05]:
            lgb_m = lgb.LGBMRegressor(n_estimators=500, max_depth=depth, learning_rate=lr, num_leaves=50, min_data_in_leaf=3, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
            lgb_m.fit(X_all, y_all)
            y_pred = np.expm1(lgb_m.predict(X_test_all))
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            results.append((r2, 'LGB', f"depth={depth} lr={lr}", rmse, lgb_m, y_pred))
            print(f"  LGB: depth={depth} lr={lr:.2f} → R²={r2:.4f} RMSE={rmse:.2f}")

# 상위 5개 모델
results.sort(key=lambda x: x[0], reverse=True)
print("\n" + "="*80)
print("TOP 5 개별 모델:")
print("="*80)
for i, (r2, mtype, params, rmse, model, pred) in enumerate(results[:5], 1):
    print(f"{i}. {mtype:5s} ({params:30s}): R²={r2:.4f}, RMSE={rmse:.2f}")

# 앙상블: TOP 3 모델
print("\n" + "="*80)
print("최적 앙상블 탐색 (TOP 5에서):")
print("="*80)
top5 = results[:5]
best_ensemble_r2 = -np.inf
best_weights = None
best_pred = None

for i in range(len(top5)):
    for j in range(i+1, len(top5)):
        for k in range(j+1, len(top5)):
            r2_i, _, _, _, _, pred_i = top5[i]
            r2_j, _, _, _, _, pred_j = top5[j]
            r2_k, _, _, _, _, pred_k = top5[k]

            for w1 in np.arange(0.3, 0.7, 0.1):
                for w2 in np.arange(0.1, 0.5, 0.1):
                    w3 = 1 - w1 - w2
                    if w3 < 0 or w3 > 0.5:
                        continue
                    y_ensemble = w1*pred_i + w2*pred_j + w3*pred_k
                    r2_ens = r2_score(y_test, y_ensemble)
                    if r2_ens > best_ensemble_r2:
                        best_ensemble_r2 = r2_ens
                        best_weights = (w1, w2, w3)
                        best_pred = y_ensemble
                        best_combo = (i+1, j+1, k+1, top5[i][1], top5[j][1], top5[k][1])

rmse_ens = np.sqrt(mean_squared_error(y_test, best_pred))
print(f"\n최적 3-모델 앙상블:")
print(f"  모델: {best_combo[3]} (#{best_combo[0]}) + {best_combo[4]} (#{best_combo[1]}) + {best_combo[5]} (#{best_combo[2]})")
print(f"  가중치: {best_weights[0]:.2f} + {best_weights[1]:.2f} + {best_weights[2]:.2f}")
print(f"  R² = {best_ensemble_r2:.4f}")
print(f"  RMSE = {rmse_ens:.2f}")
