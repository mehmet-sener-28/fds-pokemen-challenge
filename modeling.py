import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, log_loss
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier

from .config import COL_TARGET, COL_ID, SEED


def prepare_feature_sets(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Orijinal kodundaki:
      - high correlation removal
      - SelectFromModel(LogReg + XGB) ile feature seçimi
      - X_train_log, X_test_log, X_train_xgb, X_test_xgb, y_train
    üreten kısmı burada toplar.
    """

    print("FEATURE SELECTION & PRUNING CON GRIDSEARCH")

    # === 0) Başlangıç feature listesi ===
    # train_df şu anda: [COL_ID, COL_TARGET, ...tüm engineered feature kolonları...]
    # Özellik olarak: target ve id HARİÇ tüm kolonları alıyoruz.
    base_feature_cols = [
        c for c in train_df.columns
        if c not in (COL_TARGET, COL_ID)
    ]

    # Temel X ve y
    y_train = train_df[COL_TARGET].astype(int)
    X_features = train_df[base_feature_cols]

    # === Step 0: High correlation removal (|corr| > 0.9) ===
    print("\n Step 0: Removing highly correlated features (|corr| > 0.9)...")

    correlation_matrix = X_features.corr()
    high_corr_threshold = 0.9
    features_to_remove = set()
    high_corr_pairs = []

    # Önce bir kez LogisticRegression ile önemleri hesaplıyoruz
    corr_scaler = StandardScaler()
    X_scaled_for_corr = corr_scaler.fit_transform(X_features)

    corr_log = LogisticRegression(
        penalty='l2',
        solver='saga',
        C=0.1,
        random_state=SEED,
        max_iter=1000,
    )
    corr_log.fit(X_scaled_for_corr, y_train)
    corr_importance = dict(
        zip(base_feature_cols, np.abs(corr_log.coef_[0]))
    )

    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            corr_ij = correlation_matrix.iloc[i, j]
            if abs(corr_ij) > high_corr_threshold:
                feat_1 = correlation_matrix.columns[i]
                feat_2 = correlation_matrix.columns[j]

                coef_1 = corr_importance.get(feat_1, 0.0)
                coef_2 = corr_importance.get(feat_2, 0.0)

                # Rapor için sakla (opsiyonel)
                high_corr_pairs.append({
                    'feature_1': feat_1,
                    'feature_2': feat_2,
                    'correlation': corr_ij,
                    'removed': feat_2 if coef_1 >= coef_2 else feat_1,
                    'kept': feat_1 if coef_1 >= coef_2 else feat_2,
                })

                if coef_1 >= coef_2:
                    features_to_remove.add(feat_2)
                    print(
                        f"Remove {feat_2} (corr={corr_ij:.3f} with {feat_1}, "
                        f"coef={coef_2:.4f} < {coef_1:.4f})"
                    )
                else:
                    features_to_remove.add(feat_1)
                    print(
                        f"Remove {feat_1} (corr={corr_ij:.3f} with {feat_2}, "
                        f"coef={coef_1:.4f} < {coef_2:.4f})"
                    )

    if high_corr_pairs:
        corr_report = pd.DataFrame(high_corr_pairs)
        corr_report.to_csv('high_correlation_report.csv', index=False)
        print("\n High correlation report saved: high_correlation_report.csv")

    base_feature_cols_cleaned = [
        f for f in base_feature_cols if f not in features_to_remove
    ]
    print(
        f"\n   Removed {len(features_to_remove)} features due to high correlation"
    )
    print(
        f"   Remaining: {len(base_feature_cols_cleaned)}/{len(base_feature_cols)} features"
    )

    # Temizlenmiş özellik matrisi
    X_train = train_df[base_feature_cols_cleaned].fillna(-1.0)
    X_test = test_df[base_feature_cols_cleaned].fillna(-1.0)

    # StandardScaler sadece LogReg SelectFromModel için
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    print("\n Optimizing feature selection thresholds via GridSearch...")

    cv_strategy = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=SEED
    )

    # === Step 1: LogReg - SelectFromModel GridSearch ===
    print(
        "\n Step 1: L1/L2/ElasticNet-based SelectFromModel (LogReg) - GridSearch"
    )
    log_selector_param_grid = {
        'penalty': ['l1', 'l2', 'elasticnet'],
        'C': [0.05, 0.1, 0.2],
        'threshold': ['mean', 'median', '0.5*mean', '0.75*mean', '1.25*mean'],
    }

    best_log_score = 0.0
    best_log_params = None

    for penalty in log_selector_param_grid['penalty']:
        for C in log_selector_param_grid['C']:
            for threshold in log_selector_param_grid['threshold']:
                try:
                    if penalty == 'elasticnet':
                        selector = SelectFromModel(
                            LogisticRegression(
                                penalty='elasticnet',
                                solver='saga',
                                C=C,
                                l1_ratio=0.5,
                                random_state=SEED,
                                max_iter=5000,
                            ),
                            threshold=threshold,
                        )
                    else:
                        selector = SelectFromModel(
                            LogisticRegression(
                                penalty=penalty,
                                solver='saga',
                                C=C,
                                random_state=SEED,
                                max_iter=5000,
                            ),
                            threshold=threshold,
                        )

                    selector.fit(X_train_scaled, y_train)
                    selected_features = [
                        f
                        for f, s in zip(
                            base_feature_cols_cleaned, selector.get_support()
                        )
                        if s
                    ]

                    if not selected_features:
                        continue

                    # hızlı CV scoring
                    if penalty == 'elasticnet':
                        temp_pipeline = Pipeline([
                            ('scaler', StandardScaler()),
                            ('clf', LogisticRegression(
                                penalty='elasticnet',
                                solver='saga',
                                C=0.1,
                                l1_ratio=0.5,
                                random_state=SEED,
                                max_iter=1000,
                            )),
                        ])
                    else:
                        temp_pipeline = Pipeline([
                            ('scaler', StandardScaler()),
                            ('clf', LogisticRegression(
                                penalty=penalty,
                                solver='saga',
                                C=0.1,
                                random_state=SEED,
                                max_iter=1000,
                            )),
                        ])

                    from sklearn.model_selection import cross_val_score

                    scores = cross_val_score(
                        temp_pipeline,
                        X_train[selected_features],
                        y_train,
                        cv=cv_strategy,
                        scoring='accuracy',
                        n_jobs=-1,
                    )
                    score = scores.mean()

                    if score > best_log_score:
                        best_log_score = score
                        best_log_params = {
                            'penalty': penalty,
                            'C': C,
                            'threshold': threshold,
                        }
                        print(
                            f"  ✓ New best: penalty={penalty}, C={C}, "
                            f"threshold={threshold}, "
                            f"n_features={len(selected_features)}, CV={score:.4f}"
                        )
                except Exception:
                    # Bazı kombinasyonlar solver ile uyumsuz olabilir, geç.
                    continue

    if best_log_params is None:
        # Fallback – hiçbiri çalışmazsa
        best_log_params = {
            'penalty': 'l2',
            'C': 0.1,
            'threshold': 'mean',
        }
        print("\n WARNING: No valid LogReg selector config found, using fallback.")

    if best_log_params['penalty'] == 'elasticnet':
        l2_selector = SelectFromModel(
            LogisticRegression(
                penalty='elasticnet',
                solver='saga',
                C=best_log_params['C'],
                l1_ratio=0.5,
                random_state=SEED,
                max_iter=5000,
            ),
            threshold=best_log_params['threshold'],
        )
    else:
        l2_selector = SelectFromModel(
            LogisticRegression(
                penalty=best_log_params['penalty'],
                solver='saga',
                C=best_log_params['C'],
                random_state=SEED,
                max_iter=5000,
            ),
            threshold=best_log_params['threshold'],
        )

    l2_selector.fit(X_train_scaled, y_train)
    l2_selected = [
        f
        for f, s in zip(base_feature_cols_cleaned, l2_selector.get_support())
        if s
    ]
    print(f"\n   Best LogReg: {best_log_params}")
    print(
        f"   Selected: {len(l2_selected)}/{len(base_feature_cols_cleaned)} features"
    )

    # === Step 2: XGBoost - SelectFromModel GridSearch ===
    print("\n Step 2: XGBoost-based SelectFromModel - GridSearch")

    xgb_selector_param_grid = {
        'n_estimators': [600, 800],
        'max_depth': [2, 3],
        'threshold': ['mean', 'median', '0.5*mean', '0.75*mean'],
    }

    best_xgb_score = 0.0
    best_xgb_params = None

    for n_est in xgb_selector_param_grid['n_estimators']:
        for max_d in xgb_selector_param_grid['max_depth']:
            for threshold in xgb_selector_param_grid['threshold']:
                try:
                    selector = SelectFromModel(
                        XGBClassifier(
                            n_estimators=n_est,
                            max_depth=max_d,
                            random_state=SEED,
                            eval_metric='logloss',
                        ),
                        threshold=threshold,
                    )
                    selector.fit(X_train, y_train)
                    selected_features = [
                        f
                        for f, s in zip(
                            base_feature_cols_cleaned, selector.get_support()
                        )
                        if s
                    ]

                    if len(selected_features) < 10:
                        continue

                    from sklearn.model_selection import cross_val_score

                    temp_model = XGBClassifier(
                        n_estimators=200,
                        max_depth=3,
                        random_state=SEED,
                        eval_metric='logloss',
                    )
                    scores = cross_val_score(
                        temp_model,
                        X_train[selected_features],
                        y_train,
                        cv=cv_strategy,
                        scoring='accuracy',
                        n_jobs=-1,
                    )
                    score = scores.mean()

                    if score > best_xgb_score:
                        best_xgb_score = score
                        best_xgb_params = {
                            'n_estimators': n_est,
                            'max_depth': max_d,
                            'threshold': threshold,
                        }
                        print(
                            f"  ✓ New best: n_est={n_est}, max_depth={max_d}, "
                            f"threshold={threshold}, "
                            f"n_features={len(selected_features)}, CV={score:.4f}"
                        )
                except Exception:
                    continue

    if best_xgb_params is None:
        best_xgb_params = {
            'n_estimators': 800,
            'max_depth': 3,
            'threshold': 'mean',
        }
        print("\n WARNING: No valid XGB selector config found, using fallback.")

    xgb_selector = SelectFromModel(
        XGBClassifier(
            n_estimators=best_xgb_params['n_estimators'],
            max_depth=best_xgb_params['max_depth'],
            random_state=SEED,
            eval_metric='logloss',
        ),
        threshold=best_xgb_params['threshold'],
    )
    xgb_selector.fit(X_train, y_train)
    xgb_selected = [
        f
        for f, s in zip(base_feature_cols_cleaned, xgb_selector.get_support())
        if s
    ]
    print(f"\n   Best XGBoost: {best_xgb_params}")
    print(
        f"   Selected: {len(xgb_selected)}/{len(base_feature_cols_cleaned)} features"
    )

    # === Step 3: Union & ayrık feature set'ler ===
    selected_base_features = list(set(l2_selected) | set(xgb_selected))
    final_feature_cols = selected_base_features

    print(f"\n Final feature set: {len(final_feature_cols)} features")

    log_features = [f for f in final_feature_cols if f in l2_selected]
    xgb_features = [f for f in final_feature_cols if f in xgb_selected]

    X_train_log = train_df[log_features].fillna(-1.0)
    X_test_log = test_df[log_features].fillna(-1.0)

    X_train_xgb = train_df[xgb_features].fillna(-1.0)
    X_test_xgb = test_df[xgb_features].fillna(-1.0)

    print("\n Feature sets per model (after optimized selection):")
    print(f"   LogReg: {len(log_features)} features")
    print(f"   XGBoost: {len(xgb_features)} features")

    return X_train_log, X_test_log, X_train_xgb, X_test_xgb, y_train


def train_models_and_ensemble(
    X_train_log,
    X_test_log,
    X_train_xgb,
    X_test_xgb,
    y_train,
):
    """
    Orijinal kodundan:
      - LogReg GridSearchCV
      - XGB GridSearchCV
      - 10-fold CV ile OOF tahminler (oof_log, oof_xgb, test_pred_log, test_pred_xgb)
      - alpha search ile blending (ENSEMBLE WEIGHT OPTIMIZATION)
      - stacking meta-model
    kısımlarını uygular.
    """
    print("GRID SEARCH & HYPERPARAMETER OPTIMIZATION")

    cv_strategy = StratifiedKFold(
        n_splits=10, shuffle=True, random_state=SEED
    )

    # 1. Logistic Regression GridSearch
    print("\n Optimizing Logistic Regression...")
    log_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            solver='saga',
            random_state=SEED,
            max_iter=5000,
        )),
    ])

    log_param_grid = {
        'clf__penalty': ['l1', 'l2', 'elasticnet'],
        'clf__C': [0.01, 0.1, 0.5, 1.0, 2.0],
        'clf__class_weight': [None, 'balanced'],
    }

    log_grid = GridSearchCV(
        log_pipeline,
        log_param_grid,
        cv=cv_strategy,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
    )
    log_grid.fit(X_train_log, y_train)

    print(f"\n   Best params (LogReg): {log_grid.best_params_}")
    print(f"   Best CV score (LogReg): {log_grid.best_score_:.4f}")

    # 2. XGBoost GridSearch
    print("\n Optimizing XGBoost...")
    xgb_param_grid = {
        'n_estimators': [800, 1000],
        'max_depth': [2, 3],
        'learning_rate': [0.02, 0.03],
        'subsample': [0.6, 0.7],
        'colsample_bytree': [0.6, 0.7],
        'min_child_weight': [60, 75],
        'gamma': [0.5],
        'reg_lambda': [5],
        'reg_alpha': [0.4],
    }

    xgb_grid = GridSearchCV(
        XGBClassifier(
            random_state=SEED,
            eval_metric='logloss',
            tree_method='hist',
        ),
        xgb_param_grid,
        cv=cv_strategy,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
    )
    xgb_grid.fit(X_train_xgb, y_train)

    print(f"\n   Best params (XGB): {xgb_grid.best_params_}")
    print(f"   Best CV score (XGB): {xgb_grid.best_score_:.4f}")

    # === 2) 10-fold CV + OOF predictions ===
    print("CROSS-VALIDATION WITH OPTIMIZED MODELS")

    NFOLDS = 10
    kfold = StratifiedKFold(
        n_splits=NFOLDS, shuffle=True, random_state=SEED
    )

    oof_log = np.zeros(len(X_train_log))
    oof_xgb = np.zeros(len(X_train_xgb))

    test_pred_log = np.zeros(len(X_test_log))
    test_pred_xgb = np.zeros(len(X_test_xgb))

    fold_results = []

    for fold, (tr_idx, va_idx) in enumerate(
        kfold.split(X_train_log, y_train), 1
    ):
        print(f"\n[FOLD {fold}]")

        # --- Logistic Regression fold ---
        X_tr_log, X_va_log = (
            X_train_log.iloc[tr_idx],
            X_train_log.iloc[va_idx],
        )
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

        best_log_kwargs = {
            k.replace('clf__', ''): v
            for k, v in log_grid.best_params_.items()
            if k.startswith('clf__')
        }

        log_fold = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                **best_log_kwargs,
                random_state=SEED,
                max_iter=5000,
                solver='saga',
            )),
        ])
        log_fold.fit(X_tr_log, y_tr)
        oof_log[va_idx] = log_fold.predict_proba(X_va_log)[:, 1]
        test_pred_log += (
            log_fold.predict_proba(X_test_log)[:, 1] / NFOLDS
        )

        # --- XGBoost fold ---
        X_tr_xgb, X_va_xgb = (
            X_train_xgb.iloc[tr_idx],
            X_train_xgb.iloc[va_idx],
        )

        xgb_fold = XGBClassifier(
            **xgb_grid.best_params_,
            random_state=SEED,
            eval_metric='logloss',
        )
        xgb_fold.fit(X_tr_xgb, y_tr)
        oof_xgb[va_idx] = xgb_fold.predict_proba(X_va_xgb)[:, 1]
        test_pred_xgb += (
            xgb_fold.predict_proba(X_test_xgb)[:, 1] / NFOLDS
        )

        # --- Fold metrics ---
        acc_log = accuracy_score(
            y_va, (oof_log[va_idx] >= 0.5).astype(int)
        )
        acc_xgb = accuracy_score(
            y_va, (oof_xgb[va_idx] >= 0.5).astype(int)
        )

        ll_log = log_loss(y_va, oof_log[va_idx])
        ll_xgb = log_loss(y_va, oof_xgb[va_idx])

        fold_results.append({
            'fold': fold,
            'log_acc': acc_log,
            'xgb_acc': acc_xgb,
            'log_ll': ll_log,
            'xgb_ll': ll_xgb,
        })

        print(
            f"   LOG: ACC={acc_log:.4f}, LogLoss={ll_log:.4f}\n"
            f"   XGB: ACC={acc_xgb:.4f}, LogLoss={ll_xgb:.4f}"
        )

    # === OOF metrics ===
    print("\n" + "=" * 70)
    print("OVERALL OUT-OF-FOLD METRICS")
    print("=" * 70)

    acc_log_oof = accuracy_score(
        y_train, (oof_log >= 0.5).astype(int)
    )
    ll_log_oof = log_loss(y_train, oof_log)
    print(f"\n[OOF] LOG   ACC={acc_log_oof:.4f}  LOGLOSS={ll_log_oof:.4f}")

    acc_xgb_oof = accuracy_score(
        y_train, (oof_xgb >= 0.5).astype(int)
    )
    ll_xgb_oof = log_loss(y_train, oof_xgb)
    print(f"[OOF] XGB   ACC={acc_xgb_oof:.4f}  LOGLOSS={ll_xgb_oof:.4f}")

    # === 3) Ensemble weight optimization ===
    print("ENSEMBLE WEIGHT OPTIMIZATION")

    best_alpha = 0.6
    best_score = 0.0

    print("\nSearching for optimal blend weight...")
    for alpha in np.arange(0.2, 0.8, 0.02):
        oof_blend = alpha * oof_log + (1.0 - alpha) * oof_xgb
        preds = (oof_blend >= 0.5).astype(int)
        score = accuracy_score(y_train, preds)

        if score > best_score:
            best_score = score
            best_alpha = alpha
            print(
                f"  ✓ New best: Alpha={alpha:.2f}, Accuracy={score:.4f}"
            )

    BLEND_W_LOG = best_alpha
    BLEND_W_XGB = 1.0 - best_alpha

    oof_blend = BLEND_W_LOG * oof_log + BLEND_W_XGB * oof_xgb
    test_pred = BLEND_W_LOG * test_pred_log + BLEND_W_XGB * test_pred_xgb

    acc_b = accuracy_score(
        y_train, (oof_blend >= 0.5).astype(int)
    )
    ll_b = log_loss(y_train, oof_blend)

    print(
        f"FINAL ENSEMBLE: LOG={BLEND_W_LOG:.2f}, XGB={BLEND_W_XGB:.2f}"
    )
    print(f"[OOF] ENS   ACC={acc_b:.4f}  LOGLOSS={ll_b:.4f}")

    # === 4) Stacking meta-model ===
    print("\n" + "=" * 70)
    print("STACKING ENSEMBLE")
    print("=" * 70)

    X_meta_train = np.column_stack([oof_log, oof_xgb])
    print(f"Meta-features shape: {X_meta_train.shape}")
    print(
        f"  - oof_log: mean={oof_log.mean():.3f}, std={oof_log.std():.3f}"
    )
    print(
        f"  - oof_xgb: mean={oof_xgb.mean():.3f}, std={oof_xgb.std():.3f}"
    )

    meta_model = LogisticRegression(
        C=1.0, random_state=SEED, max_iter=1000
    )
    meta_model.fit(X_meta_train, y_train)

    print("\nMeta-model learned weights (coefficients):")
    print(f"  LogReg coefficient: {meta_model.coef_[0][0]:+.4f}")
    print(f"  XGBoost coefficient: {meta_model.coef_[0][1]:+.4f}")
    print(f"  Intercept: {meta_model.intercept_[0]:+.4f}")

    meta_pred_train = meta_model.predict_proba(X_meta_train)[:, 1]

    acc_meta = accuracy_score(
        y_train, (meta_pred_train >= 0.5).astype(int)
    )
    ll_meta = log_loss(y_train, meta_pred_train)

    print(
        f"\n[TRAIN] STACKING  ACC={acc_meta:.4f}  LOGLOSS={ll_meta:.4f}"
    )

    X_meta_test = np.column_stack([test_pred_log, test_pred_xgb])
    test_pred_stack = meta_model.predict_proba(X_meta_test)[:, 1]

    print("\n" + "=" * 70)
    print("COMPARISON: Weighted Average vs Stacking")
    print("=" * 70)
    print(f"  Weighted Average: ACC={acc_b:.4f}, LogLoss={ll_b:.4f}")
    print(f"  Stacking:         ACC={acc_meta:.4f}, LogLoss={ll_meta:.4f}")
    print(
        f"  Improvement:      {acc_meta - acc_b:+.4f} accuracy"
    )

    if acc_meta > acc_b:
        print("\n Stacking is better! Use this for final submission.")
        final_test_pred = test_pred_stack
    else:
        print("\n  Weighted average is better. Stick with it.")
        final_test_pred = test_pred

    # Fonksiyon: tüm test tahminlerini geri döndürür
    return test_pred_log, test_pred_xgb, test_pred_stack, final_test_pred
