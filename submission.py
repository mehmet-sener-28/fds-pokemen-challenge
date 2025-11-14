import pandas as pd
from .config import COL_ID, COL_TARGET

def save_submissions(test_df,
                     test_pred_log,
                     test_pred_xgb,
                     test_pred_stack,
                     prefix: str = ""):
    """
    Orijinal script’te en sonda yaptığın üç submission'ı burada fonksiyonlaştırıyoruz.
    """

    # Logistic
    sub_log = pd.DataFrame({
        COL_ID: test_df[COL_ID],
        COL_TARGET: (test_pred_log >= 0.5).astype(int)
    })
    file_log = f"{prefix}submission_logistic.csv"
    sub_log.to_csv(file_log, index=False)
    print(f"Saved: {file_log}  shape={sub_log.shape}")

    # XGBoost
    sub_xgb = pd.DataFrame({
        COL_ID: test_df[COL_ID],
        COL_TARGET: (test_pred_xgb >= 0.5).astype(int)
    })
    file_xgb = f"{prefix}submission_xgb.csv"
    sub_xgb.to_csv(file_xgb, index=False)
    print(f"Saved: {file_xgb}  shape={sub_xgb.shape}")

    # Ensemble (stacking / final)
    sub_final = pd.DataFrame({
        COL_ID: test_df[COL_ID],
        COL_TARGET: (test_pred_stack >= 0.5).astype(int)
    })
    file_final = f"{prefix}submission_final.csv"
    sub_final.to_csv(file_final, index=False)
    print(f"Saved: {file_final}  shape={sub_final.shape}")