from v2_upgrade.src.data.load import load_raw, basic_clean
from v2_upgrade.src.features.engineering import feature_engineer
from v2_upgrade.src.config import DATA_PROCESSED

def main():
    df = load_raw()
    df = basic_clean(df)
    df = feature_engineer(df)

    out = DATA_PROCESSED / "telco_engineered.csv"
    df.to_csv(out, index=False)
    print(f"[OK] saved: {out}")
    print(f"[INFO] shape: {df.shape}")
    print(f"[INFO] cols: {len(df.columns)}")

if __name__ == "__main__":
    main()
