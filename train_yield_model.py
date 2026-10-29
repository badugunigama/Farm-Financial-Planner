"""
train_yield_model.py - train a Random Forest to estimate crop yield (prototype)

Needs a REAL dataset with district-wise, season-wise, crop-wise Area and Production
(Ministry of Agriculture & Farmers Welfare, "Area Production Yield" / APY data).
Run:  python train_yield_model.py
"""
import glob
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ------------------------------------------------------------------
# 1. SETTINGS - edit these to match the file you downloaded
# ------------------------------------------------------------------
CSV_PATH = "crop_production.csv"   # the CSV must be in the same folder as this script

# Left = name used in this script, right = column name in YOUR csv
COLUMNS = {
    "year": "Crop_Year",
    "state": "State_Name",
    "district": "District_Name",
    "crop": "Crop",
    "season": "Season",
    "area": "Area",             # must be in hectares
    "production": "Production", # must be in tonnes (check the unit column!)
}

STATE_FILTER = None   # e.g. "Telangana" to train on one state only
N_TEST_YEARS = 2      # newest years kept aside for honest testing

# Crop names in the file (lower case) -> name used in the app.
# Add the spellings your file uses.
CROP_NAMES = {
    "rice": "Rice", "cotton": "Cotton", "cotton(lint)": "Cotton",
    "maize": "Maize", "groundnut": "Groundnut",
}

# Multiply Production by this to get TONNES. Cotton is normally reported in bales
# of 170 kg (0.17 tonne) in official statistics. VERIFY this for your file.
PRODUCTION_TO_TONNE = {"Cotton": 0.17}

MIN_AREA_HA = 10      # ignore tiny plots: their yield figures are mostly noise

CATEGORY_FEATURES = ["crop", "state", "district", "season"]
NUMBER_FEATURES = ["year"]
TONNE_PER_HA_TO_QUINTAL_PER_ACRE = 10 / 2.47105   # 1 tonne = 10 quintal, 1 ha = 2.47105 acre


# ------------------------------------------------------------------
# 2. LOAD AND CLEAN
# ------------------------------------------------------------------
def load_data():
    if not glob.glob(CSV_PATH):
        found = glob.glob("*.csv")
        raise SystemExit(f"Cannot find '{CSV_PATH}' in {__import__('os').getcwd()}\n"
                         f"CSV files in this folder: {found or 'none'}\n"
                         "Copy your dataset here, or change CSV_PATH at the top of the script.")
    raw = pd.read_csv(CSV_PATH)
    missing = [c for c in COLUMNS.values() if c not in raw.columns]
    if missing:
        raise SystemExit(f"Columns not found: {missing}\nYour file has: {list(raw.columns)}\n"
                         "Edit COLUMNS at the top of this script.")
    df = raw.rename(columns={v: k for k, v in COLUMNS.items()})[list(COLUMNS)].copy()

    df["crop"] = df["crop"].astype(str).str.strip().str.lower().map(CROP_NAMES)
    if df["crop"].isna().all():
        names = raw[COLUMNS["crop"]].astype(str).str.strip().unique()[:30]
        raise SystemExit(f"No matching crops. Crop names in your file include: {list(names)}\n"
                         "Add them to CROP_NAMES.")
    df = df.dropna(subset=["crop"])
    for col in ["state", "district", "season"]:
        df[col] = df[col].astype(str).str.strip()
    if STATE_FILTER:
        df = df[df["state"].str.lower() == STATE_FILTER.lower()]

    df["year"] = pd.to_numeric(df["year"].astype(str).str[:4], errors="coerce")  # "2019-20" -> 2019
    df["area"] = pd.to_numeric(df["area"], errors="coerce")
    df["production"] = pd.to_numeric(df["production"], errors="coerce")
    df = df.dropna(subset=["year", "area", "production"])
    df = df[(df["area"] >= MIN_AREA_HA) & (df["production"] > 0)]
    df["production"] = df["production"] * df["crop"].map(PRODUCTION_TO_TONNE).fillna(1.0)

    # Drop years with very few rows (e.g. a partly-filled final year)
    per_year = df.groupby("year").size()
    keep = per_year[per_year >= 0.2 * per_year.median()].index
    if len(keep) < len(per_year):
        print("Dropped incomplete years:", [int(y) for y in per_year.index.difference(keep)])
    df = df[df["year"].isin(keep)]

    df["yield_t_ha"] = df["production"] / df["area"]   # tonnes per hectare
    cap = df.groupby("crop")["yield_t_ha"].transform(lambda s: s.quantile(0.995))
    df = df[df["yield_t_ha"] <= cap]                   # drop obvious data errors
    if df.empty:
        raise SystemExit("No usable rows left after cleaning. Check STATE_FILTER and units.")
    return df


# ------------------------------------------------------------------
# 3. TRAIN AND TEST
# ------------------------------------------------------------------
def make_model():
    prep = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORY_FEATURES)],
        remainder="passthrough")                       # "year" passes through unchanged
    forest = RandomForestRegressor(n_estimators=200, min_samples_leaf=3,
                                   random_state=42, n_jobs=-1)
    return Pipeline([("prep", prep), ("forest", forest)])


def main():
    df = load_data()
    years = sorted(df["year"].unique())
    print(f"Rows: {len(df):,} | years {int(years[0])}-{int(years[-1])}")
    print(df["crop"].value_counts().to_string(), "\n")
    if len(years) <= N_TEST_YEARS + 2:
        raise SystemExit("Not enough different years to make a train/test split.")

    # Time-based split: train on older years, test on the newest years.
    test_years = years[-N_TEST_YEARS:]
    train = df[~df["year"].isin(test_years)]
    test = df[df["year"].isin(test_years)].copy()
    features = CATEGORY_FEATURES + NUMBER_FEATURES

    model = make_model()
    model.fit(train[features], train["yield_t_ha"])
    test["predicted"] = model.predict(test[features])

    # Baseline: average past yield of the same crop in the same district.
    key = ["crop", "state", "district"]
    past = train.groupby(key)["yield_t_ha"].mean().rename("baseline").reset_index()
    base = test[key].merge(past, on=key, how="left")["baseline"].to_numpy()
    crop_avg = test["crop"].map(train.groupby("crop")["yield_t_ha"].mean()).to_numpy()
    test["baseline"] = np.where(np.isnan(base), crop_avg, base)   # crop average if district is new

    mae_rf = mean_absolute_error(test["yield_t_ha"], test["predicted"])
    mae_base = mean_absolute_error(test["yield_t_ha"], test["baseline"])
    r2 = r2_score(test["yield_t_ha"], test["predicted"])
    q = TONNE_PER_HA_TO_QUINTAL_PER_ACRE
    print(f"Test years: {[int(y) for y in test_years]} ({len(test):,} rows)")
    print(f"Random Forest MAE: {mae_rf:.3f} t/ha  (= {mae_rf * q:.2f} quintal/acre)   R2: {r2:.3f}")
    print(f"Baseline MAE (district average): {mae_base:.3f} t/ha  (= {mae_base * q:.2f} quintal/acre)")
    if mae_rf < mae_base:
        print(f"The model beats the baseline by {(1 - mae_rf / mae_base) * 100:.1f}%.")
    else:
        print("The model does NOT beat the simple baseline. Report that honestly.")

    test["abs_error"] = (test["yield_t_ha"] - test["predicted"]).abs()
    print("\nMAE by crop (t/ha):")
    print(test.groupby("crop")["abs_error"].mean().round(3).to_string())

    names = model.named_steps["prep"].get_feature_names_out()
    importance = pd.Series(model.named_steps["forest"].feature_importances_, index=names)
    print("\nTop features:")
    print(importance.sort_values(ascending=False).head(8).round(3).to_string())

    # Refit on ALL years for the saved model; the metrics above come from the test split.
    model.fit(df[features], df["yield_t_ha"])
    joblib.dump({
        "model": model, "features": features, "latest_year": int(years[-1]),
        "mae_t_per_ha": float(mae_rf), "baseline_mae_t_per_ha": float(mae_base),
        "r2": float(r2), "rows": len(df),
        "districts": df.groupby("state")["district"].unique().apply(sorted).to_dict(),
    }, "yield_model.joblib")
    print("\nSaved yield_model.joblib")


if __name__ == "__main__":
    main()