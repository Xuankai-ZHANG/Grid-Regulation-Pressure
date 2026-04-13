# -*- coding: utf-8 -*-
"""
quantile_regression.py — Grid Regulation Pressure Weight Estimation

Estimates peak concentration (α) and ramp contribution (β) weights
via τ-specific quantile regression across 9 pressure scenarios.

Model structure:
  Base  model: Q_τ(y) = β₀(τ) + β₁(τ)·PeakConcentration_τ(i)
  Full  model: Q_τ(y) = β₀(τ) + β₁(τ)·PeakConcentration_τ(i)
                       + β₂(τ)·RampContribution(i) + β₃(τ)·L_i
  y_i = standardize( cv_modified_i )
  L_i = standardize( base_share_i )
  α(τ) = ΔR²(τ) / (1 − R²_base(τ))
  β(τ) = 1 − α(τ)

τ–feature mapping (1 + 8 structure):
  τ = 1/48 ≈ 0.0208  →  peak_concent        (single peak)
  τ = 0.05           →  peak_top5_concent
  τ = 0.10           →  peak_top10_concent
  τ = 0.15           →  peak_top15_concent
  τ = 0.20           →  peak_top20_concent
  τ = 0.25           →  peak_top25_concent
  τ = 0.30           →  peak_top30_concent
  τ = 0.50           →  peak_top50_concent
  τ = 0.75           →  peak_top75_concent

Data source : data/grid_data.xlsx  (sheets: mesh_main, model_data)
Output      : output_model/qr_fitting_evaluation.csv
              output_model/qr_feature_mapping.csv
              output_model/qr_weights_results.pkl
              output_model/qr_regression_report.txt
"""
import sys, os, warnings, pickle
import io as _io
if sys.platform == 'win32':
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from sklearn.preprocessing import StandardScaler
from datetime import datetime

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_FILE  = os.path.join(_DIR, 'data', 'grid_data.xlsx')
OUTPUT_DIR = os.path.join(_DIR, 'output_model')
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUT_FITTING  = os.path.join(OUTPUT_DIR, 'qr_fitting_evaluation.csv')
OUT_WEIGHTS  = os.path.join(OUTPUT_DIR, 'qr_weights_results.pkl')
OUT_REPORT   = os.path.join(OUTPUT_DIR, 'qr_regression_report.txt')

# ── τ–feature mapping ──────────────────────────────────────────────────────
FEATURE_TO_TAU = {
    'peak_concent':        1.0 / 48,   # ≈ 0.0208 — single peak
    'peak_top5_concent':   0.05,
    'peak_top10_concent':  0.10,
    'peak_top15_concent':  0.15,
    'peak_top20_concent':  0.20,
    'peak_top25_concent':  0.25,
    'peak_top30_concent':  0.30,
    'peak_top50_concent':  0.50,
    'peak_top75_concent':  0.75,
}

# canonical order (τ ascending)
PEAK_ORDER = [
    'peak_concent',
    'peak_top5_concent', 'peak_top10_concent', 'peak_top15_concent',
    'peak_top20_concent', 'peak_top25_concent', 'peak_top30_concent',
    'peak_top50_concent',
]


# ════════════════════════════════════════════════════════════════════════════
# 1. Data loading
# ════════════════════════════════════════════════════════════════════════════
def load_data():
    """Read mesh_main and model_data sheets, merge on mesh_code."""
    print('\n' + '=' * 80)
    print('Step 1: Loading data from grid_data.xlsx')
    print('=' * 80)

    df_main  = pd.read_excel(DATA_FILE, sheet_name='mesh_main')
    df_model = pd.read_excel(DATA_FILE, sheet_name='model_data')
    print(f'  mesh_main  : {len(df_main)} rows  | columns: {list(df_main.columns)}')
    print(f'  model_data : {len(df_model)} rows | columns: {list(df_model.columns)}')

    # harmonise grid identifier
    if 'grid_code' in df_main.columns and 'mesh_code' not in df_main.columns:
        df_main = df_main.rename(columns={'grid_code': 'mesh_code'})

    # harmonise key type
    df_main['mesh_code']  = df_main['mesh_code'].astype(str)
    df_model['mesh_code'] = df_model['mesh_code'].astype(str)

    # keep only the columns we need from mesh_main
    main_cols = ['mesh_code', 'base_share', 'ramp_contribution']
    df = df_model.merge(df_main[main_cols], on='mesh_code', how='inner')
    print(f'  Merged     : {len(df)} rows')
    return df


# ════════════════════════════════════════════════════════════════════════════
# 2. Feature detection
# ════════════════════════════════════════════════════════════════════════════
def detect_peak_features(df):
    """Return available peak_*_concent columns in canonical τ-ascending order."""
    available = [f for f in PEAK_ORDER if f in df.columns]
    print(f'\n  Peak Concentration features detected ({len(available)}): {available}')
    if not available:
        raise ValueError('No peak_*_concent columns found in data.')
    return available


# ════════════════════════════════════════════════════════════════════════════
# 3. Quantile generation & τ mapping
# ════════════════════════════════════════════════════════════════════════════
def generate_quantiles(peak_features):
    """Map each feature to its τ value; return sorted (τ, feature) pairs."""
    pairs = []
    for feat in peak_features:
        if feat not in FEATURE_TO_TAU:
            raise ValueError(f'Unknown feature: {feat}')
        pairs.append((FEATURE_TO_TAU[feat], feat))
    pairs.sort(key=lambda x: x[0])

    quantiles     = [p[0] for p in pairs]
    feats_ordered = [p[1] for p in pairs]

    print('\n  τ – Peak Concentration mapping:')
    for tau, feat in pairs:
        print(f'    τ = {tau:.4f}  →  {feat}')
    return quantiles, feats_ordered


# ════════════════════════════════════════════════════════════════════════════
# 4. Regression data preparation
# ════════════════════════════════════════════════════════════════════════════
def prepare_regression_data(df):
    """
    Build y and L_i:
      y   = standardize( cv_modified )
      L_i = standardize( base_share )
    """
    print('\n' + '=' * 80)
    print('Step 2: Preparing regression data')
    print('=' * 80)

    y_raw = df['cv_modified']
    y = (y_raw - y_raw.mean()) / y_raw.std()
    print(f'  y  = standardize( cv_modified )')
    print(f'       mean={y.mean():.4f}  std={y.std():.4f}  '
          f'range=[{y.min():.4f}, {y.max():.4f}]')

    L_i = (df['base_share'] - df['base_share'].mean()) / df['base_share'].std()
    print(f'  L_i = standardize( base_share )')
    print(f'        mean={L_i.mean():.4f}  std={L_i.std():.4f}')

    # drop rows with NaN in y, L_i, ramp_contribution
    mask = ~(y.isna() | L_i.isna() | df['ramp_contribution'].isna())
    if mask.sum() < len(df):
        print(f'  Dropped {(~mask).sum()} rows with NaN values.')
    df_c = df[mask].reset_index(drop=True)
    y    = y[mask].reset_index(drop=True)
    L_i  = L_i[mask].reset_index(drop=True)

    print(f'  Final sample: {len(df_c)} grids')
    return y, L_i, df_c


# ════════════════════════════════════════════════════════════════════════════
# 5. Pseudo R² (Koenker-Machado)
# ════════════════════════════════════════════════════════════════════════════
def pseudo_r2(y, y_pred, tau):
    def rho(u):
        return u * (tau - (u < 0).astype(float))
    res  = y - y_pred
    num  = np.sum(rho(res))
    denom = np.sum(rho(y - np.quantile(y, tau)))
    if denom == 0 or np.isnan(denom):
        return 0.0
    r2 = 1.0 - num / denom
    return float(np.clip(r2, 0.0, 1.0))


# ════════════════════════════════════════════════════════════════════════════
# 6. Quantile regression
# ════════════════════════════════════════════════════════════════════════════
def run_quantile_regression(y, L_i, df_c, quantiles, peak_features, max_iter=1000):
    """
    For each τ:
      Base model : [const, peak_concent_τ]
      Full model : [const, peak_concent_τ, ramp_contribution, L_i]
      α(τ) = ΔR² / (1 − R²_base),  β(τ) = 1 − α
    """
    print('\n' + '=' * 80)
    print('Step 3: Quantile regression')
    print('=' * 80)
    print(f'  Grids: {len(y)}  |  τ values: {len(quantiles)}  |  max_iter: {max_iter}')

    results = {}

    for tau, feat in zip(quantiles, peak_features):
        print(f'\n  τ = {tau:.4f}  ({feat})')

        try:
            peak_vals = df_c[feat].values.reshape(-1, 1)
            ramp_vals = df_c['ramp_contribution'].values.reshape(-1, 1)
            L_vals    = L_i.values.reshape(-1, 1)

            # scale
            sc_base = StandardScaler()
            sc_full = StandardScaler()
            X_base  = sc_base.fit_transform(peak_vals)
            X_full  = sc_full.fit_transform(
                np.hstack([peak_vals, ramp_vals, L_vals])
            )

            X_base_c = sm.add_constant(X_base)
            X_full_c = sm.add_constant(X_full)

            res_base = QuantReg(y, X_base_c).fit(q=tau, max_iter=max_iter)
            r2_base  = pseudo_r2(y.values, res_base.predict(X_base_c), tau)

            res_full = QuantReg(y, X_full_c).fit(q=tau, max_iter=max_iter)
            r2_full  = pseudo_r2(y.values, res_full.predict(X_full_c), tau)

            delta_r2 = r2_full - r2_base
            alpha    = float(np.clip(delta_r2 / (1 - r2_base), 0, 1)) \
                       if r2_base < 1 else 0.5
            beta     = 1.0 - alpha

            print(f'    R²_base={r2_base:.6f}  R²_full={r2_full:.6f}  '
                  f'ΔR²={delta_r2:.6f}  α={alpha:.6f}  β={beta:.6f}')

            results[tau] = dict(
                peak_feature  = feat,
                r2_base       = r2_base,
                r2_full       = r2_full,
                delta_r2      = delta_r2,
                alpha         = alpha,
                beta          = beta,
                coeffs_base   = res_base.params.copy(),
                coeffs_full   = res_full.params.copy(),
            )

        except Exception as exc:
            print(f'    ERROR: {exc}')
            results[tau] = dict(peak_feature=feat, r2_base=np.nan,
                                r2_full=np.nan, delta_r2=np.nan,
                                alpha=np.nan, beta=np.nan)

    print('\n  Quantile regression complete.')
    return results


# ════════════════════════════════════════════════════════════════════════════
# 7. Weight stability analysis
# ════════════════════════════════════════════════════════════════════════════
def analyze_weight_stability(results, quantiles):
    print('\n' + '=' * 80)
    print('Step 4: Weight stability analysis')
    print('=' * 80)

    alphas = [results[t]['alpha'] for t in quantiles
              if not np.isnan(results[t].get('alpha', np.nan))]
    betas  = [results[t]['beta']  for t in quantiles
              if not np.isnan(results[t].get('beta',  np.nan))]

    if not alphas:
        print('  No valid weight values found.')
        return None

    alphas, betas = np.array(alphas), np.array(betas)

    def _stats(v):
        return dict(mean=v.mean(), std=v.std(),
                    cv=v.std()/v.mean() if v.mean() else 0,
                    min=v.min(), max=v.max())

    sa, sb = _stats(alphas), _stats(betas)
    df_stab = pd.DataFrame({
        'Metric': ['Alpha (Peak Concentration)', 'Beta (Ramp Contribution)'],
        'Mean':   [sa['mean'], sb['mean']],
        'Std':    [sa['std'],  sb['std']],
        'CV':     [sa['cv'],   sb['cv']],
        'Min':    [sa['min'],  sb['min']],
        'Max':    [sa['max'],  sb['max']],
    })

    print(df_stab.to_string(index=False))

    level = ('High'   if sa['cv'] < 0.10 and sb['cv'] < 0.10 else
             'Medium' if sa['cv'] < 0.20 and sb['cv'] < 0.20 else 'Low')
    print(f'\n  Stability level: {level}  '
          f'(CV_α={sa["cv"]:.4f}, CV_β={sb["cv"]:.4f})')
    return df_stab


# ════════════════════════════════════════════════════════════════════════════
# 8. Fitting evaluation table
# ════════════════════════════════════════════════════════════════════════════
def generate_fitting_evaluation(results, quantiles):
    rows = []
    for tau in quantiles:
        r = results[tau]
        rows.append({
            'Quantile':     tau,
            'Peak_Feature': r.get('peak_feature', ''),
            'R2_base':      r.get('r2_base',  np.nan),
            'R2_full':      r.get('r2_full',  np.nan),
            'Delta_R2':     r.get('delta_r2', np.nan),
            'Alpha':        r.get('alpha',    np.nan),
            'Beta':         r.get('beta',     np.nan),
        })
    df_fit = pd.DataFrame(rows)
    print('\n  Fitting evaluation:')
    print(df_fit.to_string(index=False))
    return df_fit


# ════════════════════════════════════════════════════════════════════════════
# 9. Save outputs
# ════════════════════════════════════════════════════════════════════════════
def save_outputs(results, df_fitting, df_stability, quantiles):
    print('\n' + '=' * 80)
    print('Step 6: Saving outputs')
    print('=' * 80)

    # CSV: fitting evaluation
    df_fitting.to_csv(OUT_FITTING, index=False, encoding='utf-8-sig')
    print(f'  Saved: {OUT_FITTING}')

    # PKL: weights (strip statsmodels objects)
    with open(OUT_WEIGHTS, 'wb') as fh:
        pickle.dump(results, fh)
    print(f'  Saved: {OUT_WEIGHTS}')

    # TXT: report
    with open(OUT_REPORT, 'w', encoding='utf-8') as fh:
        fh.write('=' * 80 + '\n')
        fh.write('Grid Regulation Pressure — Quantile Regression Report\n')
        fh.write('=' * 80 + '\n\n')
        fh.write(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')

        fh.write('Model:\n')
        fh.write('  y_i = standardize( cv_modified_i )\n')
        fh.write('  L_i = standardize( base_share_i )\n')
        fh.write('  Base : Q_τ(y) = β₀ + β₁·PeakConc_τ\n')
        fh.write('  Full : Q_τ(y) = β₀ + β₁·PeakConc_τ + β₂·Ramp + β₃·L_i\n')
        fh.write('  α(τ) = ΔR²(τ) / (1 − R²_base(τ));  β(τ) = 1 − α(τ)\n\n')

        fh.write('Feature Mapping (τ → Peak Concentration):\n')
        for tau, feat in zip(quantiles, [results[t]['peak_feature'] for t in quantiles]):
            fh.write(f'  τ={tau:.4f}  →  {feat}\n')
        fh.write('\n')

        fh.write('Fitting Evaluation:\n')
        fh.write(df_fitting.to_string(index=False) + '\n\n')

        if df_stability is not None:
            fh.write('Weight Stability:\n')
            fh.write(df_stability.to_string(index=False) + '\n\n')

        fh.write('Peak Concentration (α) by τ:\n')
        for tau in quantiles:
            r = results[tau]
            fh.write(f'  τ={tau:.4f}  α={r.get("alpha", float("nan")):.6f}  '
                     f'β={r.get("beta", float("nan")):.6f}\n')

    print(f'  Saved: {OUT_REPORT}')


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
def main():
    print('\n' + '=' * 80)
    print('Grid Regulation Pressure — Quantile Regression')
    print(f'Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 80)

    MAX_ITER = 1000

    try:
        # 1. Load
        df = load_data()

        # 2. Detect peak features
        peak_features = detect_peak_features(df)

        # 3. Build τ mapping
        quantiles, peak_features = generate_quantiles(peak_features)

        # 4. Prepare y and L_i
        y, L_i, df_c = prepare_regression_data(df)

        # 5. Run regression
        results = run_quantile_regression(y, L_i, df_c, quantiles, peak_features,
                                          max_iter=MAX_ITER)

        # 6. Stability
        df_stability = analyze_weight_stability(results, quantiles)

        # 7. Fitting table
        print('\n' + '=' * 80)
        print('Step 5: Fitting evaluation table')
        print('=' * 80)
        df_fitting = generate_fitting_evaluation(results, quantiles)

        # 8. Feature mapping table (inline, no CSV)
        df_mapping = pd.DataFrame({
            'Quantile':     quantiles,
            'Peak_Feature': peak_features,
        })

        # 9. Save
        save_outputs(results, df_fitting, df_stability, quantiles)

        print('\n' + '=' * 80)
        print('Done. Outputs written to:', OUTPUT_DIR)
        print('=' * 80)

    except Exception as exc:
        import traceback
        print(f'\nERROR: {exc}')
        traceback.print_exc()


if __name__ == '__main__':
    main()
