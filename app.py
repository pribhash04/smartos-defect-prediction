import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.io import arff
from collections import Counter

from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SMART-OS | Defect Prediction",
    page_icon="🔬",
    layout="wide",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Sora:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Sora', sans-serif;
    }
    .main { background-color: #0d0f14; }
    .block-container { padding: 2rem 3rem; }

    h1 { font-family: 'Sora', sans-serif !important; font-weight: 700 !important; }
    h2, h3 { font-family: 'Sora', sans-serif !important; font-weight: 600 !important; }

    .metric-card {
        background: #161b27;
        border: 1px solid #2a3044;
        border-radius: 12px;
        padding: 1.2rem 1rem;
        text-align: center;
    }
    .metric-label {
        font-size: 11px;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #7a8499;
        font-family: 'JetBrains Mono', monospace;
        margin-bottom: 4px;
    }
    .metric-winner {
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 2px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #e8eaf0;
    }
    .tag {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
    }
    .info-box {
        background: #161b27;
        border-left: 3px solid #4a7cf7;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        font-size: 13px;
        color: #a0aabf;
        margin-bottom: 1rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #4a7cf7, #7c3aed);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-family: 'Sora', sans-serif;
        font-weight: 600;
        font-size: 15px;
        width: 100%;
        cursor: pointer;
    }
    .stButton > button:hover {
        opacity: 0.9;
    }
    div[data-testid="stSelectbox"] label {
        font-size: 13px;
        color: #7a8499;
        font-family: 'JetBrains Mono', monospace;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SMART-OS CLASS
# ─────────────────────────────────────────────
class SMARTOS:
    def __init__(self, random_state=42):
        self.k_min = 3
        self.k_max = 10
        self.noise_threshold = 0.5
        self.safe_zone_threshold = 0.3
        self.rng = np.random.RandomState(random_state)
        self.random_state = random_state

    def fit_resample(self, X, y):
        X, y = np.array(X, dtype=float), np.array(y)
        counts = Counter(y)
        if len(counts) < 2:
            return X, y
        maj_cls = max(counts, key=counts.get)
        min_cls = min(counts, key=counts.get)
        n_to_gen = counts[maj_cls] - counts[min_cls]
        if n_to_gen == 0:
            return X, y
        X_min = X[y == min_cls]
        X_maj = X[y == maj_cls]
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)
        Xm_sc = X_sc[y == min_cls]
        Xj_sc = X_sc[y == maj_cls]
        Xm_sc_clean, X_min_clean = self._filter_noise(Xm_sc, Xj_sc, X_min)
        if len(X_min_clean) < 2:
            Xm_sc_clean, X_min_clean = Xm_sc, X_min
        weights = self._weights(Xm_sc_clean, Xj_sc)
        labels = self._cluster(Xm_sc_clean)
        k_vals = self._dynamic_k(Xm_sc_clean)
        synthetic = self._generate(X_min_clean, Xm_sc_clean, weights, labels, k_vals, n_to_gen, scaler, Xj_sc)
        if len(synthetic) == 0:
            return X, y
        return np.vstack([X, synthetic]), np.concatenate([y, np.full(len(synthetic), min_cls)])

    def _filter_noise(self, Xm, Xj, X_min_orig):
        k = min(self.k_min + 2, len(Xm) - 1, len(Xj))
        if k < 1:
            return Xm, X_min_orig
        all_X = np.vstack([Xm, Xj])
        labels = np.array([0]*len(Xm) + [1]*len(Xj))
        _, idx = NearestNeighbors(n_neighbors=k+1).fit(all_X).kneighbors(Xm)
        keep = np.array([np.mean(labels[i[1:]] == 1) < self.noise_threshold for i in idx])
        return Xm[keep], X_min_orig[keep]

    def _weights(self, Xm, Xj):
        k = min(self.k_max, len(Xm) + len(Xj) - 1)
        all_X = np.vstack([Xm, Xj])
        labels = np.array([0]*len(Xm) + [1]*len(Xj))
        _, idx = NearestNeighbors(n_neighbors=k+1).fit(all_X).kneighbors(Xm)
        w = np.array([np.mean(labels[i[1:]] == 1) + 1e-6 for i in idx])
        return w / w.sum()

    def _cluster(self, Xm):
        n = len(Xm)
        nc = min(max(2, int(np.sqrt(n))), n - 1)
        if nc < 2:
            return np.zeros(n, dtype=int)
        return KMeans(n_clusters=nc, random_state=self.random_state, n_init=10).fit_predict(Xm)

    def _dynamic_k(self, Xm):
        n = len(Xm)
        pk = min(self.k_max, n - 1)
        if pk < 1:
            return np.full(n, self.k_min)
        dist, _ = NearestNeighbors(n_neighbors=pk+1).fit(Xm).kneighbors(Xm)
        md = dist[:, 1:].mean(axis=1)
        lo, hi = md.min(), md.max()
        if hi == lo:
            return np.full(n, self.k_min)
        norm = (md - lo) / (hi - lo)
        return np.clip((self.k_min + norm * (self.k_max - self.k_min)).astype(int), self.k_min, pk)

    def _generate(self, X_min, Xm_sc, weights, labels, k_vals, n_gen, scaler, Xj_sc):
        n = len(X_min)
        synth, attempts = [], 0
        clusters = {}
        for c in np.unique(labels):
            mask = labels == c
            if mask.sum() < 2:
                continue
            Xc = X_min[mask]
            kc = min(self.k_max, len(Xc) - 1)
            clusters[c] = (Xc, NearestNeighbors(n_neighbors=kc+1).fit(Xc), np.where(mask)[0])
        while len(synth) < n_gen and attempts < n_gen * 25:
            attempts += 1
            si = self.rng.choice(n, p=weights)
            c = labels[si]
            if c not in clusters:
                continue
            Xc, nbrs, g_idx = clusters[c]
            lpos = np.where(g_idx == si)[0]
            if len(lpos) == 0:
                continue
            li = lpos[0]
            ku = min(k_vals[si], len(Xc) - 1)
            if ku < 1:
                continue
            _, nn = nbrs.kneighbors(Xc[li:li+1], n_neighbors=ku+1)
            nn = nn[0][1:]
            if len(nn) == 0:
                continue
            pt = X_min[si] + self.rng.uniform(0, 1) * (Xc[self.rng.choice(nn)] - X_min[si])
            pt_sc = scaler.transform(pt.reshape(1, -1))[0]
            all_X = np.vstack([Xm_sc, Xj_sc])
            k = min(10, len(all_X) - 1)
            _, idx = NearestNeighbors(n_neighbors=k).fit(all_X).kneighbors(pt_sc.reshape(1, -1))
            if np.mean(idx[0] < len(Xm_sc)) >= self.safe_zone_threshold:
                synth.append(pt)
        return np.array(synth) if synth else np.empty((0, X_min.shape[1]))


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def load_and_clean(file):
    data, _ = arff.loadarff(file)
    df = pd.DataFrame(data)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
    return df

def get_label_column(df):
    for candidate in ['isDefective', 'class', 'label', 'bug', 'defective']:
        if candidate in df.columns:
            return candidate
    return df.columns[-1]

def split_features_labels(df):
    label_col = get_label_column(df)
    return df.drop(columns=[label_col]), df[label_col]

def process_labels(y):
    y = y.astype(str).str.strip()
    mapping = {
        'TRUE': 1, 'FALSE': 0, 'true': 1, 'false': 0,
        'buggy': 1, 'clean': 0, 'yes': 1, 'no': 0,
        'Y': 1, 'N': 0, '1': 1, '0': 0,
        '1.0': 1, '0.0': 0, 'defective': 1, 'non-defective': 0,
    }
    return y.map(mapping)

def make_chart(results_df):
    metrics = ['F1', 'Precision', 'Recall', 'AUC', 'MCC']
    techniques = ['No Resampling', 'SMOTE', 'ADASYN', 'SMART-OS']
    colors = ['#4a7cf7', '#1D9E75', '#f97316', '#a855f7']
    hatches = ['', '///', '...', 'xxx']

    avg = results_df.groupby("Technique")[metrics].mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#0d0f14')
    ax.set_facecolor('#0d0f14')

    x = np.arange(len(metrics))
    n = len(techniques)
    width = 0.18
    offsets = np.linspace(-(n-1)/2, (n-1)/2, n) * width

    for i, (tech, color, hatch) in enumerate(zip(techniques, colors, hatches)):
        if tech not in avg.index:
            continue
        vals = [avg.loc[tech, m] for m in metrics]
        bars = ax.bar(x + offsets[i], vals, width,
                      color=color, alpha=0.85, hatch=hatch,
                      edgecolor='#0d0f14', linewidth=0.8, zorder=3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                    f'{val:.3f}', ha='center', va='bottom',
                    fontsize=7, color=color, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12, fontweight='bold', color='#c8d0e0')
    ax.set_ylabel('Score', fontsize=10, color='#7a8499')
    ax.set_ylim(0, 0.95)
    ax.yaxis.set_tick_params(labelsize=9, labelcolor='#7a8499')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#2a3044')
    ax.spines['bottom'].set_color('#2a3044')
    ax.yaxis.grid(True, color='#1e2535', linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    legend_patches = [
        mpatches.Patch(facecolor=c, hatch=h, edgecolor='#0d0f14', label=t, alpha=0.85)
        for t, c, h in zip(techniques, colors, hatches)
    ]
    ax.legend(handles=legend_patches, loc='upper right',
              fontsize=9, framealpha=0.2, edgecolor='#2a3044',
              ncol=2, handlelength=1.8, handleheight=1.2,
              labelcolor='#c8d0e0', facecolor='#161b27')

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="padding: 1rem 0 0.5rem;">
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #4a7cf7; letter-spacing: 0.15em; margin-bottom: 8px;">
        SOFTWARE ENGINEERING · DEFECT PREDICTION
    </div>
    <h1 style="font-size: 2.2rem; margin: 0; color: #e8eaf0;">
        SMART-OS <span style="color: #4a7cf7;">⚡</span>
    </h1>
    <p style="color: #7a8499; font-size: 14px; margin-top: 6px;">
        Adaptive Oversampling for Cross-Project Software Defect Prediction · GaussianNB Classifier
    </p>
</div>
<hr style="border: none; border-top: 1px solid #2a3044; margin: 1rem 0 1.5rem;">
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    files_available = ["Apache.arff", "Safe.arff", "Zxing.arff"]
    test_choice = st.selectbox(
        "SELECT TEST DATASET",
        options=files_available,
        index=0,
        help="The remaining two datasets will be used for training."
    )
    train_files = [f for f in files_available if f != test_choice]

    st.markdown(f"""
    <div class="info-box">
        <b style="color:#4a7cf7;">Train:</b> {train_files[0].replace('.arff','')} + {train_files[1].replace('.arff','')}<br>
        <b style="color:#a855f7;">Test:</b> {test_choice.replace('.arff','')}
    </div>
    """, unsafe_allow_html=True)

    run_all = st.checkbox("Run all 3 splits", value=False)

    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""
    <div style="font-size:12px; color:#7a8499; line-height:1.7;">
    Compares four techniques:<br>
    • <b style="color:#4a7cf7;">No Resampling</b> — baseline<br>
    • <b style="color:#1D9E75;">SMOTE</b> — interpolation<br>
    • <b style="color:#f97316;">ADASYN</b> — adaptive density<br>
    • <b style="color:#a855f7;">SMART-OS</b> — proposed method
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    run_btn = st.button("▶ Run Experiment")


# ─────────────────────────────────────────────
# MAIN EXPERIMENT LOGIC
# ─────────────────────────────────────────────
def run_one(train_files, test_file):
    results = []
    logs = []

    train_df = pd.concat([load_and_clean(f) for f in train_files], ignore_index=True)
    X_train, y_raw_train = split_features_labels(train_df)
    y_train = process_labels(y_raw_train)
    train_data = pd.concat([X_train, y_train], axis=1).dropna()
    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1].astype(int)

    test_df = load_and_clean(test_file)
    X_test, y_raw_test = split_features_labels(test_df)
    y_test = process_labels(y_raw_test)
    test_data = pd.concat([X_test, y_test], axis=1).dropna()
    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1].astype(int)

    common_cols = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    train_dist = dict(Counter(y_train))
    test_dist = dict(Counter(y_test))

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    techniques = {"No Resampling": (X_train_scaled, y_train)}

    try:
        X_sm, y_sm = SMOTE(random_state=42).fit_resample(X_train_scaled, y_train)
        techniques["SMOTE"] = (X_sm, y_sm)
        logs.append(f"✅ SMOTE: {dict(Counter(y_sm))}")
    except Exception as e:
        techniques["SMOTE"] = (X_train_scaled, y_train)
        logs.append(f"⚠️ SMOTE skipped: {e}")

    try:
        X_ad, y_ad = ADASYN(random_state=42).fit_resample(X_train_scaled, y_train)
        techniques["ADASYN"] = (X_ad, y_ad)
        logs.append(f"✅ ADASYN: {dict(Counter(y_ad))}")
    except Exception as e:
        techniques["ADASYN"] = (X_train_scaled, y_train)
        logs.append(f"⚠️ ADASYN skipped: {e}")

    try:
        X_so, y_so = SMARTOS(random_state=42).fit_resample(X_train_scaled, y_train)
        techniques["SMART-OS"] = (X_so, y_so)
        logs.append(f"✅ SMART-OS: {dict(Counter(y_so))}")
    except Exception as e:
        techniques["SMART-OS"] = (X_train_scaled, y_train)
        logs.append(f"⚠️ SMART-OS skipped: {e}")

    train_label = f"{train_files[0].replace('.arff','')}+{train_files[1].replace('.arff','')}"

    for tech_name, (X_tr, y_tr) in techniques.items():
        model = GaussianNB()
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        f1 = f1_score(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = float('nan')
        results.append([train_label, test_file.replace('.arff', ''), tech_name,
                        round(f1, 3), round(precision, 3), round(recall, 3),
                        round(auc, 3), round(mcc, 3)])

    return results, train_dist, test_dist, logs


if run_btn:
    all_results = []

    if run_all:
        splits = [(["Apache.arff", "Safe.arff"], "Zxing.arff"),
                  (["Apache.arff", "Zxing.arff"], "Safe.arff"),
                  (["Safe.arff", "Zxing.arff"],   "Apache.arff")]
        prog = st.progress(0, text="Running experiments...")
        for i, (tr, te) in enumerate(splits):
            res, _, _, _ = run_one(tr, te)
            all_results.extend(res)
            prog.progress((i+1)/3, text=f"Completed {i+1}/3 splits")
        prog.empty()
    else:
        with st.spinner(f"Running experiment: {train_files[0].replace('.arff','')} + {train_files[1].replace('.arff','')} → {test_choice.replace('.arff','')}..."):
            res, train_dist, test_dist, logs = run_one(train_files, test_choice)
            all_results.extend(res)

    results_df = pd.DataFrame(
        all_results,
        columns=["Train", "Test", "Technique", "F1", "Precision", "Recall", "AUC", "MCC"]
    )

    # ── Distribution info ──
    if not run_all:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**📊 Train class distribution**")
            st.json(train_dist)
        with c2:
            st.markdown("**📊 Test class distribution**")
            st.json(test_dist)
        with st.expander("📋 Resampling logs"):
            for l in logs:
                st.markdown(f"`{l}`")

    st.markdown("---")

    # ── Results table ──
    st.markdown("### 📋 Results Table")

    colors_map = {
        'No Resampling': '#4a7cf7',
        'SMOTE': '#1D9E75',
        'ADASYN': '#f97316',
        'SMART-OS': '#a855f7'
    }

    def highlight_best(df):
        styled = df.copy()
        for col in ['F1', 'Precision', 'Recall', 'AUC', 'MCC']:
            max_val = df[col].max()
            styled[col] = df[col].apply(
                lambda v: f'**{v}** ⬆' if v == max_val else str(v)
            )
        return styled

    st.dataframe(
        results_df.style.background_gradient(subset=['F1','Precision','Recall','AUC','MCC'], cmap='Blues'),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # ── Average results ──
    st.markdown("### 📈 Average Performance Across All Splits")
    avg = results_df.groupby("Technique")[["F1", "Precision", "Recall", "AUC", "MCC"]].mean().round(4)
    st.dataframe(avg.style.background_gradient(cmap='Purples'), use_container_width=True)

    # ── Bar chart ──
    st.markdown("### 📊 Comparison Chart")
    fig = make_chart(results_df)
    st.pyplot(fig, use_container_width=True)

    st.markdown("---")

    # ── Winner cards ──
    st.markdown("### 🏆 Winner Per Metric")
    metric_cols = st.columns(5)
    metric_list = ['F1', 'Precision', 'Recall', 'AUC', 'MCC']
    tech_colors = {
        'No Resampling': ('#4a7cf7', '#1a2540'),
        'SMOTE':         ('#1D9E75', '#0d2318'),
        'ADASYN':        ('#f97316', '#2a1a0d'),
        'SMART-OS':      ('#a855f7', '#1e1030'),
    }

    for i, metric in enumerate(metric_list):
        winner = avg[metric].idxmax()
        score = avg.loc[winner, metric]
        color, bg = tech_colors.get(winner, ('#4a7cf7', '#1a2540'))
        with metric_cols[i]:
            st.markdown(f"""
            <div class="metric-card" style="border-color: {color}33; background: {bg};">
                <div class="metric-label">{metric}</div>
                <div class="metric-winner" style="color: {color};">{winner}</div>
                <div class="metric-value">{score:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#3a4460; font-size:12px; font-family:'JetBrains Mono',monospace;">
        SMART-OS · Cross-Project Software Defect Prediction · GaussianNB
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Landing state ──
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🔬</div>
        <h2 style="color: #c8d0e0; font-size: 1.5rem;">Ready to Run</h2>
        <p style="color: #7a8499; font-size: 14px; max-width: 480px; margin: 0 auto;">
            Select a test dataset from the sidebar and click <b style="color:#4a7cf7;">▶ Run Experiment</b> 
            to compare No Resampling, SMOTE, ADASYN, and SMART-OS techniques.
        </p>
    </div>

    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-top: 2rem;">
        <div class="metric-card" style="border-color: #4a7cf733;">
            <div class="metric-label">Baseline</div>
            <div class="metric-winner" style="color:#4a7cf7;">No Resampling</div>
            <div style="font-size:12px; color:#7a8499; margin-top:6px;">Original imbalanced data</div>
        </div>
        <div class="metric-card" style="border-color: #1D9E7533;">
            <div class="metric-label">Classic</div>
            <div class="metric-winner" style="color:#1D9E75;">SMOTE</div>
            <div style="font-size:12px; color:#7a8499; margin-top:6px;">k-NN interpolation</div>
        </div>
        <div class="metric-card" style="border-color: #f9731633;">
            <div class="metric-label">Adaptive</div>
            <div class="metric-winner" style="color:#f97316;">ADASYN</div>
            <div style="font-size:12px; color:#7a8499; margin-top:6px;">Density-based weighting</div>
        </div>
        <div class="metric-card" style="border-color: #a855f733;">
            <div class="metric-label">Proposed</div>
            <div class="metric-winner" style="color:#a855f7;">SMART-OS</div>
            <div style="font-size:12px; color:#7a8499; margin-top:6px;">Safe-zone + clustering</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
