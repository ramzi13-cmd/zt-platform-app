import streamlit as st
import pandas as pd
import numpy as np
import warnings
import joblib
import os
import requests
warnings.filterwarnings('ignore')

# ── API key ───────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))
except ImportError:
    pass

# When deployed on Streamlit Cloud, secrets are provided via st.secrets
# (set in the app's dashboard settings), not a local .env file. This copies
# every st.secrets entry into os.environ, so every existing os.environ.get(...)
# call throughout this file keeps working unchanged, locally and once deployed.
try:
    for _key, _val in st.secrets.items():
        if _key not in os.environ:
            os.environ[_key] = str(_val)
except Exception:
    pass  # no secrets.toml locally — fine, .env already covers local dev

try:
    _ENV_OPENROUTER_KEY = st.secrets.get("OPENROUTER_API_KEY", "") or os.environ.get("OPENROUTER_API_KEY", "")
except:
    _ENV_OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# ── Supabase (Research Feed) ────────────────────────────────────
from datetime import datetime, date
try:
    from supabase import create_client
except ImportError:
    create_client = None

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_PUBLISHABLE_KEY = os.environ.get("SUPABASE_PUBLISHABLE_KEY", "")

@st.cache_resource
def get_supabase_client():
    if create_client is None or not SUPABASE_URL or not SUPABASE_PUBLISHABLE_KEY:
        return None
    return create_client(SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY)

supabase = get_supabase_client()

@st.cache_data(ttl=300)
def load_papers():
    if supabase is None:
        return pd.DataFrame()
    response = supabase.table('papers').select('*').order('pub_date', desc=True).execute()
    return pd.DataFrame(response.data)

@st.cache_data(ttl=60)
def load_saved_ids():
    if supabase is None:
        return set()
    response = supabase.table('saved').select('paper_id').execute()
    return set(r['paper_id'] for r in response.data)

def toggle_save(paper_id, currently_saved):
    if currently_saved:
        supabase.table('saved').delete().eq('paper_id', paper_id).execute()
    else:
        supabase.table('saved').insert({'paper_id': paper_id}).execute()
    st.cache_data.clear()

@st.cache_data(ttl=60)
def load_extraction_status():
    """Returns dict: paper_id -> list of extraction records (figure_label, created_at)"""
    if supabase is None:
        return {}
    response = supabase.table('extracted_data').select('paper_id, figure_label, created_at').execute()
    status = {}
    for r in response.data:
        pid = r['paper_id']
        if pid not in status:
            status[pid] = []
        status[pid].append(r)
    return status

def relative_date(pub_date_str):
    try:
        pub_date = datetime.strptime(str(pub_date_str), '%Y-%m-%d').date()
        days = (date.today() - pub_date).days
        if days <= 0: return "today"
        elif days == 1: return "yesterday"
        elif days < 7: return f"{days} days ago"
        else: return pub_date.strftime("%b %d")
    except:
        return str(pub_date_str)

def relevance_color(score):
    if score >= 0.7: return "#22543D"
    elif score >= 0.4: return "#27AE60"
    else: return "#F39C12"

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Thermoelectric ZT Platform | NTNU",
    page_icon="🔥", layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_DIR = './'
DATA_PATH  = PROJECT_DIR + 'thermoelectric_dataset_magpie_v2.csv'
MODEL_PATH = PROJECT_DIR + 'zt_gradboost_model.pkl'
FEAT_PATH  = PROJECT_DIR + 'feature_cols.pkl'
SHAP_PATH  = PROJECT_DIR + 'shap_feature_importance.csv'

# ── CSS (same style as LiAgent) ───────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600;700&family=Source+Sans+3:wght@300;400;600&display=swap');
html,body,[class*="css"]{font-family:'Source Sans 3',sans-serif;}
[data-testid="stSidebar"]{background:#1B2A4A;color:white;}
[data-testid="stSidebar"] *{color:white !important;}
.stat-card{background:white;border:1px solid #DDE3ED;border-top:3px solid #C0392B;border-radius:4px;padding:16px;text-align:center;}
.stat-value{font-family:'Source Serif 4',serif;font-size:2rem;font-weight:700;color:#1B2A4A;}
.stat-label{font-size:0.8rem;color:#5A6478;margin-top:4px;text-transform:uppercase;letter-spacing:0.05em;}
.section-title{font-family:'Source Serif 4',serif;font-size:1.1rem;font-weight:600;color:#1B2A4A;border-bottom:2px solid #DDE3ED;padding-bottom:6px;margin:18px 0 12px 0;}
.info-box{background:#FFF5F5;border-left:4px solid #C0392B;padding:10px 14px;border-radius:0 4px 4px 0;font-size:0.9rem;color:#2D3748;margin-bottom:16px;}
.result-box{background:white;border:1px solid #DDE3ED;border-radius:8px;padding:20px 24px;text-align:center;margin:16px 0;}
.result-zt{font-family:'Source Serif 4',serif;font-size:2.8rem;font-weight:700;color:#1B2A4A;}
.badge{display:inline-block;padding:4px 12px;border-radius:12px;font-size:0.78rem;font-weight:600;letter-spacing:0.05em;text-transform:uppercase;}
.badge-excellent{background:#C6F6D5;color:#22543D;}
.badge-good{background:#FEFCBF;color:#744210;}
.badge-poor{background:#FED7D7;color:#742A2A;}
.feed-card{background:white;border:1px solid #DDE3ED;border-radius:12px;padding:16px 18px;margin-bottom:12px;transition:.12s;}
.feed-card:hover{border-color:#9AA5B8;}
.feed-meta{display:flex;align-items:center;gap:8px;margin-bottom:7px;flex-wrap:wrap;}
.feed-src{font-family:monospace;font-size:10px;font-weight:700;padding:2px 7px;border-radius:5px;text-transform:uppercase;letter-spacing:.3px;}
.feed-src.arxiv{color:#B26A12;background:#FBF0DC;}
.feed-src.journal{color:#2563EB;background:#E7EFFC;}
.feed-date{font-size:12px;color:#6B7683;}
.feed-title{font-size:16px;font-weight:600;color:#1B2A4A;margin:4px 0;line-height:1.35;}
.feed-authors{font-size:12.5px;color:#6B7683;margin-bottom:8px;}
.feed-rel{margin-left:auto;font-family:monospace;font-size:11px;font-weight:600;}
</style>
""", unsafe_allow_html=True)

# ── ZT grade helpers ──────────────────────────────────────────
# ── Human-friendly Magpie feature names ──────────────────────
# Raw matminer column names (e.g. "MagpieData mean GSbandgap") are
# accurate but not user-friendly. This translates them into plain
# English wherever feature names are shown in the UI.
MAGPIE_PROPERTY_NAMES = {
    "Number": "Atomic Number",
    "MendeleevNumber": "Mendeleev Number",
    "AtomicWeight": "Atomic Weight",
    "MeltingT": "Melting Point",
    "Column": "Periodic Table Group",
    "Row": "Periodic Table Period",
    "CovalentRadius": "Covalent Radius",
    "Electronegativity": "Electronegativity",
    "NsValence": "s-Valence Electrons",
    "NpValence": "p-Valence Electrons",
    "NdValence": "d-Valence Electrons",
    "NfValence": "f-Valence Electrons",
    "NValence": "Total Valence Electrons",
    "NsUnfilled": "Unfilled s-Orbital Electrons",
    "NpUnfilled": "Unfilled p-Orbital Electrons",
    "NdUnfilled": "Unfilled d-Orbital Electrons",
    "NfUnfilled": "Unfilled f-Orbital Electrons",
    "NUnfilled": "Total Unfilled Orbital Electrons",
    "GSvolume_pa": "Ground-State Volume per Atom",
    "GSbandgap": "Ground-State Band Gap",
    "GSmagmom": "Ground-State Magnetic Moment",
    "SpaceGroupNumber": "Space Group Number",
}

MAGPIE_STAT_NAMES = {
    "minimum": "Minimum",
    "maximum": "Maximum",
    "range": "Range of",
    "mean": "Average",
    "avg_dev": "Deviation in",
    "mode": "Most Common",
}

def humanize_feature_name(raw_name):
    """Converts a raw matminer Magpie column name into a plain-English
    label, e.g. 'MagpieData mean GSbandgap' -> 'Average Ground-State Band Gap'.
    Non-Magpie names (like 'Temperature (K)') pass through unchanged."""
    if not isinstance(raw_name, str) or not raw_name.startswith("MagpieData "):
        return raw_name
    parts = raw_name.replace("MagpieData ", "", 1).split(" ", 1)
    if len(parts) != 2:
        return raw_name
    stat, prop = parts
    stat_label = MAGPIE_STAT_NAMES.get(stat, stat.replace("_", " ").title())
    prop_label = MAGPIE_PROPERTY_NAMES.get(prop, prop)
    return f"{stat_label} {prop_label}"


def get_zt_grade(zt):
    if zt >= 1.5:   return "Outstanding", "#22543D", "badge-excellent"
    elif zt >= 1.0: return "Excellent",   "#27AE60", "badge-excellent"
    elif zt >= 0.5: return "Good",        "#F39C12", "badge-good"
    else:           return "Poor",        "#C0392B", "badge-poor"

def get_zt_color(zt):
    if zt >= 1.5:   return "#22543D"
    elif zt >= 1.0: return "#27AE60"
    elif zt >= 0.5: return "#F39C12"
    else:           return "#C0392B"

# ── Loaders ───────────────────────────────────────────────────
@st.cache_data
def load_data():
    try: return pd.read_csv(DATA_PATH)
    except: return None

@st.cache_resource
def load_model():
    try:
        m  = joblib.load(MODEL_PATH)
        fc = joblib.load(FEAT_PATH)
        return m, fc
    except: return None, None

@st.cache_data
def load_shap():
    try:
        d = pd.read_csv(SHAP_PATH)
        if 'Feature' in d.columns:
            d['Feature'] = d['Feature'].apply(humanize_feature_name)
        return d
    except: return None

df              = load_data()
model, feat_cols = load_model()
model_loaded    = model is not None
shap_df         = load_shap()

# ── Featurizer (cached so it only initialises once) ───────────
@st.cache_resource
def get_featurizer():
    from matminer.featurizers.composition import ElementProperty
    featurizer = ElementProperty.from_preset("magpie")
    featurizer.set_n_jobs(1)  # avoid Windows multiprocessing re-execution issues
    return featurizer

# ── Feature generation ────────────────────────────────────────
def generate_features(comp_str, temperature_K):
    from pymatgen.core import Composition
    featurizer = get_featurizer()
    comp = Composition(comp_str)
    comp_df = pd.DataFrame({'composition_obj': [comp]})
    feat_df = featurizer.featurize_dataframe(comp_df, col_id='composition_obj', ignore_errors=True)
    feat_df = feat_df.drop(columns=['composition_obj'])
    feat_df['Temperature (K)'] = temperature_K
    return feat_df[feat_cols]

def predict_zt(comp_str, temperature_K):
    X = generate_features(comp_str, temperature_K)
    return float(model.predict(X)[0])

# ── Plotly layout helper ──────────────────────────────────────
def plotly_layout(fig, height=380):
    fig.update_layout(height=height, paper_bgcolor='white', plot_bgcolor='#F7F9FC',
                      font=dict(family='Source Sans 3'), margin=dict(l=40,r=20,t=30,b=40))
    fig.update_xaxes(gridcolor='#DDE3ED')
    fig.update_yaxes(gridcolor='#DDE3ED')
    return fig

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔥 Thermoelectric ZT")
    st.markdown("**Prediction Platform**")
    st.markdown("---")
    _pages = [
        "🏠  Overview",
        "🔍  Data Explorer",
        "🤖  ZT Prediction",
        "📊  Model Performance",
        "🔬  Feature Importance",
        "📰  Research Feed",
        "🔎  Custom Search",
        "📈  Digitizer",
        "💬  AI Assistant",
    ]
    page = st.radio("Navigation", _pages, label_visibility="collapsed")
    st.markdown("---")
    n_rows  = len(df) if df is not None else 0
    n_comps = df['Composition'].nunique() if df is not None else 0
    n_dois  = df['DOI'].notna().sum() if df is not None else 0
    st.markdown(f"**Dataset:** {n_rows:,} rows")
    st.markdown(f"**Unique compositions:** {n_comps}")
    st.markdown(f"**DOI-linked rows:** {n_dois:,}")
    if model_loaded: st.markdown("**Model:** Gradient Boosting (R²=0.956) ✅")
    else:            st.markdown("**Model:** Not loaded ⚠️")
    st.markdown("---")
    st.markdown("NTNU · Dept. of Mechanical & Industrial Engineering")
    st.markdown("Thermoelectric ZT ML Platform")

# ── Digitizer content (function, defined once here, used by both the
#    full page and the modal dialog triggered from Research Feed) ──
# ── Unit conversion table for the digitizer ─────────────────────
# Every property has one CANONICAL unit that all points get converted to
# before saving, so the whole database stays consistent regardless of
# what unit a given paper's figure happened to use.
#
# Temperature uses formulas (K = C + 273.15) since it's non-linear (has
# an offset); everything else uses simple multiplicative factors.
UNIT_CONVERSIONS = {
    "Temperature": {
        "canonical": "K",
        "units": {
            "K": {"to_canonical": lambda x: x},
            "°C": {"to_canonical": lambda x: x + 273.15},
            "°F": {"to_canonical": lambda x: (x - 32) * 5 / 9 + 273.15},
        },
    },
    "Electrical conductivity": {
        "canonical": "S/cm",
        "units": {
            "S/cm": {"to_canonical": lambda x: x},
            "S/m": {"to_canonical": lambda x: x * 0.01},
        },
    },
    "Seebeck coefficient": {
        "canonical": "μV/K",
        "units": {
            "μV/K": {"to_canonical": lambda x: x},
            "V/K": {"to_canonical": lambda x: x * 1e6},
            "mV/K": {"to_canonical": lambda x: x * 1e3},
        },
    },
    "Thermal conductivity": {
        "canonical": "W/(m·K)",
        "units": {
            "W/(m·K)": {"to_canonical": lambda x: x},
            "mW/(cm·K)": {"to_canonical": lambda x: x * 0.1},
        },
    },
    "Power factor": {
        "canonical": "W/(m·K²)",
        "units": {
            "W/(m·K²)": {"to_canonical": lambda x: x},
            "μW/(cm·K²)": {"to_canonical": lambda x: x * 1e-4},
        },
    },
    "Carrier concentration": {
        "canonical": "cm⁻³",
        "units": {
            "cm⁻³": {"to_canonical": lambda x: x},
            "m⁻³": {"to_canonical": lambda x: x * 1e-6},
        },
    },
    "ZT": {
        "canonical": "dimensionless",
        "units": {"dimensionless": {"to_canonical": lambda x: x}},
    },
}


def render_digitizer_content(linked_paper_id, linked_paper_title, in_dialog=False):
    """Renders the digitizer UI. Reused by both the full page and the modal dialog."""
    import streamlit.components.v1 as components
    import json as _json

    if not in_dialog:
        st.markdown("# 📈 Graph Digitizer")
        st.markdown('<div class="info-box">Load a plot image from a paper, calibrate the axes, click points on the curve, and extract real (x, y) data. Copy the CSV output below and save it to the database, linked to the source paper.</div>', unsafe_allow_html=True)

    if linked_paper_id:
        st.success(f"📎 Linked to paper: **{linked_paper_title}**")
        if not in_dialog:
            if st.button("Unlink / extract without a paper"):
                st.session_state.pop('_digitizer_paper_id', None)
                st.session_state.pop('_digitizer_paper_title', None)
                st.rerun()

        if supabase is not None:
            try:
                existing_check = supabase.table('extracted_data').select(
                    'figure_label, series_name, created_at'
                ).eq('paper_id', linked_paper_id).execute()
                if existing_check.data:
                    labels = [
                        f"{r.get('figure_label') or 'Untitled'}"
                        + (f" ({r['series_name']})" if r.get('series_name') else "")
                        for r in existing_check.data
                    ]
                    st.warning(
                        f"⚠️ This paper already has **{len(existing_check.data)}** extracted dataset(s): "
                        f"{', '.join(labels)}. If you're digitizing a different figure or series, "
                        f"continue below. If it's the same figure, check the 'Previously extracted data' "
                        f"section at the bottom before saving again to avoid duplicates."
                    )
            except Exception:
                pass
    elif not in_dialog:
        st.info("Not linked to a specific paper. Go to Research Feed and click 'Extract data' on a paper first if you want the data tied to a source.")

    st.markdown('<div class="section-title">1. Digitize the plot</div>', unsafe_allow_html=True)

    DIGITIZER_PATH = PROJECT_DIR + "digitizer.html"
    try:
        with open(DIGITIZER_PATH, "r", encoding="utf-8") as f:
            digitizer_html = f.read()
        components.html(digitizer_html, height=560 if in_dialog else 760, scrolling=True)
    except FileNotFoundError:
        st.error(f"digitizer.html not found at {DIGITIZER_PATH}. Make sure it's in the same folder as this app.")
        return

    st.markdown('<div class="section-title">2. Save extracted data</div>', unsafe_allow_html=True)
    st.markdown("Click **Copy CSV** in the digitizer above, then paste it here:")

    key_suffix = "_dlg" if in_dialog else ""

    X_AXIS_PROPERTIES = ["Temperature", "Composition x", "Doping concentration", "Time", "Other (no auto-conversion)"]
    Y_AXIS_PROPERTIES = ["ZT", "Electrical conductivity", "Seebeck coefficient",
                          "Thermal conductivity", "Power factor", "Carrier concentration",
                          "Other (no auto-conversion)"]

    dc1, dc2 = st.columns(2)
    with dc1:
        figure_label = st.text_input("Figure label", placeholder="e.g. Fig 3a", key=f"fig_label{key_suffix}")
        composition_field = st.text_input("Material / Composition", placeholder="e.g. Bi2Te3, Mg3Sb1.5Bi0.5", key=f"comp{key_suffix}")

        x_property = st.selectbox("X-axis property", X_AXIS_PROPERTIES, key=f"x_prop{key_suffix}")
        if x_property in UNIT_CONVERSIONS:
            x_units_available = list(UNIT_CONVERSIONS[x_property]["units"].keys())
            x_unit_original = st.selectbox(
                f"Unit used in the figure (converts to {UNIT_CONVERSIONS[x_property]['canonical']})",
                x_units_available, key=f"x_unit{key_suffix}"
            )
            x_label = f"{x_property} ({UNIT_CONVERSIONS[x_property]['canonical']})"
        else:
            x_unit_original = st.text_input("X-axis unit (no auto-conversion for custom properties)", placeholder="e.g. GPa", key=f"x_unit_custom{key_suffix}")
            x_label = st.text_input("Custom X-axis label", placeholder="e.g. Pressure", key=f"x_label_custom{key_suffix}")
    with dc2:
        property_type = st.selectbox(
            "Property type",
            ["Electrical conductivity", "Thermal conductivity", "Seebeck coefficient",
             "Power factor", "ZT", "Carrier concentration", "Other"],
            key=f"prop_type{key_suffix}"
        )

        y_property = st.selectbox("Y-axis property", Y_AXIS_PROPERTIES, key=f"y_prop{key_suffix}")
        if y_property in UNIT_CONVERSIONS:
            y_units_available = list(UNIT_CONVERSIONS[y_property]["units"].keys())
            y_unit_original = st.selectbox(
                f"Unit used in the figure (converts to {UNIT_CONVERSIONS[y_property]['canonical']})",
                y_units_available, key=f"y_unit{key_suffix}"
            )
            y_label = f"{y_property} ({UNIT_CONVERSIONS[y_property]['canonical']})"
        else:
            y_unit_original = st.text_input("Y-axis unit (no auto-conversion for custom properties)", placeholder="e.g. arb. units", key=f"y_unit_custom{key_suffix}")
            y_label = st.text_input("Custom Y-axis label", placeholder="e.g. Hall mobility", key=f"y_label_custom{key_suffix}")

        series_name = st.text_input("Series name (optional)", placeholder="e.g. sample A, undoped", key=f"series{key_suffix}")

    if x_property in UNIT_CONVERSIONS or y_property in UNIT_CONVERSIONS:
        st.caption("ℹ️ Points will be automatically converted to the platform's standard units before saving, so the database stays consistent regardless of what the original figure used.")

    comments = st.text_area("Comments (optional)", placeholder="e.g. sample annealed at 800°C, undoped reference", key=f"comments{key_suffix}", height=70)

    # --- Dynamic "+ Add field" section ---
    st.markdown("**Additional fields** (e.g. Quenching Method, Rolling Type, anything else)")

    fields_key = f"custom_fields{key_suffix}"
    if fields_key not in st.session_state:
        st.session_state[fields_key] = []  # list of (name, value) pairs

    for i, (fname, fval) in enumerate(st.session_state[fields_key]):
        fc1, fc2, fc3 = st.columns([2, 3, 0.5])
        with fc1:
            new_name = st.text_input("Field name", value=fname, key=f"{fields_key}_name_{i}", label_visibility="collapsed", placeholder="Field name")
        with fc2:
            new_val = st.text_input("Field value", value=fval, key=f"{fields_key}_val_{i}", label_visibility="collapsed", placeholder="Value")
        with fc3:
            delete_clicked = st.button("✕", key=f"{fields_key}_del_{i}")
        if delete_clicked:
            st.session_state[fields_key].pop(i)
            # The list just shrank, so every index after this one is now
            # stale (an off-by-one shift) — writing to them would crash
            # with IndexError, and rendering them would show wrong data.
            # Break immediately; the button click already triggers
            # Streamlit's natural rerun, which redraws the correct,
            # updated list from scratch on the next pass.
            break
        st.session_state[fields_key][i] = (new_name, new_val)

    if st.button("➕ Add field", key=f"{fields_key}_add"):
        st.session_state[fields_key].append(("", ""))
        # Same reasoning — no explicit st.rerun() needed or wanted here.

    pasted_csv = st.text_area("Pasted CSV data", height=150, placeholder="x,y\n300,0.45\n350,0.52\n...", key=f"csv{key_suffix}")

    if st.button("💾 Save extracted data", use_container_width=True, type="primary", key=f"save_btn{key_suffix}"):
        if not pasted_csv.strip():
            st.warning("Paste the CSV data first.")
        elif supabase is None:
            st.error("Supabase not connected.")
        else:
            try:
                lines = [l.strip() for l in pasted_csv.strip().split('\n') if l.strip()]
                data_lines = lines
                first_cell = lines[0].split(',')[0].strip()
                try:
                    float(first_cell)
                except ValueError:
                    data_lines = lines[1:]

                points = []
                for line in data_lines:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        x_val = float(parts[0].strip())
                        y_val = float(parts[1].strip())
                        points.append({'x': x_val, 'y': y_val})

                if not points:
                    st.warning("No valid data points found in the pasted CSV.")
                else:
                    # --- Convert to canonical units before saving ---
                    x_canonical_unit = x_unit_original
                    y_canonical_unit = y_unit_original
                    conversion_notes = []

                    if x_property in UNIT_CONVERSIONS:
                        x_conv_fn = UNIT_CONVERSIONS[x_property]["units"][x_unit_original]["to_canonical"]
                        for p in points:
                            p['x'] = round(x_conv_fn(p['x']), 6)
                        x_canonical_unit = UNIT_CONVERSIONS[x_property]["canonical"]
                        if x_unit_original != x_canonical_unit:
                            conversion_notes.append(f"X: {x_unit_original} → {x_canonical_unit}")

                    if y_property in UNIT_CONVERSIONS:
                        y_conv_fn = UNIT_CONVERSIONS[y_property]["units"][y_unit_original]["to_canonical"]
                        for p in points:
                            p['y'] = round(y_conv_fn(p['y']), 6)
                        y_canonical_unit = UNIT_CONVERSIONS[y_property]["canonical"]
                        if y_unit_original != y_canonical_unit:
                            conversion_notes.append(f"Y: {y_unit_original} → {y_canonical_unit}")

                    if conversion_notes:
                        st.info(f"🔄 Converted units — {', '.join(conversion_notes)}")

                    record = {
                        'paper_id': linked_paper_id,
                        'figure_label': figure_label or None,
                        'composition': composition_field or None,
                        'property_type': property_type,
                        'x_label': x_label or None,
                        'y_label': y_label or None,
                        'x_unit': x_canonical_unit,
                        'y_unit': y_canonical_unit,
                        'x_unit_original': x_unit_original,
                        'y_unit_original': y_unit_original,
                        'series_name': series_name or None,
                        'comments': comments or None,
                        'points_json': _json.dumps(points),
                    }
                    insert_result = supabase.table('extracted_data').insert(record).execute()
                    new_id = insert_result.data[0]['id']

                    # Save any custom fields into the flexible fields table
                    custom_fields = [
                        {'extracted_data_id': new_id, 'field_name': fname.strip(), 'field_value': fval.strip()}
                        for fname, fval in st.session_state[fields_key]
                        if fname.strip()
                    ]
                    if custom_fields:
                        supabase.table('extracted_data_fields').insert(custom_fields).execute()

                    st.session_state[fields_key] = []  # reset for next entry
                    st.cache_data.clear()
                    st.success(f"Saved {len(points)} points to the database"
                              + (f" with {len(custom_fields)} custom field(s)." if custom_fields else "."))
                    st.dataframe(pd.DataFrame(points), use_container_width=True, height=200)
            except Exception as e:
                st.error(f"Could not parse/save data: {e}")

    if linked_paper_id and supabase is not None:
        st.markdown('<div class="section-title">Previously extracted data for this paper</div>', unsafe_allow_html=True)
        try:
            existing = supabase.table('extracted_data').select('*').eq('paper_id', linked_paper_id).execute()
            if existing.data:
                for row in existing.data:
                    pts = _json.loads(row['points_json'])
                    header = f"**{row.get('figure_label') or 'Untitled figure'}**"
                    if row.get('composition'):
                        header += f" — {row['composition']}"
                    if row.get('property_type'):
                        header += f" ({row['property_type']})"
                    st.markdown(header)
                    st.caption(f"{row.get('x_label') or 'x'} vs {row.get('y_label') or 'y'} "
                               f"({len(pts)} points)"
                               + (f" — *{row['series_name']}*" if row.get('series_name') else ""))

                    unit_notes = []
                    if row.get('x_unit_original') and row.get('x_unit') and row['x_unit_original'] != row['x_unit']:
                        unit_notes.append(f"X converted from {row['x_unit_original']} → {row['x_unit']}")
                    if row.get('y_unit_original') and row.get('y_unit') and row['y_unit_original'] != row['y_unit']:
                        unit_notes.append(f"Y converted from {row['y_unit_original']} → {row['y_unit']}")
                    if unit_notes:
                        st.caption(f"🔄 {' · '.join(unit_notes)}")

                    if row.get('comments'):
                        st.caption(f"💬 {row['comments']}")

                    # Show any custom fields attached to this extraction
                    try:
                        custom = supabase.table('extracted_data_fields').select('field_name, field_value').eq('extracted_data_id', row['id']).execute()
                        if custom.data:
                            field_str = " · ".join(f"{f['field_name']}: {f['field_value']}" for f in custom.data)
                            st.caption(f"🏷️ {field_str}")
                    except Exception:
                        pass
                    st.markdown("---")
            else:
                st.caption("No data extracted yet for this paper.")
        except Exception as e:
            st.caption(f"Could not load existing data: {e}")


# Modal dialog version — opened directly when "Extract data" is clicked on a paper card
if hasattr(st, "dialog"):
    @st.dialog("📈 Graph Digitizer", width="large")
    def digitizer_dialog(paper_id, paper_title):
        render_digitizer_content(paper_id, paper_title, in_dialog=True)
else:
    digitizer_dialog = None  # older Streamlit version — falls back to full page navigation

# ══════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == "🏠  Overview":
    st.markdown("# Thermoelectric ZT Prediction Platform")
    st.markdown('<div class="info-box">A machine learning platform for predicting and screening thermoelectric figure of merit (ZT) across diverse material families. Trained on 3,841 data points from 314 unique compositions using matminer Magpie elemental descriptors. Deployed model: Gradient Boosting (R²=0.956 random split, R²=0.820 composition-grouped — best of 5 algorithms benchmarked under honest, composition-grouped evaluation).</div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col,(val,label) in zip([c1,c2,c3,c4],[
        ("3,841","Total Data Points"),
        ("314","Unique Compositions"),
        ("62","Published Papers"),
        ("0.820","Grouped CV R²"),
    ]):
        col.markdown(f'<div class="stat-card"><div class="stat-value">{val}</div><div class="stat-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Platform Features</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    for col,(icon,title,desc) in zip([c1,c2,c3],[
        ("🔍","Data Explorer","Browse 3,841 ZT measurements. Filter by composition, ZT range, and temperature."),
        ("🤖","ZT Prediction","Predict ZT for any composition at any temperature using Gradient Boosting + matminer Magpie features."),
        ("💬","AI Assistant","Agentic RAG chat that predicts ZT, searches literature, and cites sources for grounded answers."),
    ]):
        col.markdown(f'<div style="background:white;border:1px solid #DDE3ED;border-radius:8px;padding:16px;height:120px;"><div style="font-size:1.5rem;">{icon}</div><div style="font-weight:600;color:#1B2A4A;margin-top:4px;">{title}</div><div style="font-size:0.83rem;color:#5A6478;margin-top:4px;">{desc}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">ZT Grade Guide</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    for col,(grade,rng,color,ex) in zip([c1,c2,c3,c4],[
        ("Outstanding","ZT ≥ 1.5","#22543D","GeTe, SnSe (single crystal)"),
        ("Excellent","1.0 ≤ ZT < 1.5","#27AE60","PbTe, BiSbTe alloys"),
        ("Good","0.5 ≤ ZT < 1.0","#F39C12","CoSb3, half-Heuslers"),
        ("Poor","ZT < 0.5","#C0392B","Undoped/unoptimised"),
    ]):
        col.markdown(f'<div style="background:white;border:1px solid #DDE3ED;border-radius:8px;padding:14px;border-top:4px solid {color};"><div style="font-weight:700;color:{color};font-size:1rem;">{grade}</div><div style="font-size:1.1rem;font-weight:600;color:#1B2A4A;">{rng}</div><div style="font-size:0.78rem;color:#5A6478;margin-top:4px;">e.g. {ex}</div></div>', unsafe_allow_html=True)

    # Dataset insights
    if df is not None:
        import plotly.graph_objects as go
        st.markdown('<div class="section-title">Dataset Insights</div>', unsafe_allow_html=True)
        ov1, ov2 = st.columns(2)

        with ov1:
            zt_hist = go.Figure()
            zt_hist.add_trace(go.Histogram(x=df['ZT'], nbinsx=40, marker_color='#C0392B', opacity=0.8))
            zt_hist.update_layout(title="ZT Distribution", xaxis_title="ZT", yaxis_title="Count", height=300,
                                  paper_bgcolor='white', plot_bgcolor='#F7F9FC', font=dict(family='Source Sans 3'),
                                  margin=dict(l=40,r=20,t=40,b=40))
            zt_hist.update_xaxes(gridcolor='#DDE3ED')
            zt_hist.update_yaxes(gridcolor='#DDE3ED')
            st.plotly_chart(zt_hist, use_container_width=True)

        with ov2:
            temp_hist = go.Figure()
            temp_hist.add_trace(go.Histogram(x=df['Temperature (K)'], nbinsx=40, marker_color='#2E5FA3', opacity=0.8))
            temp_hist.update_layout(title="Temperature Distribution", xaxis_title="Temperature (K)", yaxis_title="Count",
                                    height=300, paper_bgcolor='white', plot_bgcolor='#F7F9FC',
                                    font=dict(family='Source Sans 3'), margin=dict(l=40,r=20,t=40,b=40))
            temp_hist.update_xaxes(gridcolor='#DDE3ED')
            temp_hist.update_yaxes(gridcolor='#DDE3ED')
            st.plotly_chart(temp_hist, use_container_width=True)

        # ZT vs Temperature scatter
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=df['Temperature (K)'], y=df['ZT'],
            mode='markers',
            marker=dict(color=df['ZT'], colorscale='RdYlGn', size=4, opacity=0.5,
                        colorbar=dict(title='ZT')),
            hovertemplate='<b>%{text}</b><br>T=%{x:.0f} K<br>ZT=%{y:.3f}<extra></extra>',
            text=df['Composition']
        ))
        fig_scatter.update_layout(title="ZT vs Temperature (all compositions)",
                                  xaxis_title="Temperature (K)", yaxis_title="ZT",
                                  height=350, paper_bgcolor='white', plot_bgcolor='#F7F9FC',
                                  font=dict(family='Source Sans 3'), margin=dict(l=40,r=20,t=40,b=40))
        fig_scatter.update_xaxes(gridcolor='#DDE3ED')
        fig_scatter.update_yaxes(gridcolor='#DDE3ED')
        st.plotly_chart(fig_scatter, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE: DATA EXPLORER
# ══════════════════════════════════════════════════════════════
elif page == "🔍  Data Explorer":
    import plotly.graph_objects as go
    st.markdown("# Data Explorer")
    st.markdown('<div class="info-box">Browse 3,841 ZT measurements from 314 unique compositions. Search by composition or element, filter by temperature, ZT range, and publication year.</div>', unsafe_allow_html=True)

    if df is None:
        st.error("Dataset not loaded."); st.stop()

    has_year = 'Publication_Year' in df.columns

    c1,c2,c3,c4 = st.columns(4)
    with c1: search = st.text_input("Search composition", placeholder="e.g. Bi2Te3, PbTe, Mg")
    with c2: zt_min = st.number_input("Min ZT", value=0.0, min_value=0.0, max_value=3.0, step=0.1)
    with c3: zt_max = st.number_input("Max ZT", value=3.0, min_value=0.0, max_value=3.0, step=0.1)
    with c4: temp_filter = st.slider("Temperature range (K)", 0, 1400, (0, 1400))

    if has_year:
        year_series = df['Publication_Year'].dropna()
        if len(year_series) > 0:
            year_min_data, year_max_data = int(year_series.min()), int(year_series.max())
            year_filter = st.slider("Publication year range", year_min_data, year_max_data,
                                     (year_min_data, year_max_data))
        else:
            year_filter = None
    else:
        year_filter = None
        st.caption("ℹ️ Publication year data not available — run 20_add_publication_year_to_main_dataset.py to enable this filter.")

    hide_no_doi = st.checkbox("Hide entries without a linked publication (DOI)", value=True)

    df_exp = df.copy()
    if hide_no_doi:
        df_exp = df_exp[df_exp['DOI'].notna()]
    if search:
        df_exp = df_exp[df_exp['Composition'].str.contains(search, case=False, na=False)]
    df_exp = df_exp[(df_exp['ZT'] >= zt_min) & (df_exp['ZT'] <= zt_max)]
    df_exp = df_exp[(df_exp['Temperature (K)'] >= temp_filter[0]) & (df_exp['Temperature (K)'] <= temp_filter[1])]
    if has_year and year_filter is not None:
        # Rows with no known year are excluded once a year filter is actively narrowed,
        # but included by default when the slider covers the full range (nothing to hide)
        full_range = (year_filter[0] == year_min_data and year_filter[1] == year_max_data)
        if full_range:
            df_exp = df_exp[(df_exp['Publication_Year'].isna()) |
                            ((df_exp['Publication_Year'] >= year_filter[0]) & (df_exp['Publication_Year'] <= year_filter[1]))]
        else:
            df_exp = df_exp[(df_exp['Publication_Year'] >= year_filter[0]) & (df_exp['Publication_Year'] <= year_filter[1])]

    st.markdown(f'<div class="info-box">Showing <b>{len(df_exp):,}</b> rows · <b>{df_exp["Composition"].nunique()}</b> unique compositions</div>', unsafe_allow_html=True)

    # ZT vs T — per-composition lines when few compositions match, otherwise a
    # scatter colored by publication year (or by ZT if year isn't available)
    if len(df_exp) > 0:
        if df_exp['Composition'].nunique() <= 20:
            fig = go.Figure()
            for comp, grp in df_exp.groupby('Composition'):
                grp = grp.sort_values('Temperature (K)')
                fig.add_trace(go.Scatter(x=grp['Temperature (K)'], y=grp['ZT'], mode='lines+markers',
                                         name=comp, marker=dict(size=5),
                                         hovertemplate=f'<b>{comp}</b><br>T=%{{x:.0f}} K<br>ZT=%{{y:.3f}}<extra></extra>'))
            fig.update_layout(title="ZT vs Temperature", xaxis_title="Temperature (K)", yaxis_title="ZT",
                              height=400, paper_bgcolor='white', plot_bgcolor='#F7F9FC',
                              font=dict(family='Source Sans 3'), margin=dict(l=40,r=20,t=40,b=40))
        else:
            color_col = 'Publication_Year' if has_year else 'ZT'
            color_label = 'Publication Year' if has_year else 'ZT'
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_exp['Temperature (K)'], y=df_exp['ZT'], mode='markers',
                marker=dict(color=df_exp[color_col], colorscale='Viridis', size=5, opacity=0.6,
                            colorbar=dict(title=color_label)),
                hovertemplate='<b>%{text}</b><br>T=%{x:.0f} K<br>ZT=%{y:.3f}<extra></extra>',
                text=df_exp['Composition']
            ))
            fig.update_layout(title=f"ZT vs Temperature (colored by {color_label.lower()})",
                              xaxis_title="Temperature (K)", yaxis_title="ZT",
                              height=400, paper_bgcolor='white', plot_bgcolor='#F7F9FC',
                              font=dict(family='Source Sans 3'), margin=dict(l=40,r=20,t=40,b=40))
        fig.update_xaxes(gridcolor='#DDE3ED')
        fig.update_yaxes(gridcolor='#DDE3ED')
        st.plotly_chart(fig, use_container_width=True)

    # Table
    show_cols = ['Composition', 'ZT', 'Temperature (K)', 'DOI']
    if has_year:
        show_cols.append('Publication_Year')
    st.dataframe(df_exp[show_cols].reset_index(drop=True), use_container_width=True, height=400)

    # Download
    csv = df_exp[show_cols].to_csv(index=False)
    st.download_button("📥 Download filtered data", data=csv,
                       file_name="filtered_zt_data.csv", mime="text/csv")

# ══════════════════════════════════════════════════════════════
# PAGE: ZT PREDICTION
# ══════════════════════════════════════════════════════════════
elif page == "🤖  ZT Prediction":
    import plotly.graph_objects as go
    st.markdown("# ZT Prediction")
    st.markdown('<div class="info-box">Enter any thermoelectric composition and temperature to predict ZT using the Gradient Boosting model trained on matminer Magpie features.</div>', unsafe_allow_html=True)

    if not model_loaded:
        st.error("Model not loaded. Place zt_gradboost_model.pkl and feature_cols.pkl in the app folder.")
        st.stop()

    c1, c2 = st.columns([2, 1])
    with c1: comp_input = st.text_input("Composition", value="Bi2Te3", placeholder="e.g. PbTe, Mg3Sb2, CoSb3")
    with c2: temp_input = st.number_input("Temperature (K)", value=500, min_value=50, max_value=1400, step=25)

    predict_btn = st.button("🔮 Predict ZT", use_container_width=True)

    if predict_btn:
        with st.spinner("Generating features and predicting..."):
            try:
                zt_pred = predict_zt(comp_input, temp_input)
                grade, color, badge_class = get_zt_grade(zt_pred)
                st.session_state['_pred_comp'] = comp_input
                st.session_state['_pred_temp'] = temp_input
                st.session_state['_pred_zt']   = zt_pred
                st.session_state['_pred_grade'] = grade
                st.session_state['_pred_color'] = color
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    if '_pred_zt' in st.session_state:
        zt   = st.session_state['_pred_zt']
        comp = st.session_state['_pred_comp']
        temp = st.session_state['_pred_temp']
        grade = st.session_state['_pred_grade']
        color = st.session_state['_pred_color']

        st.markdown(f"""
        <div class="result-box">
            <div style="font-size:0.85rem;color:#5A6478;text-transform:uppercase;letter-spacing:0.05em;">{comp} at {temp} K</div>
            <div class="result-zt" style="color:{color};">{zt:.4f}</div>
            <div style="font-size:1rem;color:#5A6478;">Predicted ZT</div>
            <div style="margin-top:8px;"><span style="background:{color};color:white;padding:4px 16px;border-radius:12px;font-size:0.85rem;font-weight:600;">{grade}</span></div>
        </div>
        """, unsafe_allow_html=True)

        # Compare to database
        if df is not None:
            db_match = df[df['Composition'] == comp]
            if len(db_match) > 0:
                st.markdown('<div class="section-title">Experimental Data (from database)</div>', unsafe_allow_html=True)
                fig = go.Figure()
                db_sorted = db_match.sort_values('Temperature (K)')
                fig.add_trace(go.Scatter(x=db_sorted['Temperature (K)'], y=db_sorted['ZT'],
                                         mode='lines+markers', name='Experimental',
                                         line=dict(color='#1B2A4A', width=2),
                                         marker=dict(size=6)))
                fig.add_trace(go.Scatter(x=[temp], y=[zt], mode='markers', name='ML Prediction',
                                         marker=dict(color=color, size=14, symbol='star')))
                fig.update_layout(title=f"ZT vs Temperature: {comp}",
                                  xaxis_title="Temperature (K)", yaxis_title="ZT",
                                  height=350, paper_bgcolor='white', plot_bgcolor='#F7F9FC',
                                  font=dict(family='Source Sans 3'), margin=dict(l=40,r=20,t=40,b=40))
                fig.update_xaxes(gridcolor='#DDE3ED')
                fig.update_yaxes(gridcolor='#DDE3ED')
                st.plotly_chart(fig, use_container_width=True)

        # Temperature sweep
        st.markdown('<div class="section-title">ZT vs Temperature Sweep</div>', unsafe_allow_html=True)
        with st.spinner("Running temperature sweep..."):
            try:
                temps = np.arange(300, 1050, 50)
                preds = [predict_zt(comp, t) for t in temps]
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=temps, y=preds, mode='lines+markers',
                                          line=dict(color='#C0392B', width=2.5),
                                          marker=dict(size=6),
                                          hovertemplate='T=%{x} K<br>ZT=%{y:.4f}<extra></extra>'))
                # ZT grade lines
                for val, label, lcolor in [(1.5,"Outstanding","#22543D"),(1.0,"Excellent","#27AE60"),(0.5,"Good","#F39C12")]:
                    fig2.add_hline(y=val, line_dash="dot", line_color=lcolor,
                                   annotation_text=label, annotation_position="right",
                                   annotation_font_size=9)
                fig2.update_layout(title=f"Predicted ZT vs Temperature: {comp}",
                                   xaxis_title="Temperature (K)", yaxis_title="Predicted ZT",
                                   height=380, paper_bgcolor='white', plot_bgcolor='#F7F9FC',
                                   font=dict(family='Source Sans 3'), margin=dict(l=40,r=150,t=40,b=40))
                fig2.update_xaxes(gridcolor='#DDE3ED')
                fig2.update_yaxes(gridcolor='#DDE3ED')
                st.plotly_chart(fig2, use_container_width=True)

                sweep_df = pd.DataFrame({'Temperature (K)': temps, 'Predicted ZT': preds})
                st.download_button("📥 Download sweep data", sweep_df.to_csv(index=False),
                                   file_name=f"zt_sweep_{comp}.csv", mime="text/csv")
            except Exception as e:
                st.warning(f"Could not run sweep: {e}")

        # SHAP top features
        if shap_df is not None:
            st.markdown('<div class="section-title">Top Driving Features (SHAP)</div>', unsafe_allow_html=True)
            top_shap = shap_df.head(10)
            import plotly.graph_objects as go
            fig_s = go.Figure(go.Bar(
                x=top_shap['Mean_Abs_SHAP'][::-1],
                y=top_shap['Feature'][::-1],
                orientation='h',
                marker_color='#C0392B',
            ))
            fig_s.update_layout(title="Top 10 Features by Mean |SHAP|",
                                 xaxis_title="Mean |SHAP value|",
                                 height=350, paper_bgcolor='white', plot_bgcolor='#F7F9FC',
                                 font=dict(family='Source Sans 3'), margin=dict(l=200,r=20,t=40,b=40))
            st.plotly_chart(fig_s, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════
elif page == "📊  Model Performance":
    import plotly.graph_objects as go
    st.markdown("# Model Performance")
    st.markdown('<div class="info-box">Performance of 5 ML models evaluated on 3,841 data points from 314 unique thermoelectric compositions, using both random and composition-grouped 5-fold cross-validation.</div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col,(val,label) in zip([c1,c2,c3,c4],[
        ("0.962","Best R² — Random split"),
        ("0.820","Best R² — Grouped CV"),
        ("5","Models Benchmarked"),
        ("132","Magpie Features"),
    ]):
        col.markdown(f'<div class="stat-card"><div class="stat-value">{val}</div><div class="stat-label">{label}</div></div>', unsafe_allow_html=True)

    dm_random = pd.DataFrame({
        'Model':   ['XGBoost','RandomForest','GradientBoosting','LightGBM','SVR'],
        'R2_mean': [0.962,    0.961,          0.956,             0.952,     0.876],
        'R2_std':  [0.009,    0.010,          0.009,             0.013,     0.015],
    })
    dm_grouped = pd.DataFrame({
        'Model':   ['GradientBoosting','LightGBM','RandomForest','XGBoost','SVR'],
        'R2_mean': [0.820,              0.810,     0.802,         0.798,    0.683],
        'R2_std':  [0.012,              0.029,     0.031,         0.042,    0.157],
    })

    tab1, tab2 = st.tabs(["Random Split (reference)", "Grouped CV (honest)"])

    for tab, dm, title in [(tab1, dm_random, "Random 80/20 Split"),
                            (tab2, dm_grouped, "Composition-Grouped 5-Fold CV")]:
        with tab:
            c1, c2 = st.columns(2)
            with c1:
                best_r2 = dm['R2_mean'].max()
                fig = go.Figure(go.Bar(
                    x=dm['Model'], y=dm['R2_mean'],
                    error_y=dict(type='data', array=dm['R2_std'], visible=True),
                    marker_color=['#C0392B' if r == best_r2 else '#E8A89B' for r in dm['R2_mean']],
                    text=dm['R2_mean'].round(3), textposition='outside',
                ))
                fig.update_layout(title=f"R² — {title}", yaxis_title="R²",
                                  yaxis_range=[0, 1.05], xaxis_tickangle=-20, showlegend=False,
                                  height=380, paper_bgcolor='white', plot_bgcolor='#F7F9FC',
                                  font=dict(family='Source Sans 3'), margin=dict(l=40,r=20,t=40,b=60))
                fig.update_xaxes(gridcolor='#DDE3ED')
                fig.update_yaxes(gridcolor='#DDE3ED')
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                fig2 = go.Figure(go.Bar(
                    x=dm['Model'], y=dm['R2_std'],
                    marker_color='#2E5FA3', opacity=0.8,
                    text=dm['R2_std'].round(3), textposition='outside',
                ))
                fig2.update_layout(title=f"R² Std (stability) — {title}",
                                   yaxis_title="R² Standard Deviation",
                                   xaxis_tickangle=-20, showlegend=False,
                                   height=380, paper_bgcolor='white', plot_bgcolor='#F7F9FC',
                                   font=dict(family='Source Sans 3'), margin=dict(l=40,r=20,t=40,b=60))
                fig2.update_xaxes(gridcolor='#DDE3ED')
                fig2.update_yaxes(gridcolor='#DDE3ED')
                st.plotly_chart(fig2, use_container_width=True)

            st.dataframe(dm.set_index('Model'), use_container_width=True)

    st.markdown("""
| Metric | Meaning | Note |
|--------|---------|------|
| **Random split R²** | Rows randomly split — same composition can be in train and test | Optimistic, inflated by data leakage |
| **Grouped CV R²** | Each composition is entirely in train OR test | Honest estimate of generalization to new materials |
| **R² Std** | Variation across folds — lower = more stable | Should be < 0.05 |
""")

# ══════════════════════════════════════════════════════════════
# PAGE: FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════
elif page == "🔬  Feature Importance":
    import plotly.graph_objects as go
    st.markdown("# Feature Importance (SHAP)")
    st.markdown('<div class="info-box">SHAP (SHapley Additive exPlanations) values showing which elemental descriptors most strongly drive ZT predictions. Based on the Gradient Boosting model trained on a random split.</div>', unsafe_allow_html=True)

    if shap_df is None:
        st.warning("SHAP importance file not found. Run 04_shap_analysis_random_split.py first to generate shap_feature_importance.csv.")
        st.stop()

    c1, c2 = st.columns([2, 1])
    with c2:
        n_features = st.slider("Number of features to show", 5, min(50, len(shap_df)), 20)

    top_n = shap_df.head(n_features)

    fig = go.Figure(go.Bar(
        x=top_n['Mean_Abs_SHAP'][::-1],
        y=top_n['Feature'][::-1],
        orientation='h',
        marker_color='#C0392B',
        opacity=0.85,
    ))
    fig.update_layout(
        title=f"Top {n_features} Features by Mean |SHAP value|",
        xaxis_title="Mean |SHAP value| (impact on ZT prediction)",
        height=max(400, n_features * 22),
        paper_bgcolor='white', plot_bgcolor='#F7F9FC',
        font=dict(family='Source Sans 3'),
        margin=dict(l=260, r=20, t=40, b=40)
    )
    fig.update_xaxes(gridcolor='#DDE3ED')
    fig.update_yaxes(gridcolor='#DDE3ED')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Full Ranking Table</div>', unsafe_allow_html=True)
    st.dataframe(shap_df.reset_index(drop=True), use_container_width=True, height=400)
    st.download_button("📥 Download SHAP ranking", shap_df.to_csv(index=False),
                       file_name="shap_feature_importance.csv", mime="text/csv")

# ══════════════════════════════════════════════════════════════
# PAGE: RESEARCH FEED
# ══════════════════════════════════════════════════════════════
elif page == "📰  Research Feed":
    st.markdown("# 📰 Research Feed")
    st.markdown('<div class="info-box">Daily-updating feed of thermoelectric papers from arXiv, filtered by relevance. Papers are fetched server-side (fetch_arxiv.py) — this page only reads from the database.</div>', unsafe_allow_html=True)

    if supabase is None:
        st.error("Supabase not connected. Check SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY in your .env file.")
        st.stop()

    feed_df = load_papers()
    saved_ids = load_saved_ids()
    extraction_status = load_extraction_status()

    if feed_df.empty:
        st.warning("No papers in the database yet. Run fetch_arxiv.py to populate the feed.")
        st.stop()

    fc1, fc2, fc3 = st.columns([3, 1, 1])
    with fc1:
        feed_search = st.text_input(
            "🔍 Search your keywords",
            placeholder="e.g. Bi2Te3, half-Heusler, band convergence (comma-separated = match any)"
        )
    with fc2:
        feed_sort = st.selectbox("Sort", ["Newest first", "Most relevant"])
    with fc3:
        feed_tab = st.radio("View", [f"Feed ({len(feed_df)})", f"Saved ({len(saved_ids)})"], label_visibility="collapsed")

    feed_view = feed_df.copy()
    if "Saved" in feed_tab:
        feed_view = feed_view[feed_view['id'].isin(saved_ids)]

    if feed_search.strip():
        search_terms = [t.strip() for t in feed_search.split(',') if t.strip()]
        search_cols = ['title', 'authors', 'abstract', 'categories']
        if 'journal_name' in feed_view.columns:
            search_cols.append('journal_name')

        combined_mask = pd.Series(False, index=feed_view.index)
        for term in search_terms:
            term_mask = pd.Series(False, index=feed_view.index)
            for col in search_cols:
                term_mask = term_mask | feed_view[col].astype(str).str.contains(term, case=False, na=False)
            combined_mask = combined_mask | term_mask

        feed_view = feed_view[combined_mask]
        st.caption(f"Matching any of: {', '.join(search_terms)}")

    if feed_sort == "Most relevant":
        feed_view = feed_view.sort_values('score', ascending=False)
    else:
        feed_view = feed_view.sort_values('pub_date', ascending=False)

    st.markdown(f"**{len(feed_view)}** papers")
    st.markdown("---")

    if feed_view.empty:
        st.info("No papers match your filters." if "Feed" in feed_tab else "Nothing saved yet — click 'Save' on a paper below.")
    else:
        for _, p in feed_view.iterrows():
            is_saved = p['id'] in saved_ids
            rel_color = relevance_color(p['score'])
            src_class = 'arxiv' if p['source'] == 'arxiv' else 'journal'
            journal_name = p.get('journal_name') if 'journal_name' in p and pd.notna(p.get('journal_name')) else None
            src_label = 'arXiv' if p['source'] == 'arxiv' else (journal_name or 'Journal')
            extractions = extraction_status.get(p['id'], [])
            extracted_badge = (
                f'<span style="background:#C6F6D5;color:#22543D;font-size:10px;font-weight:700;'
                f'padding:2px 8px;border-radius:10px;margin-left:6px;">✓ {len(extractions)} extracted</span>'
                if extractions else ''
            )

            card_html = (
                f'<div class="feed-card">'
                f'<div class="feed-meta">'
                f'<span class="feed-src {src_class}">{src_label}</span>'
                f'<span class="feed-date">{relative_date(p["pub_date"])}</span>'
                f'<span class="feed-rel" style="color:{rel_color};">● {int(p["score"]*100)}% relevant</span>'
                f'{extracted_badge}'
                f'</div>'
                f'<div class="feed-title">{p["title"]}</div>'
                f'<div class="feed-authors">{p["authors"]}</div>'
                f'</div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)

            with st.expander("Read abstract"):
                st.write(p['abstract'])

            bcol1, bcol2, bcol3, bcol4 = st.columns([1, 1, 1, 1])
            with bcol1:
                if st.button("💾 Saved" if is_saved else "💾 Save", key=f"save_{p['id']}", use_container_width=True):
                    toggle_save(p['id'], is_saved)
                    st.rerun()
            with bcol2:
                if st.button("📊 Extract data", key=f"extract_{p['id']}", use_container_width=True):
                    st.session_state['_digitizer_paper_id'] = p['id']
                    st.session_state['_digitizer_paper_title'] = p['title']
                    if digitizer_dialog is not None:
                        digitizer_dialog(p['id'], p['title'])
                    else:
                        st.info("Go to the Digitizer page to extract data from this paper's figures. "
                                "(Modal popup needs a newer Streamlit version — run `pip install --upgrade streamlit`.)")
            with bcol3:
                st.link_button("🔗 Source", p['pdf_url'], use_container_width=True)
            with bcol4:
                st.caption(f"ID: {p['arxiv_id']}")

            st.markdown("---")

# ══════════════════════════════════════════════════════════════
# PAGE: CUSTOM SEARCH (on-demand backfill)
# ══════════════════════════════════════════════════════════════
elif page == "🔎  Custom Search":
    import re as _re

    st.markdown("# 🔎 Custom Search")
    st.markdown('<div class="info-box">Search arXiv and OpenAlex directly for any keyword and date range — not limited to the daily auto-fetched feed. Preview results, then choose which ones to add to the shared database.</div>', unsafe_allow_html=True)

    if supabase is None:
        st.error("Supabase not connected.")
        st.stop()

    cs1, cs2, cs3 = st.columns([2, 1, 1])
    with cs1:
        cs_keyword = st.text_input("Keyword", placeholder="e.g. GeTe thermoelectric, band engineering")
    with cs2:
        cs_start = st.date_input("From date", value=date(2020, 1, 1))
    with cs3:
        cs_end = st.date_input("To date", value=date.today())

    cs4, cs5 = st.columns(2)
    with cs4:
        cs_sources = st.multiselect("Sources", ["arXiv", "OpenAlex"], default=["arXiv", "OpenAlex"])
    with cs5:
        cs_max = st.slider("Max results per source", 5, 50, 20)

    def _search_arxiv_custom(keyword, start_d, end_d, max_results):
        import feedparser
        start_str = start_d.strftime('%Y%m%d0000')
        end_str = end_d.strftime('%Y%m%d2359')
        query = f'all:{keyword} AND submittedDate:[{start_str} TO {end_str}]'
        params = {
            'search_query': query,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending',
            'max_results': max_results,
        }
        url = "http://export.arxiv.org/api/query?" + requests.compat.urlencode(params)
        feed = feedparser.parse(url)
        results = []
        for entry in feed.entries:
            arxiv_id = entry.id.split('/abs/')[-1]
            title = _re.sub(r'\s+', ' ', entry.title).strip()
            abstract = _re.sub(r'\s+', ' ', entry.summary).strip()
            authors = ', '.join(a.name for a in entry.authors) if hasattr(entry, 'authors') else ''
            pub_date = entry.published[:10]
            pdf_url = next((l.href for l in entry.links if l.type == 'application/pdf'), '')
            results.append({
                'arxiv_id': arxiv_id, 'source': 'arxiv', 'title': title,
                'authors': authors, 'pub_date': pub_date, 'abstract': abstract,
                'pdf_url': pdf_url, 'categories': '', 'journal_name': None, 'score': 0.5,
            })
        return results

    def _reconstruct_abstract(inv_idx):
        if not inv_idx:
            return ''
        pw = {}
        for w, positions in inv_idx.items():
            for p in positions:
                pw[p] = w
        if not pw:
            return ''
        return ' '.join(pw.get(i, '') for i in range(max(pw.keys()) + 1))

    def _search_openalex_custom(keyword, start_d, end_d, max_results):
        params = {
            'search': keyword,
            'filter': f'from_publication_date:{start_d.isoformat()},to_publication_date:{end_d.isoformat()}',
            'sort': 'relevance_score:desc',
            'per_page': max_results,
            'mailto': os.environ.get("CONTACT_EMAIL", ""),
        }
        resp = requests.get("https://api.openalex.org/works", params=params, timeout=30)
        resp.raise_for_status()
        works = resp.json().get('results', [])
        results = []
        for w in works:
            doi = w.get('doi', '')
            if doi:
                doi = _re.sub(r'^https?://doi\.org/', '', doi)
            if not doi:
                continue
            title = (w.get('title') or w.get('display_name') or '').strip()
            abstract = _reconstruct_abstract(w.get('abstract_inverted_index'))
            authorships = w.get('authorships', [])
            authors = ', '.join(
                a.get('author', {}).get('display_name', '')
                for a in authorships if a.get('author', {}).get('display_name')
            )
            pub_date = w.get('publication_date', '')
            oa = w.get('open_access', {})
            pdf_url = oa.get('oa_url') or (w.get('primary_location') or {}).get('landing_page_url', '') or f"https://doi.org/{doi}"
            primary_location = w.get('primary_location') or {}
            source_info = primary_location.get('source') or {}
            journal_name = source_info.get('display_name', '') or None
            results.append({
                'arxiv_id': doi, 'source': 'journal', 'title': title,
                'authors': authors, 'pub_date': pub_date, 'abstract': abstract,
                'pdf_url': pdf_url, 'categories': '', 'journal_name': journal_name, 'score': 0.5,
            })
        return results

    if st.button("🔍 Search", use_container_width=True, type="primary"):
        if not cs_keyword.strip():
            st.warning("Enter a keyword to search.")
        else:
            all_results = []
            with st.spinner("Searching..."):
                if "arXiv" in cs_sources:
                    try:
                        all_results += _search_arxiv_custom(cs_keyword, cs_start, cs_end, cs_max)
                    except Exception as e:
                        st.warning(f"arXiv search failed: {e}")
                if "OpenAlex" in cs_sources:
                    try:
                        all_results += _search_openalex_custom(cs_keyword, cs_start, cs_end, cs_max)
                    except Exception as e:
                        st.warning(f"OpenAlex search failed: {e}")
            st.session_state['_custom_search_results'] = all_results
            st.session_state['_custom_search_selected'] = set()

    results = st.session_state.get('_custom_search_results', [])

    if results:
        st.markdown(f'<div class="section-title">{len(results)} results found</div>', unsafe_allow_html=True)

        existing_dois = set(load_papers()['arxiv_id']) if not load_papers().empty else set()

        selected = st.session_state.get('_custom_search_selected', set())

        for idx, r in enumerate(results):
            already_in_db = r['arxiv_id'] in existing_dois
            col_check, col_content = st.columns([0.06, 0.94])
            with col_check:
                if already_in_db:
                    st.markdown("✅")
                else:
                    checked = st.checkbox("", key=f"cs_sel_{idx}", value=(idx in selected))
                    if checked:
                        selected.add(idx)
                    else:
                        selected.discard(idx)
            with col_content:
                src_tag = 'arXiv' if r['source'] == 'arxiv' else (r.get('journal_name') or 'Journal')
                note = " *(already in database)*" if already_in_db else ""
                st.markdown(f"**{r['title']}**{note}")
                st.caption(f"{src_tag} · {r['pub_date']} · {r['authors'][:100]}")
                with st.expander("Abstract"):
                    st.write(r['abstract'] or "*No abstract available*")
            st.markdown("---")

        st.session_state['_custom_search_selected'] = selected

        if st.button(f"💾 Import {len(selected)} selected paper(s)", use_container_width=True, type="primary", disabled=(len(selected) == 0)):
            inserted, skipped = 0, 0
            for idx in selected:
                record = {k: v for k, v in results[idx].items()}
                try:
                    supabase.table('papers').insert(record).execute()
                    inserted += 1
                except Exception as e:
                    if 'duplicate key' in str(e) or '23505' in str(e):
                        skipped += 1
                    else:
                        st.error(f"Error importing '{record['title'][:50]}': {e}")
            st.cache_data.clear()
            st.success(f"Imported {inserted} paper(s). {skipped} were already in the database.")
            st.session_state['_custom_search_results'] = []
            st.session_state['_custom_search_selected'] = set()
            st.rerun()

# ══════════════════════════════════════════════════════════════
# PAGE: DIGITIZER
# ══════════════════════════════════════════════════════════════
elif page == "📈  Digitizer":
    render_digitizer_content(
        st.session_state.get('_digitizer_paper_id'),
        st.session_state.get('_digitizer_paper_title'),
        in_dialog=False
    )

# ══════════════════════════════════════════════════════════════
# PAGE: AI ASSISTANT
# ══════════════════════════════════════════════════════════════
elif page == "💬  AI Assistant":
    import json as _json_agent

    st.markdown("# 💬 AI Research Assistant")
    st.markdown('<div class="info-box">Agentic assistant with 5 tools: predicts ZT with the trained model, searches 61 full-text papers, looks up real dataset measurements, fetches abstracts for uncovered DOIs, and reads digitizer-extracted data (conductivity, Seebeck, etc.) with unit-aware interpolation. The agent decides which tools to use for each question.</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("---")
        st.markdown("**OpenRouter API Key**")
        groq_api_key = st.text_input("API Key", value=_ENV_OPENROUTER_KEY,
                                      type="password", label_visibility="collapsed")

    # ── Cached heavy resources (loaded once per session) ──────────
    @st.cache_resource
    def get_embed_model():
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('all-MiniLM-L6-v2')

    @st.cache_resource
    def get_chroma_collection():
        import chromadb
        client = chromadb.PersistentClient(path=PROJECT_DIR + "chroma_db")
        try:
            return client.get_collection("thermoelectric_papers")
        except Exception:
            return None

    embed_model = get_embed_model()
    chroma_collection = get_chroma_collection()

    # ── Tool implementations (reuse df, model, feat_cols, predict_zt, supabase from above) ──
    def agent_tool_predict_zt(composition: str, temperature: float) -> dict:
        try:
            pred = predict_zt(composition, temperature)
            return {"composition": composition, "temperature_K": temperature, "predicted_ZT": round(pred, 4)}
        except Exception as e:
            return {"error": f"Could not predict ZT for '{composition}': {e}"}

    def agent_tool_retrieve_literature(query: str, n_results: int = 3) -> dict:
        if chroma_collection is None:
            return {"error": "Literature corpus (ChromaDB) not available."}
        try:
            query_embedding = embed_model.encode([query]).tolist()
            results = chroma_collection.query(query_embeddings=query_embedding, n_results=n_results)
            chunks = []
            for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
                chunks.append({
                    "paper_title": meta['title'], "doi": meta['doi'],
                    "relevance_distance": round(dist, 3), "text": doc[:800],
                })
            return {"query": query, "results": chunks}
        except Exception as e:
            return {"error": f"Literature search failed: {e}"}

    def agent_tool_lookup_composition(composition: str) -> dict:
        if df is None:
            return {"error": "Dataset not loaded."}
        matches = df[df['Composition'].str.lower() == composition.lower()]
        if matches.empty:
            matches = df[df['Composition'].str.contains(composition, case=False, na=False)]
        if matches.empty:
            return {"error": f"No data found for composition '{composition}'."}
        measurements = matches[['Composition', 'Temperature (K)', 'ZT', 'DOI']].to_dict('records')
        return {"composition_queried": composition, "n_measurements": len(measurements),
                "measurements": measurements[:20]}

    def agent_tool_fetch_abstract(doi: str) -> dict:
        try:
            doi_clean = doi.replace('https://doi.org/', '').strip()
            url = f"https://api.openalex.org/works/https://doi.org/{doi_clean}"
            resp = requests.get(url, params={'mailto': os.environ.get("CONTACT_EMAIL", "")}, timeout=15)
            resp.raise_for_status()
            work = resp.json()
            inv_idx = work.get('abstract_inverted_index')
            abstract = ''
            if inv_idx:
                pw = {}
                for w, positions in inv_idx.items():
                    for p in positions:
                        pw[p] = w
                abstract = ' '.join(pw.get(i, '') for i in range(max(pw.keys()) + 1)) if pw else ''
            return {"doi": doi_clean, "title": work.get('title', work.get('display_name', '')),
                    "abstract": abstract[:1000], "publication_date": work.get('publication_date', '')}
        except Exception as e:
            return {"error": f"Could not fetch abstract for DOI '{doi}': {e}"}

    def agent_tool_lookup_digitized_data(composition: str, property_type: str = None,
                                           x_value: float = None, extra_filter_name: str = None,
                                           extra_filter_value: str = None) -> dict:
        if supabase is None:
            return {"error": "Supabase not configured."}
        try:
            query = supabase.table('extracted_data').select('*, papers(title, arxiv_id)')
            query = query.ilike('composition', f'%{composition}%')
            if property_type:
                query = query.ilike('property_type', f'%{property_type}%')
            result = query.execute()
            rows = result.data
            if not rows:
                return {"error": f"No digitized data found for composition '{composition}'"
                                 + (f" with property '{property_type}'" if property_type else "")}

            if extra_filter_name and extra_filter_value:
                matching_ids = set()
                for row in rows:
                    fr = supabase.table('extracted_data_fields').select('field_name, field_value').eq('extracted_data_id', row['id']).execute()
                    for f in fr.data:
                        if extra_filter_name.lower() in f['field_name'].lower() and extra_filter_value.lower() in (f['field_value'] or '').lower():
                            matching_ids.add(row['id'])
                rows = [r for r in rows if r['id'] in matching_ids]
                if not rows:
                    return {"error": f"No digitized data found matching {extra_filter_name}={extra_filter_value}"}

            entries = []
            for row in rows:
                points = sorted(_json_agent.loads(row['points_json']), key=lambda p: p['x'])
                entry = {
                    "figure_label": row.get('figure_label'), "composition": row.get('composition'),
                    "property_type": row.get('property_type'), "x_label": row.get('x_label'),
                    "y_label": row.get('y_label'), "x_unit": row.get('x_unit'), "y_unit": row.get('y_unit'),
                    "series_name": row.get('series_name'), "comments": row.get('comments'),
                    "source_paper": (row.get('papers') or {}).get('title'),
                    "source_doi": (row.get('papers') or {}).get('arxiv_id'),
                    "n_points": len(points),
                    "x_range": [points[0]['x'], points[-1]['x']] if points else None,
                }
                if x_value is not None and points:
                    if x_value <= points[0]['x']:
                        entry["interpolated_y_at_x"] = points[0]['y']
                        entry["interpolation_note"] = "x_value at/below range minimum — nearest point used"
                    elif x_value >= points[-1]['x']:
                        entry["interpolated_y_at_x"] = points[-1]['y']
                        entry["interpolation_note"] = "x_value at/above range maximum — nearest point used"
                    else:
                        for i in range(len(points) - 1):
                            x0, y0 = points[i]['x'], points[i]['y']
                            x1, y1 = points[i + 1]['x'], points[i + 1]['y']
                            if x0 <= x_value <= x1:
                                t = (x_value - x0) / (x1 - x0) if x1 != x0 else 0
                                entry["interpolated_y_at_x"] = round(y0 + t * (y1 - y0), 4)
                                entry["interpolation_note"] = f"linearly interpolated between ({x0}, {y0}) and ({x1}, {y1})"
                                break
                entries.append(entry)
            return {"composition_queried": composition, "n_datasets_found": len(entries), "datasets": entries}
        except Exception as e:
            return {"error": f"Digitized data lookup failed: {e}"}

    AGENT_TOOL_FUNCTIONS = {
        "predict_zt": agent_tool_predict_zt,
        "retrieve_literature": agent_tool_retrieve_literature,
        "lookup_composition": agent_tool_lookup_composition,
        "fetch_abstract": agent_tool_fetch_abstract,
        "lookup_digitized_data": agent_tool_lookup_digitized_data,
    }

    AGENT_TOOL_SCHEMAS = [
        {"type": "function", "function": {
            "name": "predict_zt",
            "description": "Predict ZT for a composition at a specific temperature using the trained Gradient Boosting model.",
            "parameters": {"type": "object", "properties": {
                "composition": {"type": "string", "description": "Chemical formula, e.g. 'Bi2Te3'"},
                "temperature": {"type": "number", "description": "Temperature in Kelvin"}},
                "required": ["composition", "temperature"]}}},
        {"type": "function", "function": {
            "name": "retrieve_literature",
            "description": "Semantic search over 61 full-text thermoelectric papers. Use for mechanisms, strategies, or literature-grounded explanations.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string", "description": "Natural-language search query"},
                "n_results": {"type": "integer", "description": "How many passages (default 3)"}},
                "required": ["query"]}}},
        {"type": "function", "function": {
            "name": "lookup_composition",
            "description": "Look up real experimental ZT measurements for a composition from the project's dataset.",
            "parameters": {"type": "object", "properties": {
                "composition": {"type": "string", "description": "Chemical formula, e.g. 'Bi2Te3'"}},
                "required": ["composition"]}}},
        {"type": "function", "function": {
            "name": "fetch_abstract",
            "description": "Fetch title/abstract from OpenAlex by DOI, for papers not in the local 61-paper corpus.",
            "parameters": {"type": "object", "properties": {
                "doi": {"type": "string", "description": "DOI to fetch"}},
                "required": ["doi"]}}},
        {"type": "function", "function": {
            "name": "lookup_digitized_data",
            "description": "Search digitizer-extracted data (conductivity, Seebeck coefficient, thermal conductivity, etc. from paper figures) by material and property, with optional interpolation at a specific x-value (e.g. temperature) and optional filtering by custom fields (e.g. Quenching Method).",
            "parameters": {"type": "object", "properties": {
                "composition": {"type": "string", "description": "Material to search for"},
                "property_type": {"type": "string", "description": "Optional property filter, e.g. 'Electrical conductivity'"},
                "x_value": {"type": "number", "description": "Optional x-value to interpolate at"},
                "extra_filter_name": {"type": "string", "description": "Optional custom field name to filter by"},
                "extra_filter_value": {"type": "string", "description": "Optional value to match for extra_filter_name"}},
                "required": ["composition"]}}},
    ]

    AGENT_SYSTEM_PROMPT = """You are an expert thermoelectric materials research assistant with access to tools:
- predict_zt: run the platform's trained ML model for a composition + temperature
- retrieve_literature: semantic search over 61 full-text thermoelectric papers
- lookup_composition: check the project's dataset for real experimental ZT measurements
- fetch_abstract: fetch a paper's abstract from OpenAlex by DOI, if not in the local corpus
- lookup_digitized_data: search manually-extracted data from paper figures (conductivity,
  Seebeck coefficient, and other properties beyond ZT), with interpolation at a specific
  x-value, and optional filtering by custom fields like processing history

Use tools when they would genuinely help — not every question needs one. A definitional
question needs no tools; a ZT performance question should use lookup_composition and/or
retrieve_literature; a question about conductivity/Seebeck/other non-ZT properties at a
specific condition should use lookup_digitized_data; "what would ZT be for X at Y
temperature" should use predict_zt.

Cite DOIs so the user can verify. When lookup_digitized_data returns an interpolated
value, say so explicitly and name the source paper. All digitized data is standardized
to canonical units (Kelvin, S/cm, etc.) — always state units alongside values using
x_unit/y_unit from the tool result. Be precise and scientific. If tools return nothing
useful, say so honestly rather than making things up.

CRITICAL: if a tool call returns an error and you need to retry it (e.g. to fix a
formatting mistake like element symbol capitalization), you MUST preserve every other
parameter EXACTLY as it was in the failed call and as originally given by the user —
fix ONLY the specific thing that caused the error. Never silently change a temperature,
composition, or any other value during a retry. Before finalizing your answer, double-
check that any numeric values you report (e.g. temperature) match exactly what the user
asked for."""

    FREE_MODEL = "openrouter/free"

    def run_agent(api_key, user_message, history, max_iterations=5):
        """Returns (final_answer, list_of_tool_calls_made) for transparency in the UI."""
        messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}] + history + [{"role": "user", "content": user_message}]
        tool_trace = []

        for _ in range(max_iterations):
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": FREE_MODEL, "messages": messages, "tools": AGENT_TOOL_SCHEMAS,
                      "temperature": 0, "max_tokens": 1500},
                timeout=60,
            )
            if resp.status_code == 429:
                return ("Rate limit hit on the free tier (20/min, 50-1000/day). Wait a moment and retry.", tool_trace)
            resp.raise_for_status()
            message = resp.json()["choices"][0]["message"]
            tool_calls = message.get("tool_calls")

            if not tool_calls:
                return (message.get("content", ""), tool_trace)

            messages.append(message)
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                try:
                    fn_args = _json_agent.loads(tc["function"]["arguments"])
                except _json_agent.JSONDecodeError:
                    fn_args = {}
                fn = AGENT_TOOL_FUNCTIONS.get(fn_name)
                result = fn(**fn_args) if fn else {"error": f"Unknown tool: {fn_name}"}
                tool_trace.append({"tool": fn_name, "args": fn_args, "result": result})
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": _json_agent.dumps(result)})

        return ("Reached max iterations without a final answer.", tool_trace)

    SUGGESTED = [
        "What is the predicted ZT for Bi2Te3 at 400 K?",
        "What does the literature say about reducing lattice thermal conductivity in half-Heuslers?",
        "What are the actual measured ZT values for Bi2Te3?",
        "What's the electrical conductivity for Bi2Te3 based on digitized figure data?",
        "How does band convergence enhance thermoelectric performance in PbTe?",
        "What makes half-Heusler alloys promising for thermoelectrics?",
    ]

    if "ai_display" not in st.session_state: st.session_state["ai_display"] = []
    if "ai_history" not in st.session_state: st.session_state["ai_history"] = []
    if "ai_traces" not in st.session_state: st.session_state["ai_traces"] = {}

    if not st.session_state["ai_display"]:
        st.markdown("#### 💡 Suggested questions")
        cols = st.columns(2)
        for i, q in enumerate(SUGGESTED):
            if cols[i % 2].button(q, key=f"sug_{i}", use_container_width=True):
                st.session_state["_ai_prefill"] = q
                st.rerun()

    chat_container = st.container()
    with chat_container:
        for i, (role, text) in enumerate(st.session_state["ai_display"]):
            with st.chat_message(role):
                st.markdown(text)
                if role == "assistant" and i in st.session_state["ai_traces"]:
                    trace = st.session_state["ai_traces"][i]
                    if trace:
                        with st.expander(f"🔧 Agent used {len(trace)} tool call(s)"):
                            for t in trace:
                                st.markdown(f"**{t['tool']}**({t['args']})")
                                st.code(_json_agent.dumps(t['result'], indent=2)[:800], language="json")

    prefill = st.session_state.pop("_ai_prefill", "")
    user_input = st.chat_input("Ask about ZT predictions, literature, real measurements, or digitized data...")
    if prefill and not user_input:
        user_input = prefill

    if user_input:
        if not groq_api_key:
            st.warning("Enter your OpenRouter API key in the sidebar.", icon="🔑")
        else:
            st.session_state["ai_display"].append(("user", user_input))
            with st.spinner("Agent is thinking (may call tools)..."):
                try:
                    answer, trace = run_agent(groq_api_key, user_input, st.session_state["ai_history"])
                    st.session_state["ai_history"].append({"role": "user", "content": user_input})
                    st.session_state["ai_history"].append({"role": "assistant", "content": answer})
                    idx = len(st.session_state["ai_display"])
                    st.session_state["ai_display"].append(("assistant", answer))
                    st.session_state["ai_traces"][idx] = trace
                except Exception as e:
                    err = str(e)
                    if "401" in err: st.error("Invalid API key.")
                    elif "429" in err: st.error("Rate limit hit. Wait a moment and retry.")
                    else: st.error(f"Error: {err}")
            st.rerun()

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state["ai_display"] = []
            st.session_state["ai_history"] = []
            st.session_state["ai_traces"] = {}
            st.rerun()
    with c2:
        if st.session_state["ai_display"]:
            lines = [f"[{'You' if r == 'user' else 'AI'}]\n{t}\n"
                     for r, t in st.session_state["ai_display"]]
            st.download_button("📥 Export transcript", data="\n".join(lines),
                               file_name="ai_chat_transcript.txt", mime="text/plain",
                               use_container_width=True)
        else:
            st.button("📥 Export transcript", disabled=True, use_container_width=True)
