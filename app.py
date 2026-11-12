
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import subprocess
import os
import re
from pathlib import Path
from datetime import datetime

# =========================================================
# 1. KONFIGURASI DASAR
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
METRICS_FILE = BASE_DIR / "metrics.csv"

st.set_page_config(
    page_title="Flower Federated Learning Dashboard",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# 2. CSS DESIGN
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background: #641cff;
    color: #111111;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

header[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu, footer {
    visibility: hidden;
}

/* HEADER */

.main-header {
    background: #f5f1ff;
    color: #111111;
    padding: 25px 30px;
    border: 3px solid #111111;
    box-shadow: 7px 7px 0px #111111;
    margin-bottom: 25px;
}

.header-label {
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 3px;
    color: #641cff;
}

.header-title {
    font-size: 42px;
    font-weight: 800;
    line-height: 1.1;
    margin-top: 10px;
}

.header-description {
    font-size: 15px;
    margin-top: 10px;
    color: #444444;
}

/* SECTION */

.section-title {
    color: #111111;
    font-size: 17px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
    border-bottom: 2px solid #111111;
    padding-bottom: 12px;
    margin-top: 25px;
    margin-bottom: 15px;
}

/* METRIC CARDS */

.metric-card {
    background: #f5f1ff;
    border: 3px solid #111111;
    box-shadow: 5px 5px 0px #111111;
    padding: 20px;
    min-height: 135px;
    color: #111111;
}

.metric-label {
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1.5px;
    color: #555555;
    text-transform: uppercase;
}

.metric-value {
    font-size: 34px;
    font-weight: 800;
    margin-top: 12px;
}

.metric-note {
    font-size: 12px;
    color: #666666;
    margin-top: 5px;
}

/* PANEL */

.panel {
    background: #f5f1ff;
    border: 3px solid #111111;
    box-shadow: 5px 5px 0px #111111;
    padding: 20px;
    color: #111111;
    margin-bottom: 20px;
}

/* STATUS */

.status-online {
    display: inline-block;
    background: #c8ff45;
    color: #111111;
    padding: 7px 13px;
    border: 2px solid #111111;
    font-weight: 800;
    font-size: 12px;
}

.status-offline {
    display: inline-block;
    background: #ff7777;
    color: #111111;
    padding: 7px 13px;
    border: 2px solid #111111;
    font-weight: 800;
    font-size: 12px;
}

/* BUTTON */

.stButton > button {
    background: #c8ff45 !important;
    color: #111111 !important;
    border: 3px solid #111111 !important;
    border-radius: 0px !important;
    font-weight: 800 !important;
    box-shadow: 4px 4px 0px #111111;
    min-height: 45px;
}

.stButton > button:hover {
    background: #ffffff !important;
    color: #111111 !important;
    border-color: #111111 !important;
}

/* INPUT */

.stNumberInput input {
    border: 2px solid #111111 !important;
    border-radius: 0px !important;
    background: #ffffff !important;
    color: #111111 !important;
}

.stDataFrame {
    border: 2px solid #111111;
}

.footer {
    text-align: center;
    color: #ffffff;
    padding: 30px 10px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. HEADER
# =========================================================

st.markdown("""
<div class="main-header">
    <div class="header-label">🌐 FEDERATED LEARNING SYSTEM</div>
    <div class="header-title">
        TRAIN TOGETHER.<br>
        LEARN SMARTER.
    </div>
    <div class="header-description">
        Flower Framework • Federated Learning Dashboard
        • FedAvg Algorithm
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# 4. FUNGSI MEMBACA METRIK
# =========================================================

def load_metrics():

    if not METRICS_FILE.exists():
        return pd.DataFrame(
            columns=["round", "accuracy", "loss"]
        )

    try:
        df = pd.read_csv(METRICS_FILE)

        required_columns = ["round", "accuracy", "loss"]

        if not all(col in df.columns for col in required_columns):
            return pd.DataFrame(
                columns=required_columns
            )

        df = df[required_columns].copy()

        for col in required_columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        return df.dropna()

    except Exception as e:
        st.warning(f"Gagal membaca metrics.csv: {e}")
        return pd.DataFrame(
            columns=["round", "accuracy", "loss"]
        )


metrics = load_metrics()


# =========================================================
# 5. SERVER OVERVIEW
# =========================================================

st.markdown(
    '<div class="section-title">📊 SERVER OVERVIEW</div>',
    unsafe_allow_html=True
)

if not metrics.empty:
    latest_accuracy = metrics["accuracy"].iloc[-1]
    latest_loss = metrics["loss"].iloc[-1]
    total_rounds = len(metrics)
else:
    latest_accuracy = 0
    latest_loss = 0
    total_rounds = 0

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🎯 Accuracy</div>
        <div class="metric-value">{latest_accuracy:.2f}%</div>
        <div class="metric-note">Latest model accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">📉 Loss</div>
        <div class="metric-value">{latest_loss:.4f}</div>
        <div class="metric-note">Latest training loss</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🔄 Rounds</div>
        <div class="metric-value">{total_rounds}</div>
        <div class="metric-note">Recorded rounds</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">⚙️ Framework</div>
        <div class="metric-value">Flower</div>
        <div class="metric-note">Federated Learning</div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# 6. FEDERATION TOPOLOGY
# =========================================================

st.markdown(
    '<div class="section-title">🌐 FEDERATION TOPOLOGY</div>',
    unsafe_allow_html=True
)

topology_html = """
<!DOCTYPE html>
<html>
<head>
<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    padding: 12px;
    background: #f5f1ff;
    font-family: Arial, sans-serif;
}

.topology {
    position: relative;
    width: 100%;
    min-height: 390px;
    background: #242321;
    border: 3px solid #111111;
    border-radius: 12px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 25px 15px;
}

.server {
    z-index: 2;
    background: #c8ff45;
    border: 3px solid white;
    box-shadow: 5px 5px 0px #641cff;
    padding: 15px 35px;
    text-align: center;
    color: #111111;
    border-radius: 10px;
    min-width: 220px;
}

.server-icon {
    font-size: 35px;
}

.server-title {
    font-weight: 900;
    font-size: 17px;
    margin-top: 5px;
}

.server-subtitle {
    font-size: 10px;
    font-weight: bold;
    margin-top: 5px;
}

.connection {
    width: 100%;
    height: 100px;
    margin-top: 0px;
    margin-bottom: -5px;
}

.clients {
    width: 100%;
    display: flex;
    justify-content: space-around;
    gap: 15px;
    z-index: 2;
}

.client {
    flex: 1;
    max-width: 220px;
    background: #ffffff;
    border: 3px solid #65cfff;
    box-shadow: 5px 5px 0px #641cff;
    border-radius: 10px;
    padding: 15px 8px;
    text-align: center;
    color: #111111;
}

.client-icon {
    font-size: 32px;
}

.client-title {
    font-weight: 900;
    font-size: 14px;
    margin-top: 5px;
}

.client-subtitle {
    font-size: 10px;
    margin-top: 5px;
    color: #555555;
}

.client-status {
    display: inline-block;
    background: #c8ff45;
    border: 1px solid #111111;
    padding: 4px 8px;
    font-size: 9px;
    font-weight: bold;
    margin-top: 10px;
}

.topology-footer {
    color: #ffffff;
    font-size: 11px;
    margin-top: 20px;
    letter-spacing: 1px;
    text-align: center;
}

@media(max-width: 600px) {
    .topology {
        min-height: 350px;
        padding: 20px 8px;
    }

    .server {
        min-width: 190px;
        padding: 12px;
    }

    .clients {
        gap: 5px;
    }

    .client {
        padding: 10px 3px;
    }

    .client-title {
        font-size: 11px;
    }

    .client-subtitle {
        font-size: 8px;
    }

    .client-icon {
        font-size: 24px;
    }

    .connection {
        height: 75px;
    }
}

</style>
</head>

<body>

<div class="topology">

    <div class="server">
        <div class="server-icon">🖥️</div>
        <div class="server-title">FLOWER SERVER</div>
        <div class="server-subtitle">
            CENTRAL AGGREGATION • FEDAVG
        </div>
    </div>

    <svg class="connection"
         viewBox="0 0 650 100"
         preserveAspectRatio="none">

        <path d="M325 0 V35 H108 V100"
              stroke="#65cfff"
              stroke-width="4"
              fill="none"/>

        <path d="M325 35 V100"
              stroke="#65cfff"
              stroke-width="4"
              fill="none"/>

        <path d="M325 35 H542 V100"
              stroke="#65cfff"
              stroke-width="4"
              fill="none"/>

        <circle cx="108" cy="100" r="5" fill="#c8ff45"/>
        <circle cx="325" cy="100" r="5" fill="#c8ff45"/>
        <circle cx="542" cy="100" r="5" fill="#c8ff45"/>

    </svg>

    <div class="clients">

        <div class="client">
            <div class="client-icon">💻</div>
            <div class="client-title">CLIENT 01</div>
            <div class="client-subtitle">Local Dataset A</div>
            <div class="client-status">● CLIENT NODE</div>
        </div>

        <div class="client">
            <div class="client-icon">🖥️</div>
            <div class="client-title">CLIENT 02</div>
            <div class="client-subtitle">Local Dataset B</div>
            <div class="client-status">● CLIENT NODE</div>
        </div>

        <div class="client">
            <div class="client-icon">💾</div>
            <div class="client-title">CLIENT 03</div>
            <div class="client-subtitle">Local Dataset C</div>
            <div class="client-status">● CLIENT NODE</div>
        </div>

    </div>

    <div class="topology-footer">
        FLOWER SERVER → MODEL AGGREGATION → GLOBAL MODEL
    </div>

</div>

</body>
</html>
"""

components.html(
    topology_html,
    height=435,
    scrolling=False
)

st.caption(
    "Ilustrasi arsitektur federated learning. "
    "Jumlah dan status client pada diagram belum terhubung "
    "secara otomatis dengan node Flower yang sedang berjalan."
)


# =========================================================
# 7. SIMULATION CONTROL
# =========================================================

st.markdown(
    '<div class="section-title">⚡ SIMULATION CONTROL</div>',
    unsafe_allow_html=True
)

control1, control2 = st.columns(2)

with control1:
    client_count = st.number_input(
        "Number of Clients (tampilan saja)",
        min_value=1,
        max_value=10,
        value=3,
        step=1
    )

with control2:
    round_count = st.number_input(
        "Number of Rounds (tampilan saja)",
        min_value=1,
        max_value=100,
        value=3,
        step=1
    )

st.info(
    "Catatan: input client dan round di atas belum mengubah "
    "konfigurasi Flower. Atur konfigurasi tersebut pada "
    "server_app.py atau konfigurasi proyek Flower."
)

# =========================================================
# 8. MENJALANKAN FLOWER
# =========================================================

st.markdown(
    '<div class="section-title">🚀 RUN FEDERATED SIMULATION</div>',
    unsafe_allow_html=True
)

if "simulation_output" not in st.session_state:
    st.session_state.simulation_output = ""

if "simulation_error" not in st.session_state:
    st.session_state.simulation_error = ""

if st.button("▶ START FLOWER SIMULATION", use_container_width=True):

    st.session_state.simulation_output = ""
    st.session_state.simulation_error = ""

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8:replace"
    env["NO_COLOR"] = "1"
    env["FORCE_COLOR"] = "0"

    try:
        with st.spinner("Flower simulation sedang dijalankan..."):

            result = subprocess.run(
                ["flwr", "run", ".", "--stream"],
                cwd=str(BASE_DIR),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                timeout=1800
            )

        ansi_escape = re.compile(
            r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])'
        )

        output = ansi_escape.sub("", result.stdout or "")
        error = ansi_escape.sub("", result.stderr or "")

        st.session_state.simulation_output = output
        st.session_state.simulation_error = error

        if result.returncode == 0:
            st.success("Flower simulation selesai dijalankan!")
        else:
            st.error(
                f"Simulation gagal. Exit code: {result.returncode}"
            )

    except subprocess.TimeoutExpired:
        st.error(
            "Simulation melebihi batas waktu 30 menit."
        )

    except FileNotFoundError:
        st.error(
            "Perintah 'flwr' tidak ditemukan. "
            "Pastikan Flower sudah terinstal pada environment "
            "Python yang digunakan Streamlit."
        )

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")


if st.session_state.simulation_output:
    st.markdown("### 📋 Simulation Output")
    st.code(
        st.session_state.simulation_output[-12000:],
        language="text"
    )

if st.session_state.simulation_error:
    st.markdown("### ⚠️ Error Log")
    st.code(
        st.session_state.simulation_error[-8000:],
        language="text"
    )


# =========================================================
# 9. TRAINING PERFORMANCE
# =========================================================

st.markdown(
    '<div class="section-title">📈 TRAINING PERFORMANCE</div>',
    unsafe_allow_html=True
)

metrics = load_metrics()

if not metrics.empty:

    chart1, chart2 = st.columns(2)

    with chart1:
        st.markdown("### 🎯 Accuracy Progress")

        accuracy_chart = metrics.set_index("round")[["accuracy"]]

        st.line_chart(
            accuracy_chart,
            y="accuracy",
            height=280
        )

    with chart2:
        st.markdown("### 📉 Loss Progress")

        loss_chart = metrics.set_index("round")[["loss"]]

        st.line_chart(
            loss_chart,
            y="loss",
            height=280
        )

    st.markdown("### 📋 Training Metrics Data")

    st.dataframe(
        metrics,
        use_container_width=True,
        hide_index=True
    )

else:

    st.warning(
        "Belum ada data training. Grafik akan muncul "
        "setelah metrics.csv diisi dengan hasil training Flower."
    )


# =========================================================
# 10. ACTIVITY LOG
# =========================================================

st.markdown(
    '<div class="section-title">📝 SYSTEM ACTIVITY</div>',
    unsafe_allow_html=True
)

st.markdown(f"""
<div class="panel">
    <b>Dashboard Status:</b> Active<br><br>
    <b>Framework:</b> Flower Federated Learning<br><br>
    <b>Aggregation Algorithm:</b> FedAvg<br><br>
    <b>Last Dashboard Refresh:</b> {datetime.now().strftime("%d-%m-%Y %H:%M:%S")}<br><br>
    <b>Metrics Source:</b> metrics.csv
</div>
""", unsafe_allow_html=True)


# =========================================================
# 11. FOOTER
# =========================================================

st.markdown("""
<div class="footer">
    🌐 FLOWER FEDERATED LEARNING DASHBOARD<br>
    POWERED BY STREAMLIT & FLOWER FRAMEWORK
</div>
""", unsafe_allow_html=True)
