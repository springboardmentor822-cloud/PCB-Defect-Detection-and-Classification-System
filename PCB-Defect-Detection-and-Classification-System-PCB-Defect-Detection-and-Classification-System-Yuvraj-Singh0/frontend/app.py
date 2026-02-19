import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import sys
import time
import pandas as pd
# from config.config_loader import TEMP_UPLOAD_DIR
# -------------------------------------------------
# LOCAL FRONTEND PATHS (NO CONFIG DEPENDENCY)
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
TEMP_UPLOAD_DIR = BASE_DIR / "temp_uploads"
TEMP_UPLOAD_DIR.mkdir(exist_ok=True)

# -------------------------------------------------
# PATH SETUP
# -------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.model_router import run_model_router
from src.inference_pipeline import (
    list_templates,
    generate_defect_heatmap
)


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Industrial AOI – PCB Defect Detection",
    page_icon="🧠",
    layout="wide"
)
# -------------------------------------------------
# GLOBAL UI STYLE FIX (REMOVE FORM BORDERS)
# -------------------------------------------------
st.markdown(
    """
    <style>
    /* Remove Streamlit form border */
    div[data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
        background: transparent !important;
    }

    /* Remove any surrounding block borders */
    div[data-testid="stVerticalBlock"] > div {
        border: none !important;
        box-shadow: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
for key in [
    "result", "image_path",
    "template_mode", "manual_template_index",
    "model_key"
]:
    if key not in st.session_state:
        st.session_state[key] = None

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.title("🧠 Industrial AOI – PCB Defect Detection")
st.caption(
    "Reference-based Automated Optical Inspection using "
    "template alignment, image differencing, and deep learning."
)


# -------------------------------------------------
# SIDEBAR – INSPECTION CONFIG (BEFORE UPLOAD)
# -------------------------------------------------
st.sidebar.header("⚙️ Inspection Configuration")

# -------- MODEL SELECTION --------
model_choice = st.sidebar.radio(
    "Model Selection",
    ["EfficientNet-B0", "Custom CNN"],
    index=0
)

# -------- TEMPLATE SELECTION MODE --------
template_mode = st.sidebar.radio(
    "Template Selection Mode",
    ["Automatic", "Manual"],
    index=0
)

selected_template_path = None

# -------- MANUAL TEMPLATE PREVIEW --------
if template_mode == "Manual":
    st.sidebar.markdown("### 🧩 Select Golden Template")

    templates = list_templates()
    template_names = [p.name for p in templates]

    selected_name = st.sidebar.selectbox(
        "Available Templates",
        template_names
    )

    selected_template_path = next(
        p for p in templates if p.name == selected_name
    )

    tpl_img = cv2.imread(str(selected_template_path))
    tpl_rgb = cv2.cvtColor(tpl_img, cv2.COLOR_BGR2RGB)

    st.sidebar.image(
        tpl_rgb,
        caption=f"Preview: {selected_name}",
        width="stretch"
    )

st.sidebar.markdown("---")

# -------------------------------------------------
# INSPECTION CONFIGURATION (MAIN AREA)
# -------------------------------------------------
st.markdown("## ⚙️ Inspection Configuration")

cfg_col1, cfg_col2 = st.columns([1, 1])

with cfg_col1:
    st.markdown("### 🧠 Model Selection")
    model_choice = st.radio(
        "Select Model",
        ["EfficientNet-B0", "Custom CNN"],
        index=0,
        horizontal=True
    )

with cfg_col2:
    st.markdown("### 🧩 Template Selection Mode")
    template_mode = st.radio(
        "Template Mode",
        ["Automatic", "Manual"],
        index=0,
        horizontal=True
    )
# -------------------------------------------------
# MANUAL TEMPLATE SELECTION — GRID PREVIEW (CORRECT)
# -------------------------------------------------
selected_template_path = None

if template_mode == "Manual":
    st.markdown("### 🖼️ Select Golden Template")

    templates = list_templates()
    template_names = [p.name for p in templates]

    # Initialize session state
    if "selected_template" not in st.session_state:
        st.session_state.selected_template = template_names[0]

    # 👉 Single radio (controls selection)
    selected_name = st.radio(
        "Golden Template",
        template_names,
        index=template_names.index(st.session_state.selected_template),
        horizontal=True
    )

    st.session_state.selected_template = selected_name

    # 👉 Preview grid (visual only)
    cols = st.columns(4)

    for idx, tpl_path in enumerate(templates):
        with cols[idx % 4]:
            tpl_img = cv2.imread(str(tpl_path))
            tpl_rgb = cv2.cvtColor(tpl_img, cv2.COLOR_BGR2RGB)

            border = "🟢 SELECTED" if tpl_path.name == selected_name else ""

            st.image(
                tpl_rgb,
                caption=f"{tpl_path.name} {border}",
                width=180
            )

    selected_template_path = next(
        p for p in templates if p.name == selected_name
    )

    st.success(f"✅ Selected Template: {selected_template_path.name}")
# -------------------------------------------------
# FILE UPLOAD (PREVIEW FIRST)
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload PCB Test Image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------------------------
# RESET OLD RESULTS ON NEW IMAGE
# -------------------------------------------------
if uploaded_file is not None:
    if (
        st.session_state.image_path is not None and
        uploaded_file.name != Path(st.session_state.image_path).name
    ):
        st.session_state.result = None

# -------------------------------------------------
# IMAGE PREVIEW
# -------------------------------------------------
if uploaded_file is not None:
    upload_dir = TEMP_UPLOAD_DIR

    image_path = upload_dir / uploaded_file.name
    with open(image_path, "wb") as f:
        f.write(uploaded_file.read())

    st.session_state.image_path = image_path

    original = cv2.imread(str(image_path))
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Input PCB Image")
        st.image(original_rgb, width="stretch")

    manual_index = None
    if template_mode == "Manual":
        templates = list_templates()
        manual_index = templates.index(selected_template_path)

    # -------------------------------------------------
    # RUN AOI (FORM — AFTER PREVIEW)
    # -------------------------------------------------
    with st.form("run_aoi_form", clear_on_submit=False):
        submit_run = st.form_submit_button("🚀 Run AOI Inspection")

    if submit_run:
        with st.spinner("Running industrial AOI pipeline..."):

            model_key = (
                "efficientnet"
                if model_choice == "EfficientNet-B0"
                else "custom"
            )

            start = time.perf_counter()

            result = run_model_router(
                image_path=image_path,
                model_key=model_key,
                template_mode="manual" if template_mode == "Manual" else "auto",
                manual_template_index=manual_index
            )

            result["frontend_time"] = time.perf_counter() - start
            st.session_state.result = result

# -------------------------------------------------
# RESULTS
# -------------------------------------------------
if st.session_state.result:
    result = st.session_state.result

    annotated_rgb = cv2.cvtColor(result["annotated"], cv2.COLOR_BGR2RGB)

    with col2:
        st.subheader("🛠️ AOI Output")
        st.image(annotated_rgb, width="stretch")
    # -------------------------------------------------
    # 🔥 DEFECT HEATMAP + BACKEND TIME (CENTERED)
    # -------------------------------------------------
    st.markdown("### 🔥 Defect Density Heatmap")

    heat_col1, heat_col2 = st.columns([4, 1])

    with heat_col1:
        if result["detections"]:
            heatmap = generate_defect_heatmap(
                cv2.imread(str(st.session_state.image_path)),
                result["detections"]
            )

            st.image(
                cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB),
                width=700,
                caption="Blue = Safe | Yellow/Red = Defect Density"
            )
        else:
            st.info("No defects detected — heatmap not generated.")

    with heat_col2:
        backend_time = result["timing"].get("total", None)

        if backend_time is not None:
            st.markdown(
                """
                <div style="
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100%;
                    min-height: 300px;
                    text-align: center;
                ">
                    <div>
                        <h3>⏱ Backend Time</h3>
                        <h2>{:.3f} s</h2>
                    </div>
                </div>
                """.format(backend_time),
                unsafe_allow_html=True
            )

    # -------------------------------------------------
    # 📥 DOWNLOAD SECTION
    # -------------------------------------------------
    st.markdown("## 📥 Download Results")

    dl_col1, dl_col2 = st.columns([1, 1])

    # ---------- DOWNLOAD ANNOTATED IMAGE ----------
    with dl_col1:
        annotated_bgr = result["annotated"]
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        _, img_buf = cv2.imencode(".png", annotated_bgr)

        st.download_button(
            label="📷 Download Annotated Image",
            data=img_buf.tobytes(),
            file_name="annotated_pcb_result.png",
            mime="image/png"
        )

    # ---------- DOWNLOAD CSV REPORT ----------
    with dl_col2:
        if result["detections"]:
            csv_rows = []
            for i, d in enumerate(result["detections"], 1):
                x, y, w, h = d["bbox"]
                csv_rows.append({
                    "ID": i,
                    "Defect": d["label"],
                    "Confidence": round(d["confidence"], 4),
                    "X": x,
                    "Y": y,
                    "Width": w,
                    "Height": h
                })

            df_csv = pd.DataFrame(csv_rows)
        else:
            # Empty but valid CSV
            df_csv = pd.DataFrame(
                columns=["ID", "Defect", "Confidence", "X", "Y", "Width", "Height"]
            )

        st.download_button(
            label="📊 Download Defect Report (CSV)",
            data=df_csv.to_csv(index=False),
            file_name="pcb_defect_report.csv",
            mime="text/csv"
        )


    # -------------------------------------------------
    # AOI STATUS
    # -------------------------------------------------
    st.markdown("## 🧪 Inspection Result")

    if len(result["detections"]) == 0:
        st.success("✅ PASS — No defects detected")
    else:
        st.error("❌ FAIL — Defects detected")

    # -------------------------------------------------
    # DEFECT COUNTS
    # -------------------------------------------------
    cols = st.columns(3)
    for i, (cls, cnt) in enumerate(result["counts"].items()):
        cols[i % 3].metric(cls, cnt)

    # -------------------------------------------------
    # DEFECT LOCATION TABLE
    # -------------------------------------------------
    st.markdown("## 📍 Defect Location Table")

    if result["detections"]:
        rows = []
        for i, d in enumerate(result["detections"], 1):
            x, y, w, h = d["bbox"]
            rows.append({
                "ID": i,
                "Defect": d["label"],
                "Confidence": round(d["confidence"], 3),
                "X": x,
                "Y": y,
                "Width": w,
                "Height": h
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No defect regions detected.")

    # =================================================
    # 🧩 AOI PIPELINE BREAKDOWN (OLD FEATURE RESTORED)
    # =================================================
    st.markdown("## 🧩 AOI Pipeline Breakdown")

    tabs = st.tabs([
        "Golden Template",
        "Auto Mask",
        "Stored Mask",
        "Final Mask",
        "Processing Time"
    ])

    with tabs[0]:
        st.image(
            cv2.cvtColor(result["template"], cv2.COLOR_BGR2RGB),
            caption="Selected Golden Template",
            width=700
        )
        st.markdown(
            "**Purpose:** Reference defect-free PCB used for alignment"
        )

    with tabs[1]:
        st.image(
            result["auto_mask"],
            caption="Auto Difference Mask",
            width=700
        )
        st.markdown(
            "**Method:** Image subtraction + Otsu threshold"
        )

    with tabs[2]:
        if result["stored_mask"] is not None:
            st.image(
                result["stored_mask"],
                caption="Stored Dataset Mask",
                width=700
            )
        else:
            st.info("No stored mask available")

    with tabs[3]:
        st.image(
            result["final_mask"],
            caption="Validated Final Defect Mask",
            width=700
        )
        st.markdown(
            "**Logic:** IoU-based validation + border suppression"
        )

    with tabs[4]:
        for k, v in result["timing"].items():
            st.metric(
                k.replace("_", " ").title(),
                f"{v:.3f}s"
            )

        st.metric(
            "Frontend Render Time",
            f"{result['frontend_time']:.3f}s"
        )
    # -------------------------------------------------
    # TERMINAL-STYLE DETECTION LOG
    # -------------------------------------------------
    st.markdown("## 🧾 Detection Log")

    if result["detections"]:
        log = "\n".join(
            [
                f"{i}. {d['label']} | conf={d['confidence']:.2f} | bbox={d['bbox']}"
                for i, d in enumerate(result["detections"], 1)
            ]
        )
        st.code(log, language="text")
    else:
        st.code("No detections", language="text")

    # -------------------------------------------------
    # SYSTEM INFO
    # -------------------------------------------------
    h, w = result["annotated"].shape[:2]

    st.markdown("## 🖥️ System Info")
    st.code(
        f"""[SYSTEM INFO]
Device           : cpu
Model            : {model_choice}
Image Resolution : {w}x{h}
ROIs Processed   : {len(result['rois'])}
Detections       : {len(result['detections'])}

Inference time   : {result['timing']['total']:.2f}s
""",
        language="text"
    )

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("---")
st.caption(
    "Industrial AOI PCB Defect Detection System • "
    "Deep Learning + Computer Vision"
)


