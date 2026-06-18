import streamlit as st
import cv2
import numpy as np
import pandas as pd
from fpdf import FPDF

from image_processing import preprocess_image
from ocr_engine import extract_text_and_dimensions
from ai_chat import ask_ai

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="AI Building Plan Analyzer | KLE Tech",
    page_icon="🏗️",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS - ENHANCED UI/UX
# ---------------------------------------------------

st.markdown("""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

/* Global Styles */
* {
    font-family: 'Inter', sans-serif;
}

.main {
    background: linear-gradient(135deg, #F7F3FF 0%, #E8DFFF 100%);
    background-attachment: fixed;
}

/* Subtle Blueprint Watermark */
.main::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image: 
        linear-gradient(0deg, transparent 24%, rgba(139, 92, 246, 0.02) 25%, rgba(139, 92, 246, 0.02) 26%, transparent 27%, transparent 74%, rgba(139, 92, 246, 0.02) 75%, rgba(139, 92, 246, 0.02) 76%, transparent 77%, transparent),
        linear-gradient(90deg, transparent 24%, rgba(139, 92, 246, 0.02) 25%, rgba(139, 92, 246, 0.02) 26%, transparent 27%, transparent 74%, rgba(139, 92, 246, 0.02) 75%, rgba(139, 92, 246, 0.02) 76%, transparent 77%, transparent);
    background-size: 50px 50px;
    pointer-events: none;
    z-index: 0;
}

.block-container {
    position: relative;
    z-index: 1;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Adjust header for better alignment */
.university-header {
    background: linear-gradient(135deg, #0B2E6B 0%, #1a4d8f 100%);
    padding: 1.5rem 2rem;
    border-radius: 16px;
    color: white;
    box-shadow: 0 10px 40px rgba(11, 46, 107, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.1);
    height: 100%;
    display: flex;
    align-items: center;
}

.university-name {
    font-size: 1.5rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
    letter-spacing: 0.5px;
}

.department-name {
    font-size: 0.95rem;
    font-weight: 600;
    opacity: 0.95;
    margin-bottom: 1rem;
}

.project-title {
    font-size: 1.8rem;
    font-weight: 800;
    margin: 0.5rem 0 0.3rem 0;
    background: linear-gradient(135deg, #ffffff 0%, #E8DFFF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.project-subtitle {
    font-size: 0.85rem;
    opacity: 0.9;
    font-weight: 400;
    line-height: 1.4;
}

/* Compact Info Card (Right Side) */
.compact-info-card {
    background: white;
    padding: 1.5rem;
    border-radius: 16px;
    box-shadow: 0 4px 20px rgba(139, 92, 246, 0.15);
    border: 1px solid rgba(139, 92, 246, 0.1);
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.compact-section {
    margin-bottom: 0;
}

.compact-label {
    font-size: 0.75rem;
    color: #6B7280;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.4rem;
}

.compact-content {
    font-size: 1rem;
    color: #0B2E6B;
    font-weight: 700;
    margin-bottom: 0.3rem;
}

.team-list {
    margin-top: 0.5rem;
}

.team-list div {
    color: #374151;
    font-size: 0.85rem;
    margin: 0.25rem 0;
    font-weight: 500;
}

/* Section Headers - UPDATED: Near-black for better readability */
h1 {
    color: #0B2E6B;
    font-weight: 800;
}

h2 {
    color: #1F1F1F;
    font-weight: 800;
    margin-top: 2rem;
    padding-bottom: 0.5rem;
    border-bottom: 3px solid #8B5CF6;
    display: inline-block;
}

h3 {
    color: #1F1F1F;
    font-weight: 800;
}

/* Card Styles */
.content-card {
    background: white;
    padding: 2rem;
    border-radius: 16px;
    box-shadow: 0 4px 20px rgba(139, 92, 246, 0.12);
    margin-bottom: 1.5rem;
    border: 1px solid rgba(139, 92, 246, 0.08);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.content-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(139, 92, 246, 0.2);
}

/* Metrics */
.stMetric {
    background: white;
    padding: 1.5rem;
    border-radius: 14px;
    box-shadow: 0 3px 15px rgba(139, 92, 246, 0.12);
    border-left: 4px solid #8B5CF6;
    transition: transform 0.2s ease;
}

.stMetric:hover {
    transform: scale(1.03);
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #8B5CF6 0%, #6D28D9 100%);
    color: white;
    border-radius: 12px;
    border: none;
    padding: 0.75rem 2rem;
    font-weight: 700;
    font-size: 1rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3);
    letter-spacing: 0.3px;
}

.stButton>button:hover {
    background: linear-gradient(135deg, #7C3AED 0%, #5B21B6 100%);
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(139, 92, 246, 0.4);
}

/* Text Areas */
.stTextArea textarea {
    background-color: #FAFAFA;
    color: #1F2937;
    border-radius: 12px;
    border: 2px solid #E5E7EB;
    font-family: 'Courier New', monospace;
    font-size: 0.9rem;
}

.stTextArea textarea:focus {
    border-color: #8B5CF6;
    box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.1);
}

/* Text Input */
.stTextInput input {
    border-radius: 10px;
    border: 2px solid #E5E7EB;
    padding: 0.75rem;
    font-size: 1rem;
}

.stTextInput input:focus {
    border-color: #8B5CF6;
    box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.1);
}

/* Images */
div[data-testid="stImage"] {
    background: white;
    padding: 1rem;
    border-radius: 14px;
    box-shadow: 0 3px 12px rgba(139, 92, 246, 0.1);
    transition: transform 0.3s ease;
}

div[data-testid="stImage"]:hover {
    transform: scale(1.02);
    box-shadow: 0 5px 20px rgba(139, 92, 246, 0.2);
}

/* File Uploader */
.uploadedFile {
    background: white;
    border-radius: 12px;
    border: 2px dashed #8B5CF6;
    padding: 1rem;
}

/* Info/Warning/Success Boxes */
.stAlert {
    border-radius: 12px;
    border-left: 5px solid;
    padding: 1rem 1.5rem;
    margin: 1rem 0;
}

div[data-baseweb="notification"] {
    border-radius: 12px;
    box-shadow: 0 3px 12px rgba(139, 92, 246, 0.1);
}

/* Tables - UPDATED: White text headers for better readability */
table {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 3px 15px rgba(139, 92, 246, 0.08);
}

thead tr {
    background: linear-gradient(135deg, #0B2E6B 0%, #1a4d8f 100%);
    color: #FFFFFF;
}

thead tr th {
    color: #FFFFFF !important;
    font-weight: bold !important;
    text-align: left !important;
    padding: 12px 15px !important;
    border: none !important;
}

tbody tr:nth-child(even) {
    background-color: #F9FAFB;
}

tbody tr:hover {
    background-color: #F3F4F6;
}

tbody tr td {
    padding: 10px 15px !important;
    color: #1F2937 !important;
}

/* Divider */
hr {
    border: none;
    height: 1px;
    background: #D0C4E6;
    margin: 2rem 0;
    opacity: 0.8;
}

/* Footer */
.footer {
    background: linear-gradient(135deg, #0B2E6B 0%, #1a4d8f 100%);
    padding: 2rem;
    border-radius: 16px;
    text-align: center;
    color: white;
    margin-top: 3rem;
    box-shadow: 0 -5px 20px rgba(11, 46, 107, 0.2);
}

.footer-title {
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.footer-text {
    font-size: 0.9rem;
    opacity: 0.9;
    line-height: 1.6;
}

/* Download Button Special Styling */
.stDownloadButton>button {
    background: linear-gradient(135deg, #10B981 0%, #059669 100%);
    color: white;
    border-radius: 12px;
    border: none;
    padding: 0.75rem 2rem;
    font-weight: 700;
    font-size: 1rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    width: 100%;
}

.stDownloadButton>button:hover {
    background: linear-gradient(135deg, #059669 0%, #047857 100%);
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .university-header {
        padding: 1rem;
    }
    
    .project-title {
        font-size: 1.3rem;
    }
    
    .university-name {
        font-size: 1.2rem;
    }
    
    .compact-info-card {
        margin-top: 1rem;
    }
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# UNIVERSITY HEADER WITH SIDE PROJECT INFO
# ---------------------------------------------------

import base64
from pathlib import Path

def get_image_base64(image_path):
    """Convert image to base64 string"""
    try:
        if not Path(image_path).exists():
            return None
        
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except Exception as e:
        return None

# Load logo
kle_logo_base64 = None
paths_to_try = [
    "kle_logo.png",
    "./kle_logo.png",
    str(Path(__file__).parent / "kle_logo.png"),
]

for path in paths_to_try:
    kle_logo_base64 = get_image_base64(path)
    if kle_logo_base64:
        break

# Create header
header_col1, header_col2 = st.columns([2.5, 1])

with header_col1:
    if kle_logo_base64:
        st.markdown(f"""
        <div class="university-header">
            <div style="display: flex; align-items: center; gap: 2rem;">
                <img src="data:image/png;base64,{kle_logo_base64}" 
                     style="width: 90px; height: 90px; object-fit: contain; background: white; padding: 8px; border-radius: 12px; box-shadow: 0 5px 15px rgba(0,0,0,0.2); flex-shrink: 0;"
                     alt="KLE Logo">
                <div style="text-align: left; flex: 1;">
                    <div class="university-name">KLE TECHNOLOGICAL UNIVERSITY</div>
                    <div class="department-name">School of Civil Engineering</div>
                    <div class="project-title">AI BUILDING PLAN ANALYZER</div>
                    <div class="project-subtitle">AI-Powered Floor Plan Intelligence & Engineering Assessment System</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="university-header">
            <div style="display: flex; align-items: center; gap: 2rem;">
                <div style="display: flex; width: 80px; height: 80px; background: white; border-radius: 50%; align-items: center; justify-content: center; font-size: 2.5rem; box-shadow: 0 5px 15px rgba(0,0,0,0.2); flex-shrink: 0;">🎓</div>
                <div style="text-align: left; flex: 1;">
                    <div class="university-name">KLE TECHNOLOGICAL UNIVERSITY</div>
                    <div class="department-name">School of Civil Engineering</div>
                    <div class="project-title">AI BUILDING PLAN ANALYZER</div>
                    <div class="project-subtitle">AI-Powered Floor Plan Intelligence & Engineering Assessment System</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

with header_col2:
    st.markdown("""
    <div class="compact-info-card">
        <div class="compact-section">
            <div class="compact-label">Under the Guidance of</div>
            <div class="compact-content">Dr. Roopa AK</div>
        </div>
        <hr style="margin: 1rem 0; border: none; height: 1px; background: rgba(139, 92, 246, 0.2);">
        <div class="compact-section">
            <div class="compact-label">Developed By</div>
            <div class="compact-content">Team 3 (A4)</div>
            <div class="team-list">
                <div>144 - Rahul Sanjeev</div>
                <div>143 - Adarsh</div>
                <div>154 - Gagandeep</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------
# NBC 2016 REFERENCE DATA
# ---------------------------------------------------

NBC_STANDARDS = {
    "BEDROOM": {
        "min_area": "9.5 m²",
        "min_width": "2.4 m",
        "min_height": "2.75 m"
    },
    "KITCHEN": {
        "min_area": "5.0 m²",
        "min_width": "1.8 m",
        "min_height": "2.6 m"
    },
    "LIVING": {
        "min_area": "9.5 m²",
        "min_width": "3.0 m",
        "min_height": "2.75 m"
    },
    "HALL": {
        "min_area": "9.5 m²",
        "min_width": "3.0 m",
        "min_height": "2.75 m"
    },
    "DINING": {
        "min_area": "9.5 m²",
        "min_width": "2.4 m",
        "min_height": "2.75 m"
    },
    "CORRIDOR": {
        "min_area": "-",
        "min_width": "1.0 m",
        "min_height": "2.1 m"
    },
    "TOILET": {
        "min_area": "1.1 m²",
        "min_width": "0.9 m",
        "min_height": "2.1 m"
    },
    "BATHROOM": {
        "min_area": "1.8 m²",
        "min_width": "1.2 m",
        "min_height": "2.1 m"
    },
    "WC": {
        "min_area": "1.1 m²",
        "min_width": "0.9 m",
        "min_height": "2.1 m"
    },
    "BALCONY": {
        "min_area": "-",
        "min_width": "1.2 m",
        "min_height": "2.0 m"
    },
    "STAIR": {
        "min_area": "-",
        "min_width": "0.9 m",
        "min_height": "2.0 m"
    },
    "STORE": {
        "min_area": "3.0 m²",
        "min_width": "1.2 m",
        "min_height": "2.1 m"
    },
    "UTILITY": {
        "min_area": "3.0 m²",
        "min_width": "1.5 m",
        "min_height": "2.1 m"
    }
}

# ---------------------------------------------------
# INITIALIZE VARIABLES
# ---------------------------------------------------

total_area = 0
rooms_with_area = 0
total_rooms = 0
rooms_with_pass = 0
rooms_with_fail = 0

# ---------------------------------------------------
# PROJECT OBJECTIVES
# ---------------------------------------------------

st.markdown("## 🎯 Project Objectives")

st.info("""
1. Extract room information and dimensions from floor plans using OCR.

2. Map detected spaces to NBC 2016 minimum dimensional requirements.

3. Generate AI-assisted engineering recommendations and downloadable reports.
""")

st.divider()

# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------

st.subheader("📤 Upload Floor Plan")

uploaded_file = st.file_uploader(
    "Upload building floor plan",
    type=["png", "jpg", "jpeg"],
    help="Supported formats: PNG, JPG, JPEG. Best results with 300+ DPI images"
)

# ---------------------------------------------------
# MAIN PROCESSING
# ---------------------------------------------------

if uploaded_file is not None:

    st.success("✅ Floor plan uploaded successfully!")

    try:
        # READ IMAGE
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        if image is None:
            st.error("❌ Invalid image file. Unable to decode. Please upload a valid floor plan image.")
            st.stop()

        # PREPROCESS
        with st.spinner("🔍 Processing image..."):
            gray, edges, closed, contour_img = preprocess_image(image)

        # OCR EXTRACTION
        with st.spinner("📝 Extracting text and dimensions..."):
            extracted_text, dimensions = extract_text_and_dimensions(image)

        # FILTER DIMENSIONS
        filtered_dimensions = [
            d for d in dimensions
            if isinstance(d, (int, float)) and 500 < d < 20000
        ]

        # DETECT ROOMS
        detected_rooms = []
        text_upper = extracted_text.upper()
        
        room_keywords = [
            "KITCHEN", "BEDROOM", "LIVING", "HALL", "DINING",
            "TOILET", "BATHROOM", "WC", "CORRIDOR", "BALCONY",
            "STAIR", "STORE", "UTILITY"
        ]
        
        for keyword in room_keywords:
            if keyword in text_upper and keyword not in detected_rooms:
                detected_rooms.append(keyword)

        # ---------------------------------------------------
        # PIPELINE
        # ---------------------------------------------------

        st.subheader("📷 Vision Processing Pipeline")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.image(image, caption="Original", use_column_width=True)
        with col2:
            st.image(gray, caption="Gray", use_column_width=True)
        with col3:
            st.image(edges, caption="Edges", use_column_width=True)
        with col4:
            st.image(closed, caption="Closed Walls", use_column_width=True)
        with col5:
            st.image(contour_img, caption="Detected Walls", use_column_width=True)

        st.divider()

        # ---------------------------------------------------
        # OCR SECTION
        # ---------------------------------------------------

        st.subheader("🔍 OCR Extraction")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.text_area("Extracted OCR Text", extracted_text, height=220)

        with col2:
            st.markdown("### 📏 Detected Dimensions")
            if filtered_dimensions:
                for dim in filtered_dimensions:
                    feet_value = round(dim / 304.8, 2)
                    meters_value = round(dim / 1000, 2)
                    st.success(f"{meters_value} m ({feet_value} ft)")
                st.info(f"✓ {len(filtered_dimensions)} valid dimensions extracted")
            else:
                st.warning("No valid dimensions detected.")

        # ---------------------------------------------------
        # BUILDING DIMENSIONS
        # ---------------------------------------------------

        if filtered_dimensions:
            max_dim = max(filtered_dimensions)
            min_dim = min(filtered_dimensions)
            length_m = round(max_dim / 1000, 2)
            width_m = round(min_dim / 1000, 2)

            st.divider()
            st.markdown("## 🏢 Building Dimension Summary")
            st.info(f"Largest Extracted Dimension: {length_m} m")
            st.info(f"Smallest Extracted Dimension: {width_m} m")

        # ---------------------------------------------------
        # INPUT VS OUTPUT COMPARISON
        # ---------------------------------------------------

        st.divider()
        st.subheader("🔄 Input vs Output Comparison")

        comparison_df = pd.DataFrame({
            "Input": [
                "Floor Plan Image",
                "Architectural Drawing",
                "Raw Layout Information",
                "Manual Review Process"
            ],
            "Output": [
                "OCR Extracted Text",
                "Detected Dimensions",
                "Engineering Assessment",
                "AI Recommendations"
            ]
        })
        st.table(comparison_df)

        # ---------------------------------------------------
        # ENGINEERING PARAMETER ASSESSMENT
        # ---------------------------------------------------

        st.divider()
        st.subheader("📋 Engineering Parameter Assessment")

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Assessment Status", value="Completed")
        with col2:
            st.metric(label="Rooms Detected", value=str(len(detected_rooms)))

        # ---------------------------------------------------
        # NBC 2016 MINIMUM DIMENSIONAL REQUIREMENTS
        # ---------------------------------------------------

        st.divider()
        st.markdown("## 📐 NBC 2016 Minimum Dimensional Requirements")

        if detected_rooms:
            nbc_data = {
                "Space": [],
                "Minimum Area": [],
                "Minimum Width": [],
                "Minimum Height": []
            }
            
            for room in detected_rooms:
                if room in NBC_STANDARDS:
                    nbc_data["Space"].append(room.capitalize())
                    nbc_data["Minimum Area"].append(NBC_STANDARDS[room]["min_area"])
                    nbc_data["Minimum Width"].append(NBC_STANDARDS[room]["min_width"])
                    nbc_data["Minimum Height"].append(NBC_STANDARDS[room]["min_height"])
            
            if nbc_data["Space"]:
                st.table(pd.DataFrame(nbc_data))
                st.info("""
**Dynamic NBC Mapping:**

Unlike static compliance sheets, the system dynamically generates NBC reference requirements only for the spaces identified in the uploaded floor plan.
                """)
            else:
                st.info("No matching NBC standards found for detected rooms")
        else:
            st.warning("No rooms detected in OCR text. Showing general NBC 2016 reference values.")
            general_nbc = pd.DataFrame({
                "Space": ["Bedroom", "Kitchen", "Corridor", "Toilet/WC"],
                "Minimum Area": ["9.5 m²", "5.0 m²", "-", "1.1 m²"],
                "Minimum Width": ["2.4 m", "1.8 m", "1.0 m", "0.9 m"],
                "Minimum Height": ["2.75 m", "2.6 m", "2.1 m", "2.1 m"]
            })
            st.table(general_nbc)

        # ---------------------------------------------------
        # NBC REFERENCE ASSESSMENT
        # ---------------------------------------------------

        st.divider()
        st.markdown("## 🔍 NBC Reference Assessment")

        if detected_rooms and filtered_dimensions:
            assessment_data = {
                "Room": [],
                "NBC Min Width": [],
                "Detected Width": [],
                "Status": [],
                "Deficiency": []
            }
            
            failed_rooms = []
            passed_count = 0
            
            for i, room in enumerate(detected_rooms):
                assessment_data["Room"].append(room.capitalize())
                
                if room in NBC_STANDARDS:
                    nbc_min = NBC_STANDARDS[room]["min_width"]
                    assessment_data["NBC Min Width"].append(nbc_min)
                else:
                    assessment_data["NBC Min Width"].append("-")
                
                if i < len(filtered_dimensions):
                    dim_m = round(filtered_dimensions[i] / 1000, 2)
                    assessment_data["Detected Width"].append(f"{dim_m} m")
                    
                    if room in NBC_STANDARDS:
                        nbc_min_width = float(NBC_STANDARDS[room]["min_width"].split()[0])
                        deficiency = round(nbc_min_width - dim_m, 2)
                        
                        if dim_m >= nbc_min_width:
                            assessment_data["Status"].append("✅ Satisfies")
                            assessment_data["Deficiency"].append("-")
                            passed_count += 1
                        else:
                            assessment_data["Status"].append("❌ Requires Modification")
                            assessment_data["Deficiency"].append(f"+{deficiency} m")
                            failed_rooms.append({
                                "room": room.capitalize(),
                                "detected": dim_m,
                                "required": nbc_min_width,
                                "deficiency": deficiency
                            })
                    else:
                        assessment_data["Status"].append("⚠️ Review Required")
                        assessment_data["Deficiency"].append("-")
                else:
                    assessment_data["Detected Width"].append("Not Available")
                    assessment_data["Status"].append("⚠️ Review Required")
                    assessment_data["Deficiency"].append("-")
            
            assessment_df = pd.DataFrame(assessment_data)
            st.table(assessment_df)
            
            st.warning("""
**Important Note on Room-Dimension Association:**

Dimensions are associated sequentially with detected rooms for preliminary assessment purposes.

This is an **approximate association** and does not represent automatic room boundary detection or spatial mapping.
""")

            # ---------------------------------------------------
            # ESTIMATED ROOM AREA
            # ---------------------------------------------------

            st.markdown("## 📐 Estimated Room Area (Preliminary)")

            area_data = {
                "Room": [],
                "Length (m)": [],
                "Width (m)": [],
                "Estimated Area (m²)": [],
                "NBC Min Area": [],
                "Area Status": []
            }

            total_area = 0
            rooms_with_area = 0

            for i, room in enumerate(detected_rooms):
                if i * 2 + 1 < len(filtered_dimensions):
                    length_mm = filtered_dimensions[i * 2]
                    width_mm = filtered_dimensions[i * 2 + 1]
                    
                    length_m = round(length_mm / 1000, 2)
                    width_m = round(width_mm / 1000, 2)
                    area_m2 = round(length_m * width_m, 2)
                    
                    total_area += area_m2
                    rooms_with_area += 1
                    
                    room_key = room
                    if room_key in NBC_STANDARDS:
                        nbc_min_area_str = NBC_STANDARDS[room_key]["min_area"]
                        if nbc_min_area_str == "-":
                            nbc_min_area = None
                            area_status = "⚠️ No NBC min"
                        else:
                            nbc_min_area = float(nbc_min_area_str.split()[0])
                            if area_m2 >= nbc_min_area:
                                area_status = "✅ Satisfies"
                            else:
                                area_status = f"❌ Below ({nbc_min_area} m²)"
                    else:
                        nbc_min_area = None
                        area_status = "⚠️ Review"
                    
                    area_data["Room"].append(room.capitalize())
                    area_data["Length (m)"].append(length_m)
                    area_data["Width (m)"].append(width_m)
                    area_data["Estimated Area (m²)"].append(area_m2)
                    area_data["NBC Min Area"].append(nbc_min_area_str if nbc_min_area is not None else "-")
                    area_data["Area Status"].append(area_status)

            if area_data["Room"]:
                area_df = pd.DataFrame(area_data)
                st.table(area_df)
                
                st.info(f"""
**Total Estimated Built-up Area: {total_area} m²** (from {rooms_with_area} rooms)

*Note: Areas are estimated using OCR-extracted dimensions and sequential association. 
All dimensions are interpreted as metres.*
                """)
            else:
                st.warning("Insufficient dimension pairs for area calculation.")

            # ---------------------------------------------------
            # ASSESSMENT COVERAGE DASHBOARD
            # ---------------------------------------------------

            st.markdown("## 📊 Assessment Coverage Dashboard")

            total_rooms = len(detected_rooms)
            total_dimensions = len(filtered_dimensions)
            rooms_with_pass = passed_count
            rooms_with_fail = len(failed_rooms)
            params_evaluated = 2
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric(
                    label="🏠 Rooms Detected",
                    value=total_rooms,
                    delta="rooms analyzed"
                )
            with col2:
                st.metric(
                    label="📏 Dimensions Extracted",
                    value=total_dimensions,
                    delta="valid measurements"
                )
            with col3:
                st.metric(
                    label="✅ Rooms Satisfying",
                    value=rooms_with_pass,
                    delta=f"of {total_rooms} rooms"
                )
            with col4:
                st.metric(
                    label="⚠️ Rooms Requiring Modification",
                    value=rooms_with_fail,
                    delta=f"of {total_rooms} rooms"
                )
            with col5:
                st.metric(
                    label="📋 Parameters Evaluated",
                    value=params_evaluated,
                    delta="Width, Area"
                )
            
            st.markdown("### Assessment Statistics")
            
            stats_data = {
                "Metric": [
                    "Total Rooms Evaluated",
                    "Dimensions Extracted",
                    "Rooms Satisfying Evaluated Parameters",
                    "Rooms Requiring Modification",
                    "Total Estimated Area",
                    "Assessment Scope"
                ],
                "Value": [
                    f"{total_rooms} rooms",
                    f"{total_dimensions} dimensions",
                    f"{rooms_with_pass} rooms",
                    f"{rooms_with_fail} rooms",
                    f"{total_area} m²",
                    "NBC Width & Area Reference"
                ]
            }
            
            stats_df = pd.DataFrame(stats_data)
            st.table(stats_df)
            
            st.markdown("### Assessment Result Summary")
            
            if rooms_with_fail == 0:
                st.success(f"""
✅ **ALL ROOMS SATISFYING EVALUATED PARAMETERS**

All {total_rooms} evaluated rooms satisfy NBC 2016 minimum dimensional requirements.
No modifications required for the assessed parameters.
                """)
            else:
                st.warning(f"""
⚠️ **MODIFICATIONS REQUIRED FOR COMPLIANCE**

{rooms_with_fail} out of {total_rooms} rooms require modification to meet NBC 2016 standards.
""")
                for failed in failed_rooms:
                    st.write(f"• **{failed['room']}**: Increase width by {failed['deficiency']} m (from {failed['detected']} m to {failed['required']} m)")

        elif detected_rooms and not filtered_dimensions:
            st.warning("Rooms detected but no valid dimensions could be extracted.")
        elif filtered_dimensions and not detected_rooms:
            st.info("Dimensions detected but no specific rooms identified.")
        else:
            st.info("Unable to perform assessment. Upload a clearer floor plan image.")

        # ---------------------------------------------------
        # IMPORTANT COMPLIANCE NOTE
        # ---------------------------------------------------

        st.divider()
        st.warning("""
**Important Note:**

The current system evaluates NBC 2016 reference requirements based on OCR information.

All extracted dimensions are interpreted as metres.

Advanced compliance checks such as staircase geometry, fire safety, ventilation, accessibility, and FAR/FSI are not evaluated.

**This tool serves as a preliminary assessment aid and does not replace professional engineering review.**
""")

        # ---------------------------------------------------
        # AI CHAT
        # ---------------------------------------------------

        st.divider()
        st.subheader("🧠 AI Engineering Consultant (Meta Llama 3.3 70B)")

        st.info("""
**Generative AI Model:** Meta Llama 3.3 70B Versatile

**Inference Platform:** Groq API
""")

        st.success("""
**Purpose:**

Generate engineering recommendations based on extracted floor plan information.
""")

        question = st.text_input("Ask architectural or civil engineering questions")

        response = "No AI response generated yet."

        if st.button("Ask AI"):
            if question.strip() == "":
                st.warning("Please enter a question.")
            else:
                with st.spinner("🧠 Analyzing floor plan with AI..."):
                    try:
                        enhanced_question = f"""
                    
The user has uploaded a floor plan. All extracted dimensions in this analysis are interpreted as METRES.

Analyze the following floor plan information and provide engineering recommendations:
{question}

Important: Use metres as the unit of measurement throughout your response.
"""
                        response = ask_ai(extracted_text, dimensions, enhanced_question)
                    except Exception as e:
                        st.error(f"❌ Error generating AI response: {str(e)}")
                        response = "Unable to generate AI response. Please try again."

                st.markdown("### 🧠 AI Architectural Recommendation")
                st.info(response)

        # ---------------------------------------------------
        # PDF REPORT GENERATION
        # ---------------------------------------------------

        st.divider()
        st.subheader("📄 Engineering Report")

        dimension_summary = "\n".join([f"{round(d/1000, 2)} m" for d in filtered_dimensions])
        rooms_summary = ", ".join([room.capitalize() for room in detected_rooms]) if detected_rooms else "None detected"
        
        suggestions_summary = """Review staircase geometry manually.
Verify ventilation provisions.
Verify fire safety requirements.
Verify accessibility provisions."""

        report_text = f"""
AI BUILDING PLAN ANALYZER REPORT

----------------------------------------

OCR EXTRACTED TEXT:
{extracted_text}

----------------------------------------

DETECTED ROOMS:
{rooms_summary}

DETECTED ROOM COUNT:
{len(detected_rooms)}

VALID EXTRACTED DIMENSIONS (in metres):
{dimension_summary}

TOTAL ESTIMATED AREA:
{total_area} m² (from {rooms_with_area} rooms)

----------------------------------------

ASSESSMENT SUMMARY:
- Rooms Evaluated: {total_rooms}
- Rooms Satisfying Evaluated Parameters: {rooms_with_pass}
- Rooms Requiring Modification: {rooms_with_fail}

PARAMETERS EVALUATED:
- NBC 2016 Minimum Width Requirements
- NBC 2016 Minimum Area Requirements

----------------------------------------

NBC 2016 REFERENCE ASSESSMENT:
Dimensions compared against minimum requirements.
Preliminary room-dimension association based on sequential detection.
All dimensions interpreted as metres.

----------------------------------------

ENGINEERING SUGGESTIONS:
{suggestions_summary}

----------------------------------------

AI RESPONSE:
{response}

----------------------------------------

ASSESSMENT LIMITATIONS:
- Room-to-dimension mapping is approximate
- Fire safety not evaluated
- Structural design not evaluated
- Ventilation not evaluated
- Accessibility not evaluated

DISCLAIMER:
Preliminary assessment only. Professional review required.
Unit System: Metric (metres)
"""

        st.text_area("Generated Engineering Report", report_text, height=300)

        # CREATE PDF
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=10)
            pdf.multi_cell(0, 5, report_text)
            pdf.output("Engineering_Report.pdf")

            # DOWNLOAD BUTTON
            with open("Engineering_Report.pdf", "rb") as file:
                st.download_button(
                    label="📥 Download Engineering Report PDF",
                    data=file,
                    file_name="Engineering_Report.pdf",
                    mime="application/pdf"
                )
        except Exception as e:
            st.error(f"❌ Error generating PDF: {str(e)}")

        # ---------------------------------------------------
        # FINAL CONCLUSION
        # ---------------------------------------------------

        st.divider()
        st.markdown("## 📝 Final Conclusion")

        st.success("""
**AI-Assisted Floor Plan Assessment Completed Successfully.**

The system integrates Computer Vision, OCR, Generative AI, and NBC 2016 reference data for automated floor plan analysis.

Room types are identified, dimensions extracted (interpreted as metres), and preliminary areas calculated with full transparency about limitations.

The assessment serves as a decision-support tool for preliminary engineering review.

Professional engineering verification is required for final approval.
""")

    except Exception as e:
        st.error(f"❌ An error occurred during processing: {str(e)}")
        st.info("Please try uploading a different floor plan image or check your input file.")

else:
    st.warning("⚠️ Please upload a floor plan image to begin analysis.")

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("""
<div class="footer">
    <div class="footer-title">AI Building Plan Analyzer</div>
    <div class="footer-text">
        Developed by <strong>Team 3 (A4)</strong><br>
        Under the Guidance of <strong>Dr. Roopa AK</strong><br>
        School of Civil Engineering<br>
        KLE Technological University
    </div>
</div>
""", unsafe_allow_html=True)