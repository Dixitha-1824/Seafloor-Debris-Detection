import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import datetime
import cv2
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from io import BytesIO

# page config
st.set_page_config(page_title="Marine Monitoring System", layout="wide")


# model import
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# sidebar
st.sidebar.title(" SYSTEM PANEL")

st.sidebar.markdown("###  System Status")
st.sidebar.success("AI Model: Active")
st.sidebar.success("Detection Engine: Running")
st.sidebar.info("Environment: Underwater Mode")

mode = st.sidebar.radio("Mode", ["Static Image", "Live Monitoring"])

confidence = st.sidebar.slider("Confidence", 0.1, 1.0, 0.25)

start = st.sidebar.button(" Start Detection")

# i/p
image = None

st.markdown(
    """
    <h2 style='text-align: center; color: black;'>
    Seafloor Debris Detection using <br>
    Integrated ResUNet and YOLOv8 Model
    </h2>
    <hr>
    """,
    unsafe_allow_html=True
)

if mode == "Static Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)

else:
    camera = st.camera_input(" Capture Image")
    if camera:
        image = Image.open(camera)

# session
if "logs" not in st.session_state:
    st.session_state.logs = []

# process
if image is not None and start:

    with st.spinner(" Analyzing..."):

        img_np = np.array(image)

        results = model.predict(source=img_np, conf=confidence)
        boxes = results[0].boxes
        annotated = results[0].plot()

        debris_count = len(boxes)
        confidences = [float(box.conf) for box in boxes]
        classes = [int(box.cls) for box in boxes]

        names = model.names
        labels = [names[c] for c in classes]

        # Risk logic
        if debris_count == 0:
            risk = "SAFE"
        elif debris_count < 3:
            risk = "MODERATE"
        else:
            risk = "HIGH RISK"

        # Debris score
        debris_score = min(debris_count * 20, 100)

        # Environment status
        env_status = "Clean" if debris_count == 0 else "Polluted"

        # Logs
        for label, conf in zip(labels, confidences):
            log = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {label} ({round(conf,2)})"
            st.session_state.logs.append(log)

    tab1, tab2, tab3 = st.tabs([" Results", " Analytics", " Logs"])

    # results
    with tab1:
        col1, col2 = st.columns(2)

        col1.image(image, caption="Original", use_container_width=True)
        col2.image(annotated, caption="Detected", use_container_width=True)

        st.markdown("---")

        m1, m2, m3, m4 = st.columns(4)

        avg_conf = round(sum(confidences)/len(confidences),2) if confidences else 0

        m1.metric("Objects", debris_count)
        m2.metric("Avg Confidence", avg_conf)
        m3.metric("Risk Level", risk)
        m4.metric("Debris Score", f"{debris_score}%")

        st.markdown(f"### Environment Status: **{env_status}**")

        if labels:
            st.subheader(" Detected Classes")
            class_df = pd.Series(labels).value_counts().reset_index()
            class_df.columns = ["Class", "Count"]
            st.dataframe(class_df)

    # analytics
    with tab2:

        if confidences:

            df = pd.DataFrame({
                "Debris": range(1, len(confidences)+1),
                "Confidence": confidences
            })

            st.subheader("Confidence Distribution")
            fig1 = px.histogram(df, x="Confidence", nbins=10)
            st.plotly_chart(fig1, use_container_width=True)

            st.subheader(" Detection Telemetry")
            fig2 = px.line(df, x="Debris", y="Confidence", markers=True)
            st.plotly_chart(fig2, use_container_width=True)

            st.subheader(" Detection Heatmap")
            heat = cv2.applyColorMap(annotated, cv2.COLORMAP_JET)
            st.image(heat, caption="Heatmap", use_container_width=True)

        else:
            st.warning("No detections")

    # logs/history
    with tab3:
        for log in st.session_state.logs[::-1]:
            st.code(log)

    # report
    
    def generate_pdf():

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)

        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            'title',
            parent=styles['Title'],
            textColor=colors.HexColor("#007BFF"),
            alignment=1
        )

        heading_style = ParagraphStyle(
            'heading',
            parent=styles['Heading2'],
            textColor=colors.HexColor("#00A86B")
        )

        normal_style = styles['Normal']

        content = []

        content.append(Paragraph("Marine Debris Detection Report", title_style))
        content.append(Spacer(1, 12))

        time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        content.append(Paragraph(f"<b>Timestamp:</b> {time}", normal_style))
        content.append(Spacer(1, 15))

        content.append(Paragraph("Detection Summary", heading_style))
        content.append(Spacer(1, 10))

        if risk == "SAFE":
            risk_color = colors.green
        elif risk == "MODERATE":
            risk_color = colors.orange
        else:
            risk_color = colors.red

        data = [
            ["Metric", "Value"],
            ["Total Objects", str(debris_count)],
            ["Average Confidence", str(avg_conf)],
            ["Risk Level", risk],
            ["Debris Score", f"{debris_score}%"],
            ["Environment Status", env_status]
        ]

        table = Table(data, colWidths=[200, 200])

        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#007BFF")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),

            ("BACKGROUND", (0,1), (-1,-1), colors.whitesmoke),

            ("GRID", (0,0), (-1,-1), 1, colors.grey),

            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),

            ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ]))

        content.append(table)
        content.append(Spacer(1, 20))

        content.append(Paragraph(
            f"<b>Risk Assessment:</b> <font color='{risk_color}'> {risk} </font>",
            styles['Normal']
        ))

        content.append(Spacer(1, 20))

        content.append(Paragraph(
            "Generated by Marine Debris Monitoring System",
            styles['Italic']
        ))

        doc.build(content)

        buffer.seek(0)
        return buffer


    # Download
    pdf_file = generate_pdf()

    st.download_button(
    label="Download PDF Report",
    data=pdf_file,
    file_name="Marine_Debris_Report.pdf",
    mime="application/pdf"
    )

else:
    st.info("Upload or capture image and click Start Detection")

# footer
st.markdown("---")
st.markdown("")

