"""
OncoVision AI — PDF Report Generator
======================================
Generates a professional medical-style PDF diagnostic report
using ReportLab. Designed for clinical presentation and archival.
"""

import io
import os
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    HRFlowable,
)

from typing import Optional


def generate_pdf_report(record_dict: dict, image_path: Optional[str] = None) -> bytes:
    """
    Generate a clinical-grade PDF report from a diagnosis record dictionary.

    Parameters:
        record_dict: Serialized DiagnosisRecord (from .to_dict())
        image_path: Absolute path to the saved biopsy image

    Returns:
        PDF file content as bytes
    """

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
    )

    # -- Styles ------------------------------------------------------------
    styles = getSampleStyleSheet()

    style_title = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=22,
        spaceAfter=4,
        textColor=colors.HexColor("#111111"),
        fontName="Helvetica-Bold",
    )

    style_subtitle = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#71717a"),
        spaceAfter=16,
        fontName="Helvetica",
    )

    style_section = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontSize=11,
        spaceBefore=14,
        spaceAfter=6,
        textColor=colors.HexColor("#111111"),
        fontName="Helvetica-Bold",
    )

    style_body = ParagraphStyle(
        "BodyText",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#3f3f46"),
        fontName="Helvetica",
    )

    style_label = ParagraphStyle(
        "Label",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.HexColor("#71717a"),
        fontName="Helvetica",
        spaceAfter=2,
    )

    style_disclaimer = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontSize=7.5,
        textColor=colors.HexColor("#a1a1aa"),
        fontName="Helvetica-Oblique",
        leading=10,
    )

    # -- Build flowables ---------------------------------------------------
    flowables = []

    # Header
    flowables.append(Paragraph("OncoVision AI", style_title))
    flowables.append(
        Paragraph(
            f"Histopathological Diagnostic Report &nbsp;·&nbsp; "
            f"Generated {datetime.now().strftime('%B %d, %Y at %H:%M')}",
            style_subtitle,
        )
    )
    flowables.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e4e4e7")))
    flowables.append(Spacer(1, 6 * mm))

    # Report metadata
    meta_data = [
        ["Report ID", record_dict.get("id", "N/A")],
        ["Original File", record_dict.get("filename", "N/A")],
        ["Analysis Date", record_dict.get("created_at", "N/A")[:19].replace("T", "  ")],
    ]
    meta_table = Table(meta_data, colWidths=[35 * mm, 130 * mm])
    meta_table.setStyle(
        TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#71717a")),
            ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor("#18181b")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("ALIGN", (0, 0), (0, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ])
    )
    flowables.append(meta_table)
    flowables.append(Spacer(1, 6 * mm))

    # Biopsy image (if available)
    if image_path and os.path.exists(image_path):
        flowables.append(Paragraph("BIOPSY IMAGE", style_section))
        try:
            img = RLImage(image_path, width=80 * mm, height=60 * mm)
            img.hAlign = "LEFT"
            flowables.append(img)
            flowables.append(Spacer(1, 4 * mm))
        except Exception:
            flowables.append(Paragraph("<i>Image could not be embedded.</i>", style_body))

    # Classification
    flowables.append(Paragraph("CLASSIFICATION", style_section))
    prediction = record_dict.get("prediction", "N/A")
    confidence = record_dict.get("confidence", 0)
    risk_label = record_dict.get("risk_label", "N/A")

    class_data = [
        ["Prediction", prediction],
        ["Confidence", f"{confidence}%"],
        ["Risk Level", risk_label],
    ]
    class_table = Table(class_data, colWidths=[35 * mm, 130 * mm])
    class_table.setStyle(
        TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#71717a")),
            ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor("#18181b")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("ALIGN", (0, 0), (0, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#e4e4e7")),
        ])
    )
    flowables.append(class_table)
    flowables.append(Spacer(1, 4 * mm))

    # Biological indicators
    flowables.append(Paragraph("MORPHOLOGICAL BIOMARKERS", style_section))
    indicators = record_dict.get("biological_indicators", {})

    bio_data = [
        ["Marker", "Status"],
        ["Nuclear-to-Cytoplasmic Ratio", indicators.get("nc_ratio", "N/A")],
        ["Pleomorphism", indicators.get("pleomorphism", "N/A")],
        ["Hyperchromasia", indicators.get("hyperchromasia", "N/A")],
    ]
    bio_table = Table(bio_data, colWidths=[80 * mm, 85 * mm])
    bio_table.setStyle(
        TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#71717a")),
            ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#18181b")),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f4f4f5")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#e4e4e7")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )
    flowables.append(bio_table)
    flowables.append(Spacer(1, 4 * mm))

    # Case analysis
    case_analysis = record_dict.get("case_analysis", {})
    if case_analysis:
        flowables.append(Paragraph("CASE ANALYSIS", style_section))

        for key in sorted(case_analysis.keys()):
            if key == "8_classification_percentage":
                continue  # Redundant — skip
            value = case_analysis[key]
            # Clean label: "0_pathogenesis" -> "Pathogenesis"
            label = " ".join(
                word.capitalize() for word in key.split("_")[1:]
            )
            flowables.append(Paragraph(label.upper(), style_label))
            flowables.append(Paragraph(str(value), style_body))
            flowables.append(Spacer(1, 2 * mm))

    # Disclaimer
    flowables.append(Spacer(1, 8 * mm))
    flowables.append(HRFlowable(width="100%", thickness=0.3, color=colors.HexColor("#e4e4e7")))
    flowables.append(Spacer(1, 3 * mm))
    flowables.append(
        Paragraph(
            "⚠ DISCLAIMER: This report was generated by OncoVision AI, an educational prototype. "
            "It is not a substitute for a qualified pathologist or clinical diagnosis. "
            "All findings must be verified by a licensed medical professional before any "
            "clinical decisions are made.",
            style_disclaimer,
        )
    )
    flowables.append(Spacer(1, 2 * mm))
    flowables.append(
        Paragraph(
            f"OncoVision AI v1.0 · Gemini 2.5 Flash · Logistic Regression Confidence Model · "
            f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            style_disclaimer,
        )
    )

    # -- Build PDF ---------------------------------------------------------
    doc.build(flowables)
    return buffer.getvalue()
