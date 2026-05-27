"""
generate_report.py
==================
Professional PDF radiology report generator for Visukhi Medical AI RAG.
Uses ReportLab to produce a hospital-grade PDF with:
  - Visukhi Innotech company header
  - Patient demographics from PostgreSQL
  - Clinical findings table
  - AI-generated impression & recommendations
  - Download-ready from Streamlit
"""

import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, cm
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# ── Company Details ───────────────────────────────────────────────────────────
COMPANY_NAME = "Visukhi Innotech Private Limited"
COMPANY_ADDRESS_1 = "556, 14th Main Rd, Sector 3"
COMPANY_ADDRESS_2 = "HSR Layout, Bengaluru, Karnataka 560102"
COMPANY_PHONE = "Phone: 080 4975 2707"


# ── Colour Palette ────────────────────────────────────────────────────────────
BRAND_DARK   = colors.HexColor("#1a2744")   # deep navy
BRAND_ACCENT = colors.HexColor("#2e86de")   # bright blue
BRAND_LIGHT  = colors.HexColor("#f0f4f8")   # pale background
TEXT_DARK    = colors.HexColor("#2c3e50")
TEXT_MUTED   = colors.HexColor("#7f8c8d")
BORDER_COLOR = colors.HexColor("#dce1e8")
SUCCESS_GREEN = colors.HexColor("#27ae60")
WARNING_AMBER = colors.HexColor("#f39c12")
DANGER_RED    = colors.HexColor("#e74c3c")


def _confidence_color(score: float) -> colors.Color:
    """Return a colour based on the AI confidence tier."""
    if score >= 0.85:
        return SUCCESS_GREEN
    elif score >= 0.70:
        return WARNING_AMBER
    return DANGER_RED


def _build_styles():
    """Create the stylesheet for the report."""
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        "CompanyName",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        textColor=BRAND_DARK,
        alignment=TA_CENTER,
        spaceAfter=2 * mm,
        leading=20,
    ))
    styles.add(ParagraphStyle(
        "CompanyAddress",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        textColor=TEXT_MUTED,
        alignment=TA_CENTER,
        spaceAfter=1 * mm,
        leading=12,
    ))
    styles.add(ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=14,
        textColor=BRAND_ACCENT,
        alignment=TA_CENTER,
        spaceBefore=4 * mm,
        spaceAfter=6 * mm,
        leading=18,
    ))
    styles.add(ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        textColor=BRAND_DARK,
        spaceBefore=6 * mm,
        spaceAfter=3 * mm,
        leading=14,
        borderPadding=(0, 0, 2, 0),
    ))
    styles.add(ParagraphStyle(
        "BodyText2",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        textColor=TEXT_DARK,
        leading=14,
        spaceAfter=2 * mm,
    ))
    styles.add(ParagraphStyle(
        "SmallMuted",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        textColor=TEXT_MUTED,
        alignment=TA_CENTER,
        spaceBefore=8 * mm,
        leading=10,
    ))
    styles.add(ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        textColor=TEXT_DARK,
        leading=12,
    ))
    styles.add(ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        textColor=colors.white,
        leading=12,
    ))

    return styles


def generate_pdf_report(
    patient_id: int,
    patient_name: str,
    gender: str,
    study_rows: list,
    ai_impression: str = "",
    ai_recommendations: str = "",
) -> bytes:
    """
    Generate a professional PDF radiology report and return it as bytes.

    Parameters
    ----------
    patient_id : int
    patient_name : str
    gender : str
    study_rows : list[tuple]
        Each tuple: (study_date, priority, image_type, findings, confidence_score)
    ai_impression : str
        LLM-generated clinical impression text.
    ai_recommendations : str
        LLM-generated follow-up recommendations.

    Returns
    -------
    bytes
        The PDF file content, ready for Streamlit download.
    """
    buf = io.BytesIO()
    styles = _build_styles()

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        title=f"Radiology Report — {patient_name}",
        author=COMPANY_NAME,
    )

    elements = []

    # ── HEADER ────────────────────────────────────────────────────────────────
    elements.append(Paragraph(COMPANY_NAME, styles["CompanyName"]))
    elements.append(Paragraph(COMPANY_ADDRESS_1, styles["CompanyAddress"]))
    elements.append(Paragraph(COMPANY_ADDRESS_2, styles["CompanyAddress"]))
    elements.append(Paragraph(COMPANY_PHONE, styles["CompanyAddress"]))
    elements.append(Spacer(1, 2 * mm))
    elements.append(HRFlowable(
        width="100%", thickness=1.5, color=BRAND_ACCENT,
        spaceBefore=2 * mm, spaceAfter=2 * mm
    ))
    elements.append(Paragraph("RADIOLOGY REPORT", styles["ReportTitle"]))
    elements.append(HRFlowable(
        width="100%", thickness=0.5, color=BORDER_COLOR,
        spaceBefore=0, spaceAfter=4 * mm
    ))

    # ── SECTION 1 — PATIENT INFORMATION ───────────────────────────────────────
    elements.append(Paragraph("1. Patient Information", styles["SectionHeading"]))

    report_date = datetime.now().strftime("%d %B %Y, %I:%M %p")
    gender_display = gender.capitalize() if gender else "Not specified"

    info_data = [
        [Paragraph("<b>Patient ID</b>", styles["TableCell"]),
         Paragraph(str(patient_id), styles["TableCell"]),
         Paragraph("<b>Report Date</b>", styles["TableCell"]),
         Paragraph(report_date, styles["TableCell"])],
        [Paragraph("<b>Patient Name</b>", styles["TableCell"]),
         Paragraph(patient_name, styles["TableCell"]),
         Paragraph("<b>Gender</b>", styles["TableCell"]),
         Paragraph(gender_display, styles["TableCell"])],
    ]

    info_table = Table(info_data, colWidths=[30 * mm, 55 * mm, 30 * mm, 55 * mm])
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), BRAND_LIGHT),
        ("BACKGROUND", (2, 0), (2, -1), BRAND_LIGHT),
        ("BOX",        (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("INNERGRID",  (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
    ]))
    elements.append(info_table)

    # ── SECTION 2 — CLINICAL FINDINGS ─────────────────────────────────────────
    elements.append(Paragraph("2. Clinical Findings", styles["SectionHeading"]))

    if study_rows:
        # Table header row
        header_cells = [
            Paragraph("Study Date", styles["TableHeader"]),
            Paragraph("Modality", styles["TableHeader"]),
            Paragraph("Priority", styles["TableHeader"]),
            Paragraph("Findings", styles["TableHeader"]),
            Paragraph("AI Conf.", styles["TableHeader"]),
        ]

        table_data = [header_cells]

        for row in study_rows:
            study_date, priority, image_type, findings, confidence = row

            # Format date
            if hasattr(study_date, "strftime"):
                date_str = study_date.strftime("%d %b %Y")
            else:
                date_str = str(study_date) if study_date else "—"

            modality_str = (image_type or "Unknown").upper()
            priority_str = (priority or "—").capitalize()
            findings_str = findings if findings else "No clinical findings recorded"

            # Confidence display
            if confidence is not None:
                conf_val = float(confidence)
                conf_color = _confidence_color(conf_val)
                conf_str = f'<font color="{conf_color.hexval()}">{conf_val:.0%}</font>'
            else:
                conf_str = "—"

            table_data.append([
                Paragraph(date_str, styles["TableCell"]),
                Paragraph(f"<b>{modality_str}</b>", styles["TableCell"]),
                Paragraph(priority_str, styles["TableCell"]),
                Paragraph(findings_str, styles["TableCell"]),
                Paragraph(conf_str, styles["TableCell"]),
            ])

        findings_table = Table(
            table_data,
            colWidths=[24 * mm, 22 * mm, 20 * mm, 75 * mm, 18 * mm],
            repeatRows=1,
        )
        findings_table.setStyle(TableStyle([
            # Header styling
            ("BACKGROUND",    (0, 0), (-1, 0), BRAND_DARK),
            ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
            # Alternating row shading
            *[("BACKGROUND", (0, i), (-1, i), BRAND_LIGHT)
              for i in range(2, len(table_data), 2)],
            # Grid
            ("BOX",        (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ("INNERGRID",  (0, 0), (-1, -1), 0.3, BORDER_COLOR),
            ("VALIGN",     (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING",   (0, 0), (-1, -1), 4),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
        ]))
        elements.append(findings_table)
    else:
        elements.append(Paragraph(
            "No clinical studies found for this patient.",
            styles["BodyText2"]
        ))

    # ── SECTION 3 — IMPRESSION ────────────────────────────────────────────────
    elements.append(Paragraph("3. Impression", styles["SectionHeading"]))

    if ai_impression and ai_impression.strip():
        # Split by newlines to preserve paragraph structure from LLM
        for para in ai_impression.strip().split("\n"):
            para = para.strip()
            if para:
                elements.append(Paragraph(para, styles["BodyText2"]))
    else:
        # Auto-generate summary from findings
        if study_rows:
            unique_findings = set()
            modalities_used = set()
            for row in study_rows:
                _, _, image_type, findings, _ = row
                if findings:
                    unique_findings.add(findings)
                if image_type:
                    modalities_used.add(image_type.upper())

            modality_list = ", ".join(sorted(modalities_used)) if modalities_used else "imaging"
            elements.append(Paragraph(
                f"Based on review of {len(study_rows)} {modality_list} "
                f"{'study' if len(study_rows) == 1 else 'studies'} for patient "
                f"{patient_name}:",
                styles["BodyText2"]
            ))
            for f in sorted(unique_findings):
                elements.append(Paragraph(f"• {f}", styles["BodyText2"]))
        else:
            elements.append(Paragraph(
                "No imaging data available for clinical impression.",
                styles["BodyText2"]
            ))

    # ── SECTION 4 — RECOMMENDATIONS ──────────────────────────────────────────
    elements.append(Paragraph("4. Recommendations", styles["SectionHeading"]))

    if ai_recommendations and ai_recommendations.strip():
        for para in ai_recommendations.strip().split("\n"):
            para = para.strip()
            if para:
                elements.append(Paragraph(para, styles["BodyText2"]))
    else:
        # Auto-generate from findings severity
        if study_rows:
            has_abnormality = any(
                "abnormality" in (r[3] or "").lower() or "suspicious" in (r[3] or "").lower()
                for r in study_rows
            )
            has_followup = any("follow" in (r[3] or "").lower() for r in study_rows)
            has_fracture = any("fracture" in (r[3] or "").lower() for r in study_rows)

            recs = []
            if has_abnormality or has_fracture:
                recs.append("• Urgent specialist consultation recommended.")
            if has_followup:
                recs.append("• Follow-up imaging recommended as indicated in findings.")
            if not recs:
                recs.append("• Continue routine follow-up as clinically indicated.")
            recs.append("• Correlate with clinical symptoms and laboratory results.")

            for rec in recs:
                elements.append(Paragraph(rec, styles["BodyText2"]))
        else:
            elements.append(Paragraph(
                "No recommendations — insufficient clinical data.",
                styles["BodyText2"]
            ))

    # ── FOOTER ────────────────────────────────────────────────────────────────
    elements.append(Spacer(1, 10 * mm))
    elements.append(HRFlowable(
        width="100%", thickness=0.5, color=BORDER_COLOR,
        spaceBefore=4 * mm, spaceAfter=2 * mm
    ))

    # Signature area
    sig_data = [
        [Paragraph("<b>Reporting Physician</b>", styles["TableCell"]),
         Paragraph("", styles["TableCell"]),
         Paragraph("<b>Date / Time</b>", styles["TableCell"])],
        [Paragraph("_________________________", styles["TableCell"]),
         Paragraph("", styles["TableCell"]),
         Paragraph(report_date, styles["TableCell"])],
        [Paragraph("AI-Assisted Report", styles["TableCell"]),
         Paragraph("", styles["TableCell"]),
         Paragraph(f"Report ID: VIS-{patient_id:05d}-{datetime.now().strftime('%Y%m%d')}", styles["TableCell"])],
    ]
    sig_table = Table(sig_data, colWidths=[60 * mm, 40 * mm, 60 * mm])
    sig_table.setStyle(TableStyle([
        ("VALIGN",     (0, 0), (-1, -1), "BOTTOM"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    elements.append(sig_table)

    elements.append(Paragraph(
        f"This report was generated by {COMPANY_NAME} AI-assisted radiology system. "
        "It is intended as a clinical decision support tool and should be reviewed "
        "by a qualified medical professional.",
        styles["SmallMuted"]
    ))

    # ── BUILD PDF ─────────────────────────────────────────────────────────────
    doc.build(elements)
    return buf.getvalue()


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from datetime import date
    sample_rows = [
        (date(2026, 1, 15), "high", "ct", "Suspicious nodule detected — follow-up imaging recommended.", 0.92),
        (date(2026, 2, 3),  "medium", "mri", "Mild inflammation observed in the surrounding tissue.", 0.78),
        (date(2025, 12, 1), "low", "xray", "Normal study, no abnormalities detected.", 0.95),
    ]

    pdf_bytes = generate_pdf_report(
        patient_id=42,
        patient_name="Priya Reddy",
        gender="female",
        study_rows=sample_rows,
    )

    output_path = "sample_report.pdf"
    with open(output_path, "wb") as f:
        f.write(pdf_bytes)
    print(f"✅ Sample report saved: {output_path} ({len(pdf_bytes):,} bytes)")
