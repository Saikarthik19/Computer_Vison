"""
Automated PDF Intelligence Report Generator
Produces a formal NTRO / SIH 2024 New-Road Change Alert Intelligence Dossier.
"""

import os
from typing import Dict, Any, Optional
import cv2

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether
)


def generate_change_alert_pdf(
    alerts_dossier: Dict[str, Any],
    composite_image_path: str,
    output_pdf_path: str,
    t1_image_path: Optional[str] = None,
    t2_image_path: Optional[str] = None
) -> str:
    """
    Renders a multi-page defense-grade intelligence PDF report.
    """
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0F172A'),
        alignment=1, # Center
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#DC2626'), # Alert red
        alignment=1,
        spaceAfter=15
    )

    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#1E293B'),
        spaceBefore=10,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#334155')
    )

    elements = []

    # 1. Header Banner
    header_data = [
        [
            Paragraph("<b>NATIONAL TECHNICAL RESEARCH ORGANISATION (NTRO)</b><br/>SIH 2024 · PS-1564 Automated Satellite Change Alert Center", body_style),
            Paragraph(f"<b>CLASSIFICATION:</b> OFFICIAL AUDIT<br/><b>DATE:</b> {alerts_dossier.get('generated_at')}", body_style)
        ]
    ]
    t_header = Table(header_data, colWidths=[4.0 * inch, 3.2 * inch])
    t_header.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F1F5F9')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#CBD5E1')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(t_header)
    elements.append(Spacer(1, 12))

    # 2. Main Title
    elements.append(Paragraph("SATELLITE ROAD-NETWORK EXTRACTOR & CHANGE DETECTION DOSSIER", title_style))
    severity = alerts_dossier.get("overall_severity", "HIGH")
    elements.append(Paragraph(f"ALERT STATUS: [{severity}] - {alerts_dossier.get('classification')}", subtitle_style))

    # 3. Surveillance Sector Metadata & Key Metrics
    metrics = alerts_dossier.get("summary_metrics", {})
    kpi_data = [
        [
            Paragraph(f"<b>Dossier ID:</b> {alerts_dossier.get('dossier_id')}", body_style),
            Paragraph(f"<b>Target Sector:</b> {alerts_dossier.get('target_sector')}", body_style),
            Paragraph(f"<b>Coordinates:</b> {alerts_dossier.get('center_coordinates')}", body_style)
        ],
        [
            Paragraph(f"<b>New Roads Detected:</b> {metrics.get('total_new_roads_count', 0)} Segments", body_style),
            Paragraph(f"<b>Total New Length:</b> {metrics.get('total_new_length_km', 0.0)} km ({metrics.get('total_new_length_meters', 0.0)} m)", body_style),
            Paragraph(f"<b>New Road Footprint Area:</b> {metrics.get('total_new_road_area_m2', 0.0)} m²", body_style)
        ]
    ]
    t_kpi = Table(kpi_data, colWidths=[2.4 * inch, 2.5 * inch, 2.3 * inch])
    t_kpi.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#E2E8F0')),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
    ]))
    elements.append(t_kpi)
    elements.append(Spacer(1, 14))

    # 4. Imagery Evidence Section
    elements.append(Paragraph("1. SATELLITE IMAGERY EVIDENCE & TACTICAL OVERLAY", section_heading))
    elements.append(Paragraph("Bi-temporal change map: Baseline roads displayed in cyan, newly constructed corridors highlighted in crimson red with skeleton centerlines.", body_style))
    elements.append(Spacer(1, 6))

    if os.path.exists(composite_image_path):
        # Resize image for PDF layout
        img_w = 7.2 * inch
        img_h = 3.6 * inch
        elements.append(RLImage(composite_image_path, width=img_w, height=img_h))
        elements.append(Spacer(1, 10))

    # 5. Alert Incident Breakdown Table
    elements.append(Paragraph("2. VERIFIED NEW ROAD INCIDENT MANIFEST", section_heading))
    table_rows = [
        ["Alert ID", "Severity", "Road Classification", "Length (m)", "Intersections", "Confidence"]
    ]

    for alert in alerts_dossier.get("alerts", []):
        table_rows.append([
            alert.get("alert_id"),
            alert.get("severity"),
            alert.get("type"),
            f"{alert.get('length_meters')} m",
            str(alert.get("intersections")),
            f"{alert.get('confidence_score'):.1f}%"
        ])

    if len(table_rows) == 1:
        table_rows.append(["None", "NORMAL", "No new roads identified", "0 m", "0", "100%"])

    t_alerts = Table(table_rows, colWidths=[1.4 * inch, 1.0 * inch, 2.4 * inch, 1.0 * inch, 0.7 * inch, 0.7 * inch])
    t_alerts.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E293B')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (2, 1), (2, -1), 'LEFT'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    elements.append(t_alerts)
    elements.append(Spacer(1, 14))

    # 6. Actionable Intelligence Recommendations
    elements.append(Paragraph("3. TACTICAL RECOMMENDATIONS & PROTOCOLS", section_heading))
    recs = [
        "• <b>Ground Truth Validation:</b> Dispatch high-resolution drone / UAV or aerial reconnaissance to confirm pavement grade.",
        "• <b>GIS Integration:</b> Synchronize the exported GeoJSON vectors into national geospatial road registry databases.",
        "• <b>Continuous Temporal Cadence:</b> Schedule 72-hour automated revisits to monitor lateral road spurs and secondary branching."
    ]
    for r in recs:
        elements.append(Paragraph(r, body_style))
        elements.append(Spacer(1, 3))

    doc.build(elements)
    return output_pdf_path
