"""
Script to generate executive PDF and DOCX reports for the Freight Rate ML Assessment.
Embeds candidate_december.png, validation metrics, tables, and methodology details.
"""

from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


def set_cell_background(cell, fill_color):
    """Sets background color for a docx table cell."""
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_color}"/>')
    tcPr.append(shd)


def generate_docx():
    doc = docx.Document()
    
    # Page setup - 0.75 in margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    # Styles
    primary_color = RGBColor(6, 74, 86)     # #064A56
    secondary_color = RGBColor(34, 112, 126) # #22707E
    dark_text = RGBColor(33, 37, 41)
    
    # Title
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(4)
    run_title = p_title.add_run("Freight Rate Machine Learning Assessment")
    run_title.font.name = "Segoe UI"
    run_title.font.size = Pt(22)
    run_title.font.bold = True
    run_title.font.color.rgb = primary_color
    
    # Subtitle
    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_before = Pt(0)
    p_sub.paragraph_format.space_after = Pt(14)
    run_sub = p_sub.add_run("Dynamic Load Pricing Model, Validation Strategy & December Forecast Report")
    run_sub.font.name = "Segoe UI"
    run_sub.font.size = Pt(12)
    run_sub.font.italic = True
    run_sub.font.color.rgb = secondary_color
    
    # Metadata bar
    meta_table = doc.add_table(rows=1, cols=3)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_table.autofit = False
    
    headers = [
        ("Solution", "ML Pipeline (train_predict.py)"),
        ("Validation Framework", "Out-of-Time (OOT) Split"),
        ("Target Variable", "posted_rate ($)")
    ]
    for i, (k, v) in enumerate(headers):
        cell = meta_table.cell(0, i)
        cell.width = Inches(2.3)
        set_cell_background(cell, "F0F4F6")
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(4)
        r1 = p.add_run(f"{k}\n")
        r1.font.name = "Segoe UI"
        r1.font.size = Pt(8.5)
        r1.font.bold = True
        r1.font.color.rgb = primary_color
        r2 = p.add_run(v)
        r2.font.name = "Segoe UI"
        r2.font.size = Pt(9.5)
        r2.font.color.rgb = dark_text

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    def add_heading(text, level=1):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(12)
        h.paragraph_format.space_after = Pt(4)
        h.paragraph_format.keep_with_next = True
        run = h.add_run(text)
        run.font.name = "Segoe UI"
        run.font.bold = True
        if level == 1:
            run.font.size = Pt(14)
            run.font.color.rgb = primary_color
        else:
            run.font.size = Pt(11.5)
            run.font.color.rgb = secondary_color
        return h

    def add_p(text, bold_prefix=None):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.15
        if bold_prefix:
            r_pre = p.add_run(bold_prefix)
            r_pre.font.name = "Segoe UI"
            r_pre.font.size = Pt(10)
            r_pre.font.bold = True
            r_pre.font.color.rgb = dark_text
        r = p.add_run(text)
        r.font.name = "Segoe UI"
        r.font.size = Pt(10)
        r.font.color.rgb = dark_text
        return p

    # 1. Executive Summary
    add_heading("1. Executive Summary", 1)
    add_p(
        "This report outlines the machine learning solution developed to predict freight load posted rates (`posted_rate`) "
        "across national trucking corridors. Using 48,000 historical freight loads spanning January 1 to October 31, 2025, "
        "we built a geospatial-temporal pipeline, validated it on an Out-of-Time holdout, and applied it to 12,000 validation loads "
        "(November-December 2025) and a fixed 31-day December scenario. "
        "The OOT holdout yields a MAE of approximately $129, Median Absolute Error of $56, and R² of 0.82. "
        "Both output files pass Spotter's official `score.py` evaluation with zero errors."
    )

    # 2. Validation & Split Approach
    add_heading("2. Validation Strategy & Data Split Approach", 1)
    add_p(
        "A critical pitfall in freight rate modeling is the use of naive random K-Fold cross-validation. In freight logistics, rates exhibit strong "
        "temporal dependencies, macroeconomic cycles, and produce season surges. Random K-Fold shuffling permits future market indices and rate regimes "
        "to leak into historical training records, producing dangerously optimistic CV metrics that collapse in production out-of-time deployment.",
        "The Anti-Leakage Rationale: "
    )
    add_p(
        "To mirror actual operational reality—where models trained on historical records must forecast upcoming loads—we instituted an "
        "Out-of-Time (OOT) temporal validation split: ",
        "Split Design: "
    )
    add_p(
        "• Development Train Window: January 1, 2025 to August 31, 2025 (38,477 loads, 80.2% of data).\n"
        "• Out-of-Time Holdout Window: September 1, 2025 to October 31, 2025 (9,523 loads, 19.8% of data).\n"
        "• Final Target Deployment: November 1, 2025 to December 31, 2025 (12,000 unobserved validation loads)."
    )
    add_p(
        "This temporal boundary rigorously tests model generalization across market regime shifts, seasonality transitions, and unseen corridor conditions."
    )

    # 3. Data Exploration & Quality Findings
    add_heading("3. Data Exploration & Quality Findings", 1)
    add_p("Five data-quality issues were identified during EDA:")

    add_p(
        "292 training records and 145 validation records carry a negative weight value. These are sign-flip recording errors; "
        "the correct value is the absolute value. The fix `abs(weight)` is applied before any imputation.",
        "1. Negative Weights: "
    )
    add_p(
        "0.78% of training records (373 rows: 270 with rpm > 6.0, 103 with rpm < 0.5) have implausible rate-per-mile values. "
        "The high-rpm subset may include genuine short-haul premiums; the low-rpm subset cannot be explained by any normal freight scenario and may be "
        "test or mis-keyed records. Both are flagged in code for future audit; the model sees their actual posted_rate targets.",
        "2. RPM Outliers (possible corruptions): "
    )
    add_p(
        "300 training loads (0.63%) and 165 validation loads (1.38%) lacked payload weight. Missing values are imputed with a "
        "domain-aligned median (31,000 lbs) and an explicit boolean indicator `weight_isna` is added.",
        "3. Missing Weight Values: "
    )
    add_p(
        "374 training loads (0.78%) and 249 validation loads (2.08%) lacked `market_index`. Because market index is a daily "
        "indicator with tight intra-day variance (~0.024 std), missing days are filled via a daily calendar mean lookup. "
        "A `market_index_isna` flag is included.",
        "4. Missing Market Index: "
    )
    add_p(
        "Validation data introduces 8 cities not present in training: Laredo, Charlotte, Knoxville, Jackson, Norfolk, Chicago, "
        "Allentown, and San Diego. Rather than using categorical city IDs, the model uses Haversine coordinates, tortuosity, and "
        "coordinate deltas, which generalise to any lat/lon pair.",
        "5. Unseen Cities in Validation: "
    )

    # 4. Feature Engineering
    add_heading("4. Feature Engineering", 1)
    add_p(
        "The pipeline creates 29 predictive features across four domains. Five date-level features were deliberately excluded: "
        "`month`, `quarter`, `dayofyear`, `sin_dayofyear`, `cos_dayofyear`. The model is trained on January-October only; "
        "those features take values in November-December that the model never saw, so any splits it learned on them would not generalise."
    )
    add_p(
        "• Temporal (within-week, always in range): `day`, `dayofweek`, `is_weekend`, `is_month_end`, `sin_dayofweek`, `cos_dayofweek`.\n"
        "• Geospatial: `pickup_lat/lon`, `delivery_lat/lon`, Haversine, tortuosity, `lat_diff`, `lon_diff`, midpoint coordinates.\n"
        "• Payload & Equipment: `weight_filled`, `weight_isna`, `weight_tier`, `ton_miles`, `distance`, one-hot equipment flags.\n"
        "• Market / broker signals: `quote_signal` (corr with rpm ~0.05, treated as noisy auxiliary), `est_base`, `market_index_filled`, `market_index_isna`, `market_adj_base`, `quote_x_market`."
    )

    # 5. Model Architecture & Benchmark Comparison
    add_heading("5. Model Architecture & Loss Formulation", 1)
    add_p(
        "In freight transportation, pricing distributions exhibit positive skew and sporadic rate spikes (rush spot shipments, off-hour loads). "
        "Standard L2 (MSE) loss functions are heavily penalized by high-leverage outliers, pulling predictions upward and worsening median errors. "
        "By reformulating the problem as Rate Per Mile (RPM = `posted_rate / distance`) and optimizing with L1 (Mean Absolute Error) regression loss, "
        "the model targets the conditional median rate, dramatically reducing error.",
        "Why Rate Per Mile + L1 Loss? "
    )
    
    # Table of benchmark results
    table = doc.add_table(rows=6, cols=5)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    t_headers = ["Model & Formulation", "Loss Objective", "RMSE ($)", "MAE ($)", "R² Score"]
    for j, h in enumerate(t_headers):
        cell = table.cell(0, j)
        set_cell_background(cell, "064A56")
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.name = "Segoe UI"
        r.font.size = Pt(9)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)

    data = [
        ("Naive baseline (median rpm x dist)", "—", "~$684", "~$257", "~0.80"),
        ("XGBoost Direct", "L2 (MSE)", "$758.01", "$207.05", "0.7533"),
        ("XGBoost Rate Per Mile", "L2 (MSE)", "$710.44", "$186.24", "0.7833"),
        ("LightGBM Direct", "L2 (MSE)", "$656.49", "$173.61", "0.8149"),
        ("LightGBM Rate Per Mile", "L2 (MSE)", "$649.94", "$163.88", "0.8186"),
        ("Tri-Ensemble (Ours)", "70% L1 RPM + 15% L2 RPM + 15% L1 Dir", "~$639", "~$129", "~0.824")
    ]

    # Resize table to 7 rows (header + 6 data rows)
    table = doc.add_table(rows=7, cols=5)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    t_headers = ["Model & Formulation", "Loss Objective", "RMSE ($)", "MAE ($)", "R² Score"]
    for j, h in enumerate(t_headers):
        cell = table.cell(0, j)
        set_cell_background(cell, "064A56")
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.name = "Segoe UI"
        r.font.size = Pt(9)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)

    for i, row_data in enumerate(data):
        for j, val in enumerate(row_data):
            cell = table.cell(i + 1, j)
            bg = "EAF2F4" if i == 5 else ("FFFFFF" if i % 2 == 0 else "F9FBFB")
            set_cell_background(cell, bg)
            p = cell.paragraphs[0]
            if j >= 2:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            r = p.add_run(val)
            r.font.name = "Segoe UI"
            r.font.size = Pt(8.5)
            if i == 5:
                r.font.bold = True
                r.font.color.rgb = primary_color
            else:
                r.font.color.rgb = dark_text

    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    add_p(
        "The Tri-Ensemble (70% L1 RPM + 15% L2 RPM + 15% L1 Direct) reduces MAE by ~$128 and Median AE by ~$83 "
        "vs the naive baseline on the Sep-Oct holdout. OOT MAPE is approximately 5.4%."
    )

    # 6. Fixed December Lane Forecast
    add_heading("6. Fixed December 2025 Forecast & Chart", 1)
    add_p(
        "Scenario: Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lbs, one prediction per day across December 1-31, 2025.",
        "Scenario: "
    )
    add_p(
        "The chart below (Figure 1) is generated by `score.py`. The model captures the weekly day-of-week rhythm observed "
        "throughout the training data: Mondays average ~$811, Wednesdays ~$825, weekends a few dollars lower. "
        "This pattern repeats weekly throughout December. "
        "There is no meaningful late-December surge: day >= 25 averages $783-$812, essentially the same as the rest of the month. "
        "The absolute rate level (~$795) should be treated with caution; the model has no December training data, so the true "
        "market level for this corridor in December 2025 is not reliably estimated."
    )

    # Insert Image
    img_path = Path("scorer_results/candidate_december.png")
    if img_path.is_file():
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(6)
        p_img.paragraph_format.space_after = Pt(6)
        run_img = p_img.add_run()
        run_img.add_picture(str(img_path), width=Inches(6.2))
        
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_after = Pt(8)
        r_cap = p_cap.add_run("Figure 1: Fixed December 2025 Predicted Load Rates (Produced by score.py)")
        r_cap.font.name = "Segoe UI"
        r_cap.font.size = Pt(8.5)
        r_cap.font.italic = True
        r_cap.font.color.rgb = secondary_color

    # 7. Limitations
    add_heading("7. Limitations", 1)
    add_p(
        "• December absolute rate level is uncertain. The model is trained on Jan-Oct data; whether the true Dec 2025 market level is "
        "higher or lower than the predicted ~$795-$825 range cannot be verified without actuals.\n"
        "• Single temporal split. A single OOT cut-point has higher variance than a rolling-origin CV. The metrics should be treated as indicative.\n"
        "• ~0.78% corrupted labels. Records with rpm outside [0.5, 6.0] inflate RMSE and affect R². A data audit is warranted.\n"
        "• `quote_signal` contribution is limited (corr ~0.05 with rpm); it functions as a noisy auxiliary signal.\n"
        "• Negative weights corrected with abs(weight) assuming sign-flip errors; if some are sentinel values, the imputation may need revisiting."
    )

    # 8. Verification & Deliverables Summary
    add_heading("8. Verification & Submission Deliverables", 1)
    add_p(
        "• `validation_predictions.csv`: 12,000 rows, columns `load_id,predicted_rate`, all values positive.\n"
        "• `data/december_chart_inputs.csv`: 31 rows with `predicted_rate` filled (original input CSV never overwritten).\n"
        "• `score.py` verification: 0 errors (`Validated 12,000 final predictions. Validated 31 fixed December predictions`).\n"
        "• Reproducibility: `python train_predict.py` retrains and regenerates all output files from scratch."
    )

    docx_path = Path("Freight_Rate_ML_Assessment_Report.docx")
    doc.save(docx_path)
    print(f"Successfully generated DOCX report at: {docx_path.resolve()}")
    return docx_path


def generate_pdf():
    pdf_path = Path("Freight_Rate_ML_Assessment_Report.pdf")
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Palette
    c_primary = colors.HexColor("#064A56")
    c_secondary = colors.HexColor("#22707E")
    c_dark = colors.HexColor("#212529")
    c_light_bg = colors.HexColor("#F0F4F6")
    c_table_alt = colors.HexColor("#F9FBFB")
    c_table_highlight = colors.HexColor("#EAF2F4")
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=c_primary,
        spaceAfter=4
    )
    
    sub_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=11,
        leading=14,
        textColor=c_secondary,
        spaceAfter=12
    )

    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=c_primary,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.2,
        leading=12.5,
        textColor=c_dark,
        spaceAfter=5
    )

    caption_style = ParagraphStyle(
        'Caption',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=10,
        textColor=c_secondary,
        alignment=1, # Center
        spaceAfter=6
    )

    story = []
    
    # Title & Subtitle
    story.append(Paragraph("Freight Rate Machine Learning Assessment", title_style))
    story.append(Paragraph("Dynamic Load Pricing Model, Validation Strategy & December Forecast Report", sub_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_primary, spaceAfter=8))
    
    # Metadata summary banner
    meta_data = [
        [
            Paragraph("<b>Candidate Solution:</b> Production ML Pipeline", body_style),
            Paragraph("<b>Validation:</b> Out-of-Time (OOT) Split", body_style),
            Paragraph("<b>Target Variable:</b> posted_rate ($)", body_style)
        ]
    ]
    t_meta = Table(meta_data, colWidths=[180, 180, 172])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_light_bg),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 5),
        ('BOX', (0,0), (-1,-1), 0.5, c_secondary)
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 8))

    # 1. Executive Summary
    story.append(Paragraph("1. Executive Summary", h1_style))
    story.append(Paragraph(
        "This report details the machine learning solution developed to forecast freight posted rates (`posted_rate`) "
        "across US truckload corridors. Using 48,000 historical freight loads (Jan 01 to Oct 31, 2025), "
        "we built a geospatial-temporal pipeline and generated predictions for 12,000 validation loads "
        "(Nov 01 to Dec 31, 2025) and a fixed 31-day December scenario. The OOT holdout yields MAE of ~$129, "
        "Median AE ~$56, and R² ~0.82. Both output files pass Spotter's official <code>score.py</code> with zero errors.",
        body_style
    ))

    # 2. Validation & Split Approach
    story.append(Paragraph("2. Validation Strategy & Data Split Approach", h1_style))
    story.append(Paragraph(
        "<b>The Anti-Leakage Rationale:</b> Standard random K-Fold cross-validation introduces severe temporal look-ahead leakage into "
        "freight market modeling. Freight rates undergo seasonal produce surges, macroeconomic tightening, and holiday rate adjustments. "
        "Randomly shuffling records enables a model to train on late-October freight rates to predict April loads, providing an artificially inflated "
        "CV score that collapses when deployed to upcoming months in production.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Out-of-Time Split Architecture:</b> To mimic production conditions, we established a strict chronological boundary:<br/>"
        "• <b>Development Training Window:</b> January 1, 2025 to August 31, 2025 (38,477 loads, 80.2% of data).<br/>"
        "• <b>Out-of-Time Validation Window:</b> September 1, 2025 to October 31, 2025 (9,523 loads, 19.8% of data).<br/>"
        "• <b>Final Unseen Test Set:</b> November 1, 2025 to December 31, 2025 (12,000 validation loads).",
        body_style
    ))

    # 3. Data Exploration & Quality Remediation
    story.append(Paragraph("3. Data Exploration & Quality Findings", h1_style))
    story.append(Paragraph(
        "• <b>Negative Weights:</b> 292 train / 145 val records have negative weight — sign-flip recording errors. Fixed with <code>abs(weight)</code>.<br/>"
        "• <b>RPM Outliers:</b> 373 training records (0.78%) have rpm outside [0.5, 6.0]. Low-rpm rows cannot be explained by any normal "
        "freight scenario; high-rpm rows include some genuine short-haul premiums. Both flagged for audit; model sees actual targets.<br/>"
        "• <b>Missing Weights:</b> 300 train (0.63%) / 165 val (1.38%) lack weight. Imputed with 31,000 lbs median + <code>weight_isna</code> flag.<br/>"
        "• <b>Missing Market Index:</b> 374 train (0.78%) / 249 val (2.08%) lack <code>market_index</code>. Filled via daily calendar mean + <code>market_index_isna</code> flag.<br/>"
        "• <b>Unseen Cities:</b> 8 new cities in validation (Laredo, Charlotte, Knoxville, Jackson, Norfolk, Chicago, Allentown, San Diego). "
        "The model uses Haversine coordinates rather than city IDs, so it generalises to any lat/lon pair.",
        body_style
    ))

    # 4. Model Architecture & Benchmarks
    story.append(Paragraph("4. Model Architecture & Loss Formulation", h1_style))
    story.append(Paragraph(
        "Freight spot rates are right-skewed with ~0.78% extreme outliers. L2 (MSE) loss is pulled by those outliers. "
        "Reformulating the target as <b>Rate Per Mile (RPM = posted_rate / distance)</b> and using <b>L1 (MAE) loss</b> "
        "targets the conditional median, substantially reducing error. A naive baseline (median training rpm × distance) "
        "serves as the benchmark floor.",
        body_style
    ))

    # Benchmark Table
    bench_data = [
        ["Model & Formulation", "Objective", "RMSE ($)", "MAE ($)", "R² Score"],
        ["Naive baseline (median rpm x dist)", "—", "~$684", "~$257", "~0.80"],
        ["XGBoost Direct", "L2 (MSE)", "$758.01", "$207.05", "0.7533"],
        ["XGBoost Rate Per Mile", "L2 (MSE)", "$710.44", "$186.24", "0.7833"],
        ["LightGBM Direct", "L2 (MSE)", "$656.49", "$173.61", "0.8149"],
        ["LightGBM Rate Per Mile", "L2 (MSE)", "$649.94", "$163.88", "0.8186"],
        ["Tri-Ensemble (Ours)", "70% L1 RPM + 15% L2 RPM + 15% L1 Dir", "~$639", "~$129", "~0.824"]
    ]
    t_bench = Table(bench_data, colWidths=[170, 142, 72, 72, 66])
    t_bench.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 8.5),
        ('ALIGN', (0,0), (-1,0), 'CENTER'),
        ('ALIGN', (2,1), (-1,-1), 'RIGHT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BACKGROUND', (0,1), (-1,1), colors.white),
        ('BACKGROUND', (0,2), (-1,2), c_table_alt),
        ('BACKGROUND', (0,3), (-1,3), colors.white),
        ('BACKGROUND', (0,4), (-1,4), c_table_alt),
        ('BACKGROUND', (0,5), (-1,5), c_table_alt),
        ('BACKGROUND', (0,6), (-1,6), c_table_highlight),
        ('FONTNAME', (0,6), (-1,6), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0,6), (-1,6), c_primary),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#D9E2E4")),
        ('PADDING', (0,0), (-1,-1), 3.5)
    ]))
    story.append(t_bench)
    story.append(Spacer(1, 6))

    # 5. Fixed December Forecast
    story.append(Paragraph("5. Fixed December 2025 Forecast & Verification", h1_style))
    story.append(Paragraph(
        "<b>Scenario:</b> Lexington to Fort Wayne | 360 miles | Dry Van | 32,000 lbs | one prediction per day, Dec 1-31, 2025.<br/>"
        "Figure 1 is generated by <code>score.py</code>. The model captures the weekly day-of-week rhythm: Mon ~$811, Wed ~$825, weekends lower. "
        "This pattern repeats throughout December; there is no meaningful late-December surge (day >=25 average is essentially flat vs rest of month). "
        "The absolute rate level should be treated with caution — the model has no December training data.",
        body_style
    ))

    # Embed Image
    img_path = Path("scorer_results/candidate_december.png")
    if img_path.is_file():
        story.append(RLImage(str(img_path), width=510, height=170))
        story.append(Paragraph("Figure 1: December 2025 Predicted Load Rate Chart (Produced by score.py)", caption_style))

    story.append(Paragraph(
        "<b>Submission Verification:</b> Running <code>python score.py --predictions validation_predictions.csv "
        "--december-predictions data/december_chart_inputs.csv</code> "
        "confirms 12,000 validation predictions and 31 December inputs pass with zero errors.",
        body_style
    ))

    story.append(Paragraph("6. Limitations", h1_style))
    story.append(Paragraph(
        "• <b>December level uncertain:</b> The true Dec 2025 market rate cannot be verified without Dec actuals.<br/>"
        "• <b>Single temporal split:</b> Higher variance than rolling-origin CV; metrics are indicative.<br/>"
        "• <b>~0.78% corrupted labels:</b> Inflate RMSE and affect R²; a data audit is warranted.<br/>"
        "• <b>quote_signal:</b> Corr with rpm ~0.05 — treated as noisy auxiliary, not a rate estimate.<br/>"
        "• <b>Negative weights:</b> Corrected with abs(); if some are sentinel values, imputation should be revisited.",
        body_style
    ))

    doc.build(story)
    print(f"Successfully generated PDF report at: {pdf_path.resolve()}")
    return pdf_path


if __name__ == "__main__":
    generate_docx()
    generate_pdf()
