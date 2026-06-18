from datetime import datetime


def generate_report(
    extracted_text,
    dimensions,
    compliance
):

    report = f"""
AI BUILDING PLAN ANALYSIS REPORT
========================================

Generated On:
{datetime.now().strftime("%d-%m-%Y %H:%M:%S")}

========================================
OCR EXTRACTED TEXT
========================================

{extracted_text}

========================================
DETECTED DIMENSIONS
========================================
"""

    for dim in dimensions:

        report += f"\n- {dim} mm"

    report += f"""

========================================
NBC COMPLIANCE ANALYSIS
========================================

Compliance Score:
{compliance['score']}%

Risk Level:
{compliance['risk']}

========================================
POSITIVES
========================================
"""

    for item in compliance["positives"]:

        report += f"\n+ {item}"

    report += f"""

========================================
ISSUES
========================================
"""

    if compliance["issues"]:

        for item in compliance["issues"]:

            report += f"\n- {item}"

    else:

        report += "\nNo major issues detected."

    report += f"""

========================================
ENGINEERING SUGGESTIONS
========================================
"""

    for item in compliance["suggestions"]:

        report += f"\n* {item}"

    report += """

========================================
PROJECT SUMMARY
========================================

This AI-powered system analyzes building floor plans
using computer vision, OCR, and AI-driven NBC
compliance checking.

The system helps engineers and architects:
- Understand floor plans
- Detect building dimensions
- Analyze NBC compliance
- Identify potential design risks
- Generate engineering recommendations

========================================
END OF REPORT
========================================
"""

    return report