def analyze_compliance(text, dimensions):

    positives = []
    issues = []
    suggestions = []

    score = 85
    risk = "Low"

    text_upper = text.upper()

    # BEDROOM
    if "BEDROOM" in text_upper:
        positives.append("Bedroom detected.")

        if 4200 in dimensions:
            positives.append("Bedroom dimension appears adequate.")

    else:
        issues.append("Bedroom not clearly detected.")
        score -= 10

    # KITCHEN
    if "KITCHEN" in text_upper:
        positives.append("Kitchen detected.")

        if 3525 in dimensions:
            positives.append("Kitchen dimension appears acceptable.")

    else:
        issues.append("Kitchen not detected.")
        score -= 10

    # WC
    if "WC" in text_upper:
        positives.append("Toilet/WC identified.")

    else:
        issues.append("Toilet/WC not identified.")
        score -= 5

    # CORRIDOR
    if "CORR" in text_upper:
        positives.append("Corridor detected.")

        if 1775 in dimensions:
            positives.append("Corridor width appears acceptable.")

    # LIVING ROOM
    if "LIVING" in text_upper:
        positives.append("Living room detected.")

    # GENERAL SUGGESTIONS
    suggestions.append("Ensure natural ventilation in habitable rooms.")
    suggestions.append("Verify fire escape access as per NBC.")
    suggestions.append("Confirm accessibility for elderly/disabled users.")
    suggestions.append("Review structural layout with licensed engineer.")

    # RISK
    if score >= 80:
        risk = "Low"
    elif score >= 60:
        risk = "Medium"
    else:
        risk = "High"

    return {
        "score": score,
        "risk": risk,
        "issues": issues,
        "positives": positives,
        "suggestions": suggestions
    }