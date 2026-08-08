from __future__ import annotations


def performance_band(score: float) -> str:
    if score >= 16:
        return "Excellent"
    if score >= 14:
        return "Very Good"
    if score >= 10:
        return "On Track"
    if score >= 8:
        return "Needs Improvement"
    return "At Risk"


def build_recommendations(profile: dict, predicted_score: float) -> list[str]:
    tips: list[str] = []
    studytime = int(profile.get("studytime", 2))
    failures = int(profile.get("failures", 0))
    absences = int(profile.get("absences", 0))
    g1 = float(profile.get("G1", 10))
    g2 = float(profile.get("G2", 10))
    internet = profile.get("internet", "yes")
    higher = profile.get("higher", "yes")

    if studytime <= 2:
        tips.append("Increase focused study time gradually to at least 5 hours per week.")
    else:
        tips.append("Keep your study routine consistent and use short revision sessions.")

    if failures > 0:
        tips.append("Review failed topics first and schedule targeted practice before new topics.")

    if absences >= 8:
        tips.append("Reduce avoidable absences because missed lessons can create learning gaps.")

    if g2 < g1:
        tips.append("Your recent grade is falling; use weekly quizzes to identify weak concepts early.")

    if predicted_score < 10:
        tips.append("Prioritize core concepts, teacher feedback, and a weekly recovery plan.")
    elif predicted_score < 14:
        tips.append("Aim for steady improvement by practicing past questions and tracking mistakes.")
    else:
        tips.append("Challenge yourself with advanced problems and peer teaching to maintain progress.")

    if internet == "yes":
        tips.append("Use reliable online practice resources, but keep a fixed study schedule to avoid distraction.")

    if higher == "yes" and predicted_score >= 14:
        tips.append("You appear on a strong academic path; consider building a portfolio alongside your studies.")

    return tips[:6]
