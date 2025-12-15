def calculate_demographic_risk(age: int, gender: str):
    """
    Calculates a risk score (0-1) based on age and gender.
    """
    risk = 0.0
    
    # Age-based risk
    if age < 5 or age > 60:
        risk += 0.4
    elif age > 40:
        risk += 0.2
        
    # Gender-based risk (simplified example)
    # Certain diseases might have gender bias, this is a generic baseline
    if gender.lower() == 'male':
        risk += 0.05
    
    return min(risk, 1.0)
