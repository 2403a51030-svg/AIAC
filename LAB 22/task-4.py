def detect_heart_anomaly(age, cholesterol, blood_pressure, heart_rate):
    # Simple anomaly detection logic based on thresholds
    if cholesterol > 240:
        return "High cholesterol level detected."
    if blood_pressure > 140:
        return "High blood pressure detected."
    if heart_rate < 60 or heart_rate > 100:
        return "Abnormal heart rate detected."
    return "No anomalies detected."

def main():
    print("Enter the following details to check for heart anomalies:")
    age = int(input("Age: "))
    cholesterol = int(input("Cholesterol level (mg/dL): "))
    blood_pressure = int(input("Blood pressure (mm Hg): "))
    heart_rate = int(input("Heart rate (bpm): "))

    result = detect_heart_anomaly(age, cholesterol, blood_pressure, heart_rate)
    print(result)

if __name__ == "__main__":
    main()