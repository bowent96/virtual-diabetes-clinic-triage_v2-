from sklearn.datasets import load_diabetes

def load_diabetes_df():
    Xy = load_diabetes(as_frame=True)
    df = Xy.frame
    # Ensure feature names are human-friendly to match API spec
    # scikit-learn uses these keys already: age, sex, bmi, bp, s1..s6
    return df.drop(columns=["target"]), df["target"]
