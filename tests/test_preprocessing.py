import pandas as pd

def test_risk_rule_defasagem():
    df = pd.DataFrame({"DEFASAGEM_2021": [-1.0, 0.0, 2.0, -3.0]})
    risk = (df["DEFASAGEM_2021"] < 0).astype(int).tolist()
    assert risk == [1, 0, 0, 1]
