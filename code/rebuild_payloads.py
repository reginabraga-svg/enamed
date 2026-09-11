import os, json
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.metrics import confusion_matrix

RAW = None

# ---- labels & sentiment (from the paper tables + llm_integration.md) ----
LABELS = {
    "I1_C": "Difficulty: Medium", "I1_D": "Exam considered hard", "I1_B": "Difficulty: Easy",
    "I1_.": "Difficulty: no answer",
    "I7_D": "Learned many contents", "I7_B": "Studied some, didn't learn",
    "I7_C": "Studied most, didn't learn", "I7_A": "Had not studied most yet",
    "I7_E": "Learned all contents",
    "I6_A": "Lack of content knowledge", "I6_B": "Different content framing",
    "I9_A": "Practical activities helped", "I9_B": "Practical activities: No",
    "I5_D": "Instructions: only some", "I5_C": "Sufficient instructions",
    "I4_B": "Clear question stems", "I4_D": "Few clear prompts",
    "I3_C": "Adequate exam length", "I3_.": "Exam length: no answer",
    "I2_.": "Time to finish: no answer",
}
FRICTION = {"I6_A", "I1_D"}
STRENGTH = {"I7_D", "I9_A", "I4_B"}


def label_for(code):    # code like "I1_C"
    return LABELS.get(code, code.replace("_", " "))


def sentiment_for(code):
    return "friction" if code in FRICTION else ("strength" if code in STRENGTH else "neutral")


def build():
    df = pd.read_csv(RAW, sep=";", decimal=".", low_memory=False)
    pres = df[df["TP_PRES"] == 555].copy()
    pres["NT_GER"] = pd.to_numeric(pres["NT_GER"], errors="coerce")

    perf = pres.groupby("CO_CURSO")["NT_GER"].mean().reset_index()
    perf.columns = ["CO_CURSO", "avg_score_general"]
    perf = perf.dropna(subset=["avg_score_general"]).reset_index(drop=True)  # -> 350
    median_score = perf["avg_score_general"].median()
    perf["performance_class"] = np.where(perf["avg_score_general"] >= median_score, 1, 0)

    perception_cols = [c for c in pres.columns if "CO_RS_" in c]
    feats = pd.DataFrame({"CO_CURSO": perf["CO_CURSO"]})
    for col in perception_cols:
        ct = pd.crosstab(pres["CO_CURSO"], pres[col], normalize="index") * 100
        ct.columns = [f"pct_{col}_{r}" for r in ct.columns]
        feats = feats.merge(ct.reset_index(), on="CO_CURSO", how="left")

    dfm = perf.merge(feats, on="CO_CURSO", how="inner")
    dfm = dfm.fillna(dfm.median(numeric_only=True))
    feat_all = [c for c in dfm.columns if c.startswith("pct_")]
    X_all, y = dfm[feat_all], dfm["performance_class"]

    sel = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=5).fit(X_all, y)
    imp_rank = pd.DataFrame({"f": feat_all, "imp": sel.feature_importances_}) \
        .sort_values("imp", ascending=False)
    selected = imp_rank.head(20)["f"].tolist()
    X = dfm[selected]

    best = RandomForestClassifier(random_state=42, n_estimators=50, max_depth=6, criterion="entropy")
    best.fit(X, y)
    importances = best.feature_importances_
    return dfm, X, y, selected, importances, best, median_score


def payloads(dfm, X, selected, importances, best, median_score, topk=7):
    codes = [s.replace("pct_CO_RS_", "") for s in selected]
    preds = best.predict(X)
    probs = best.predict_proba(X)
    out = []
    for i, (_, row) in enumerate(dfm.iterrows()):
        vals = row[selected].values.astype(float)
        lc = vals * importances
        order = np.argsort(lc)[::-1][:topk]
        contribs = []
        for j in order:
            code = codes[j]
            contribs.append({
                "feature": code,
                "label": label_for(code),
                "value_pct": round(float(vals[j]), 1),
                "impact": round(float(lc[j]), 2),
                "sentiment": sentiment_for(code),
            })
        out.append({
            "course_code": int(row["CO_CURSO"]),
            "avg_score": round(float(row["avg_score_general"]), 2),
            "national_median": round(float(median_score), 2),
            "predicted_tier": "High Performance" if preds[i] == 1 else "Low Performance",
            "confidence_pct": round(float(probs[i][preds[i]] * 100), 1),
            "contributions": contribs,
            "language": "English",
        })
    return out


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    ap=argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    a=ap.parse_args(); RAW=a.data
    dfm,X,y,selected,importances,best,median_score=build()
    result=payloads(dfm,X,selected,importances,best,median_score)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print('Wrote',len(result),'real-data payloads')
