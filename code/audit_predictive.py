"""Recompute course-level evaluation with feature selection inside each fold.
Usage: python audit_predictive.py --data PATH --outdir PATH
Fixed model settings are inherited from the supplied artifact, not retuned here.
"""
import argparse, json, hashlib, platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, brier_score_loss

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--outdir',required=True);a=ap.parse_args()
    out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(a.data,sep=';',low_memory=False)
    pres=df[df.TP_PRES==555].copy();pres['NT_GER']=pd.to_numeric(pres.NT_GER,errors='coerce')
    perf=pres.groupby('CO_CURSO').NT_GER.mean().dropna();median=float(perf.median());y=(perf>=median).astype(int).to_numpy()
    cols=[c for c in pres.columns if c.startswith('CO_RS_')]
    X=pd.concat([pd.crosstab(pres.CO_CURSO,pres[c],normalize='index').mul(100).add_prefix(c+'_') for c in cols],axis=1).reindex(perf.index)
    def rf():return RandomForestClassifier(n_estimators=50,max_depth=6,criterion='entropy',random_state=42)
    factories={'Majority class':lambda:DummyClassifier(strategy='most_frequent'),'Decision Tree':lambda:DecisionTreeClassifier(max_depth=6,random_state=42),'Gaussian Naive Bayes':GaussianNB,'Logistic Regression':lambda:LogisticRegression(max_iter=10000),'Random Forest':rf}
    metrics={n:[] for n in factories};oof={n:np.zeros(len(y),dtype=int) for n in factories};probs=np.zeros(len(y));ks=[5,10,15,20,30,40,60];curve={k:[] for k in ks};folds=[]
    for fold,(tr,te) in enumerate(KFold(5,shuffle=True,random_state=42).split(X),1):
        im=SimpleImputer(strategy='median');xt=im.fit_transform(X.iloc[tr]);xv=im.transform(X.iloc[te])
        rank=RandomForestClassifier(n_estimators=100,max_depth=5,random_state=42).fit(xt,y[tr]);order=np.argsort(-rank.feature_importances_,kind='stable');sel=order[:20]
        folds.append({'fold':fold,'train_n':len(tr),'test_n':len(te),'selected_features':X.columns[sel].tolist(),'test_positions':te.tolist()})
        for name,f in factories.items():
            model=f().fit(xt[:,sel],y[tr]);pred=model.predict(xv[:,sel]);oof[name][te]=pred
            metrics[name].append({'accuracy':accuracy_score(y[te],pred),'macro_f1':f1_score(y[te],pred,average='macro')})
            if name=='Random Forest':probs[te]=model.predict_proba(xv[:,sel])[:,1]
        for k in ks:curve[k].append(accuracy_score(y[te],rf().fit(xt[:,order[:k]],y[tr]).predict(xv[:,order[:k]])))
        print('Completed fold',fold,flush=True)
    models={n:{'accuracy_mean':float(np.mean([v['accuracy'] for v in vs])),'accuracy_sd':float(np.std([v['accuracy'] for v in vs],ddof=1)),'macro_f1_mean':float(np.mean([v['macro_f1'] for v in vs])),'macro_f1_sd':float(np.std([v['macro_f1'] for v in vs],ddof=1)),'oof_confusion_matrix':confusion_matrix(y,oof[n]).tolist(),'folds':vs} for n,vs in metrics.items()}
    errors=oof['Random Forest']!=y;dist=np.abs(perf.to_numpy()-median);near=dist<=2
    boundary={'band_points':2,'near_n':int(near.sum()),'near_errors':int(errors[near].sum()),'far_n':int((~near).sum()),'far_errors':int(errors[~near].sum()),'total_errors':int(errors.sum()),'mean_confidence_correct':float(np.maximum(probs,1-probs)[~errors].mean()),'mean_confidence_error':float(np.maximum(probs,1-probs)[errors].mean()),'brier':float(brier_score_loss(y,probs))}
    result={'raw_sha256':hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),'python':platform.python_version(),'sklearn':sklearn.__version__,'numpy':np.__version__,'pandas':pd.__version__,'n_present':len(pres),'n_missing_score':int(pres.NT_GER.isna().sum()),'n_courses':len(y),'n_features':X.shape[1],'median':median,'class_counts':np.bincount(y).tolist(),'protocol':'5-fold KFold shuffle=True random_state=42; training-fold median imputation and feature ranking; inherited fixed model settings; no retuning','models':models,'boundary':boundary,'curve':{str(k):{'mean':float(np.mean(v)),'sd':float(np.std(v,ddof=1))} for k,v in curve.items()},'folds':folds}
    (out/'predictive_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    pd.DataFrame({'program_index':np.arange(len(y)),'score':perf.to_numpy(),'true_class':y,'rf_oof_prediction':oof['Random Forest'],'rf_oof_probability_high':probs,'distance_from_median':dist}).to_csv(out/'oof_predictions.csv',index=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ['folds','models']},indent=2));print(json.dumps(models,indent=2))
if __name__=='__main__':main()
