"""PolERo baseline: train/validation + final inference. Run: python polero_baseline.py"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from joblib import dump

ROOT = Path(__file__).resolve().parent
train = pd.read_csv(ROOT/'train_data.csv')
test = pd.read_csv(ROOT/'test_data.csv')

def texts(df):
    return ('[PRESIDENT] ' + df.president.astype(str) + ' [CONTEXT] ' + df.context.astype(str) +
            ' [QUESTION] ' + df.interview_question.astype(str) + ' [ANSWER] ' + df.interview_answer.astype(str)).tolist()

MAP = {
    'Explicit':'Clear Reply',
    'Implicit':'Ambivalent', 'General':'Ambivalent', 'Partial/half-answer':'Ambivalent',
    'Dodging':'Ambivalent','Deflection':'Ambivalent',
    'Declining to answer':'Clear Non-Reply','Claims ignorance':'Clear Non-Reply','Clarification':'Clear Non-Reply',
}
assert all(train.evasion_label.map(MAP) == train.clarity_label)

X = texts(train)
y = train.evasion_label.to_numpy()
indices = np.arange(len(train))
a,b = train_test_split(indices, test_size=.2, random_state=42, stratify=y)

def model():
    return Pipeline([
      ('features',FeatureUnion([
        ('word',TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=100000, sublinear_tf=True, strip_accents='unicode', max_df=.98)),
        ('char',TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=3,max_features=130000,sublinear_tf=True, max_df=.99)),
      ])),
      ('classifier',LogisticRegression(max_iter=1400, class_weight='balanced', C=2.0))
    ])

clf = model()
clf.fit([X[i] for i in a], y[a])
pred = clf.predict([X[i] for i in b])
clarity_true = train.clarity_label.to_numpy()[b]
clarity_pred = np.array([MAP[z] for z in pred])
metrics = {
  'validation_size':int(len(b)),
  'validation_evasion_accuracy': float(accuracy_score(y[b],pred)),
  'validation_evasion_macro_f1': float(f1_score(y[b],pred,average='macro')),
  'validation_clarity_accuracy':float(accuracy_score(clarity_true,clarity_pred)),
  'validation_clarity_macro_f1':float(f1_score(clarity_true,clarity_pred,average='macro')),
  'evasion_classification_report':classification_report(y[b],pred,output_dict=True,zero_division=0),
  'train_label_counts':{str(k):int(v) for k,v in train.evasion_label.value_counts().items()},
}
(ROOT/'baseline_metrics.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False),encoding='utf-8')
print('VALIDATION:',{k:v for k,v in metrics.items() if k.startswith('validation_')})
print(classification_report(y[b],pred,zero_division=0))

final = model()
final.fit(X,y)
test_pred = final.predict(texts(test))
submission = pd.concat([
  pd.DataFrame({'subtaskID':1,'datapointID':test.datapointID,'answer':[MAP[p] for p in test_pred]}),
  pd.DataFrame({'subtaskID':2,'datapointID':test.datapointID,'answer':test_pred}),
],ignore_index=True)
assert submission.shape == (592,3)
assert submission.groupby('datapointID').subtaskID.nunique().eq(2).all()
submission.to_csv(ROOT/'polero_submission_baseline.csv',index=False)
dump(final,ROOT/'polero_baseline_model.joblib',compress=3)
print('SAVED:',ROOT/'polero_submission_baseline.csv')
