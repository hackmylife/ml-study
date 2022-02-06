import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# load data
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

train_x = train.drop(['Survived'], axis=1)
train_y = train['Survived']

test_x = test.copy()

# drop column
train_x = train_x.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
test_x = test_x.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# transform
for c in ['Sex', 'Embarked']:
    le = LabelEncoder()
    le.fit(train_x[c].fillna('NA'))

    train_x[c] = le.transform(train_x[c].fillna('NA'))
    test_x[c] = le.transform(test_x[c].fillna('NA'))

# modle
model = XGBClassifier(n_estimators=20, random_state=71)
model.fit(train_x, train_y)

# predict
pred = model.predict_proba(test_x)[:, 1]

# encode result
pred_label = np.where(pred > 0.5, 1, 0)

# export file
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred_label})
submission.to_csv('first_result.csv', index=False)
