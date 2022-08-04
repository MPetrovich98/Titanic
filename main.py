import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
np.random.seed(0)#da np.random generise isti vektor
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')


dataset = pd.concat([train, test], sort=False).drop(columns='Survived')
print(pd.DataFrame({'No. NaN': dataset.isna().sum(), '%': dataset.isna().sum() / len(dataset)}))



data = [train, test]
for dataset in data:
    #da popunim podatke Age koji nisu uneta
    mean = train["Age"].mean()#srednja vrednost 
    std = test["Age"].std()#standarna devijacija 
    is_null = dataset["Age"].isnull().sum()#broj podatka koji nije unet
    rand_age = np.round(np.random.randint(mean - std, mean + std, size = is_null))#generise nasumicne vrednosti, prva dva argumenta odredjuje opseg generisanja nasumicne vrednosti
    age_slice = dataset["Age"][:]#
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = np.round(age_slice)
    #da popunim podatke Fare koji nisu uneta
    mean = train["Fare"].mean()
    std = test["Fare"].std()
    is_null = dataset["Fare"].isnull().sum()
    rand_age = np.round(np.random.randint(mean - std, mean + std, size = is_null))
    age_slice = dataset["Fare"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Fare"] = age_slice
    
    

# Ako je pol misku upisuje 1 inace 0
train['IsMale'] = pd.get_dummies(train['Sex'], drop_first=True)
test['IsMale'] = pd.get_dummies(test['Sex'], drop_first=True)
#y je podatak da li su preziveli ili nisu
y=train.pop('Survived')

x=train[['Pclass','Age','SibSp','Parch','Fare','IsMale']][:]
x_test=test[['Pclass','Age','SibSp','Parch','Fare','IsMale']][:]

#moj model je klasifikator slučajne šume
model=RandomForestClassifier(n_estimators=100)
model.fit(x,y)#obučavanje modela
print(("Tačnost modela :",accuracy_score(y,model.predict(x))))
#pedviđanje na nepoznatim podacima
y_pred=model.predict(x_test)

submission=pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived":y_pred
})#kreiramo submission freim podatka i cuvamo podatke u submission.csv
submission.to_csv('submission.csv',index=False)
