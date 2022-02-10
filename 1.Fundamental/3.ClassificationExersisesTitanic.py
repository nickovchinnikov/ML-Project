# %%
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# %% [markdown]
"""
## 3. Tackle the Titanic dataset. A great place to start is on Kaggle.
"""
# %%
TITANIC_PATH = os.path.join("datasets", "titanic")


def load_titanic_data(filename, titanic_path=TITANIC_PATH):
    csv_path = os.path.join(titanic_path, filename)
    return pd.read_csv(csv_path)
# %%
train_data = load_titanic_data("train.csv")
test_data = load_titanic_data("test.csv")
# %% [markdown]
"""
Let's check the train data
"""
# %%
train_data.head()
# %% [markdown]
"""
The attributes have the following meaning:
* **PassengerId**: a unique identifier for each passenger
* **Survived**: that's the target, 0 means the passenger did not survive, while 1 means he/she survived.
* **Pclass**: passenger class.
* **Name**, **Sex**, **Age**: self-explanatory
* **SibSp**: how many siblings & spouses of the passenger aboard the Titanic.
* **Parch**: how many children & parents of the passenger aboard the Titanic.
* **Ticket**: ticket id
* **Fare**: price paid (in pounds)
* **Cabin**: passenger's cabin number
* **Embarked**: where the passenger embarked the Titanic

----

Let's explicitly set the `PassengerId` column as the index column:
"""
# %%
train_data = train_data.set_index("PassengerId")
test_data = test_data.set_index("PassengerId")
# %% [markdown]
"""
Let's get more info to see how much data is missing:
"""
# %%
train_data.info()
# %%
train_data[train_data["Sex"]=="female"]["Age"].median()
# %%
train_data.describe()
# %% [markdown]
"""
* Yikes, only 38% **Survived**! 😭 That's close enough to 40%, so accuracy will be a reasonable metric to evaluate our model.
* The mean **Fare** was £32.20, which does not seem so expensive (but it was probably a lot of money back then).
* The mean **Age** was less than 30 years old.
"""
# %%
"""
Let's check that the target is indeed 0 or 1:
"""
# %%
train_data["Survived"].value_counts()
# %%
train_data["Sex"].value_counts()
# %%
train_data["Pclass"].value_counts()
# %%
train_data["Embarked"].value_counts()
# %% [markdown]
"""
The Embarked attribute tells us where the passenger embarked:
C=Cherbourg, Q=Queenstown, S=Southampton.
"""
# %% [markdown]
"""
### Now let's build our preprocessing pipelines
starting with the pipeline for numerical attributes:
"""
# %%
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
# %% [markdown]
"""
Now we can build the pipeline for the categorical attributes:
"""
# %%
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("cat_encoder", OneHotEncoder(sparse=False)),
])
# %% [markdown]
"""
Finally, let's join the numerical and categorical pipelines:
"""
# %%
num_attribs = ["Age", "SibSp", "Parch", "Fare"]
cat_attribs = ["Pclass", "Sex", "Embarked"]

preprocess_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])
# %% [markdown]
"""
Cool! Now we have a nice preprocessing pipeline that takes the raw data and outputs
numerical input features that we can feed to any Machine Learning model we want.
"""
# %%
X_train = preprocess_pipeline.fit_transform(
    train_data[num_attribs + cat_attribs])
X_train
# %% [markdown]
"""
Let's not forget to get the labels:
"""
# %%
y_train = train_data["Survived"]
# %% [markdown]
"""
We are now ready to train a classifier.
Let's start with a `RandomForestClassifier`:
"""
# %%
forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_clf.fit(X_train, y_train)
# %%
X_test = preprocess_pipeline.transform(test_data[num_attribs + cat_attribs])
y_pred = forest_clf.predict(X_test)
# %%
forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
forest_scores.mean()
# %% [markdown]
"""
Okay, not too bad! Looking at the [leaderboard](https://www.kaggle.com/c/titanic/leaderboard)
for the Titanic competition on Kaggle, you can see that our score is in the top 2%, woohoo!
Some Kagglers reached 100% accuracy, but since you can easily find the
[list of victims](https://www.encyclopedia-titanica.org/titanic-victims/)
of the Titanic, it seems likely that there was little Machine Learning involved
in their performance! 😆
"""
# %% [markdown]
"""
Let's try an `SVC`:
"""
# %%
svm_clf = SVC(gamma="auto")
svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)
svm_scores.mean()
# %%
svm_clf.fit(X_train, y_train)
# %%
y_predict = svm_clf.predict(X_test)
y_predict
# %%
result = test_data.loc[:, []]
result["Survived"] = y_predict
result
# %%
result.to_csv(os.path.join(TITANIC_PATH, 'prediction.csv'))
# %% [markdown]
"""
Great! This model looks better.
"""
# %% [markdown]
"""
But instead of just looking at the mean accuracy across the 10 cross-validation folds,
let's plot all 10 scores for each model, along with a box plot highlighting the lower
and upper quartiles, and "whiskers" showing the extent of the scores (thanks to Nevin
Yilmaz for suggesting this visualization). Note that the `boxplot()` function detects
outliers (called "fliers") and does not include them within the whiskers. Specifically, 
if the lower quartile is $Q_1$ and the upper quartile is $Q_3$, then the interquartile 
range $IQR = Q_3 - Q_1$ (this is the box's height), and any score lower than $Q_1 - 1.5 
\times IQR$ is a flier, and so is any score greater than $Q3 + 1.5 \times IQR$.
"""
# %%
plt.figure(figsize=(8, 4))
plt.plot([1]*10, svm_scores, ".")
plt.plot([2]*10, forest_scores, ".")
plt.boxplot([svm_scores, forest_scores], labels=("SVM","Random Forest"))
plt.ylabel("Accuracy", fontsize=14)
plt.show()

# %%
