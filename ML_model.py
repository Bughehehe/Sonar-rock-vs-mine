import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
from sklearn.svm import NuSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import joblib

## Read data
data = pd.read_csv("sonar-all-data.csv")

## Preprocess data
data["Label"].replace({'R' : 0, 'M' : 1}, inplace=True)

## Edition helpers
# print(data[:10])
# print(data.isnull().sum())
# sns.heatmap(data.corr())
# plt.show()

## Preprocess again
y = data['Label']
X = data.drop(columns=["Label"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 64)

## Search for best model
# clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
# models,predictions = clf.fit(X_train, X_test, y_train, y_test)
# models.to_csv("models.csv")
# predictions.to_csv("predictions.csv")

## Create model
model = make_pipeline(StandardScaler(), 
                      NuSVC())
model.fit(X_train,y_train)
y_test_predicted = model.predict(X_test)

# ## Print metrics
accuracy = metrics.accuracy_score(y_test,y_test_predicted)
confusion_matrix = metrics.confusion_matrix(y_test,y_test_predicted)
print('The accuracy score is ',accuracy*100,'%')
sns.heatmap(confusion_matrix)
plt.show()