# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# %%
data = load_breast_cancer()

x = pd.DataFrame(
    data.data,
    columns=data.feature_names)
y = pd.Series((data.target == 0).astype(int),name="malignant")
print(y.value_counts())
print("feature matrix shape",x.shape)
print("target shape",y.shape)
print("class names",data.target_names)






# %%
class_counts = y.value_counts().sort_index()
class_distribution = pd.DataFrame({
    "Class":data.target_names,
    "count":class_counts.values,
    "Probability": class_counts.values / len(y)
})
print(class_distribution)

# %%
class_distribution.plot(
    x="Class",  # <-- Make sure this matches your exact column name!
    y="count",
    kind="bar",
    legend=False,
    color=["tomato", "steelblue"]
)

plt.ylabel("number of observations")
plt.title("class distribution")
plt.xticks(rotation=0)
plt.show()


# %%
x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,test_size=0.20,
    random_state=42,
    stratify=y

)
print("training size",len(y_train))
print("testing size",len(y_test))
print("\ntraining proportions")
print(y_train.value_counts(normalize=True).sort_index())
print("\ntesting proportions")
print(y_test.value_counts(normalize=True).sort_index())


# %%
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)
model.fit(x_train, y_train)

# %%
probabilities= model.predict_proba(x_test)
print(probabilities[:5])

# %%
results = pd.DataFrame({
    "Actual_class": y_test.values,  
    "P_malignant" : probabilities[:, 0],
    "P_benign" :probabilities[:, 1]
    })
print(results.head(10))

# %%
results["Actual_label"] = results["Actual_class"].map({ 
                            0: "Malignant" ,
                            1:"Benign"
})
print(
    results[
        ["Actual_label","P_malignant", "P_benign"]
         ].head(10)
    
    )

# %%
threshold = 0.50
results["Predicted_malignant"] = (
    results["P_malignant"] >= threshold
).astype(int)
results["Predicted_label"] = results[
    "Predicted_malignant"
].map({
    1: "Malignant",
    0: "Benign"
})
print(
    results[
     [
            "Actual_label",
            "P_malignant" ,
            "Predicted_label"
        ]
    ].head(10)
)



# %%
for threshold in [0.30, 0.50, 0.70]:
    prediction = (
        results["P_malignant"] >= threshold).astype(int)
    print(
        f"Threshold = {threshold}:"
        f"Predicted malignant cases = {prediction.sum()}"

    )
    

# %%
actual_malignant = (y_test.values == 0).astype(int)


for threshold in [0.30, 0.50, 0.70]:
    predicted_malignant = (
        probabilities[:, 0] >= threshold
    ).astype(int)

    cm = confusion_matrix(
        actual_malignant,
        predicted_malignant
    )
    print(f"\nThreshold = (threshold)")
    print(cm)
    


# %%
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

print(
    "sklearn accuracy:",
    accuracy_score(y_test, prediction)
)
print(
    "sklearn precision:",
    precision_score(y_test, prediction)
)
print(
    "sklearn recall:",
    recall_score(y_test, prediction)
)
print(
    "sklearn f1 score:",
    f1_score(y_test,prediction)
)
'''display(
    threshold_metrics.round(3)
)'''


# %%
from sklearn.metrics import ( confusion_matrix, accuracy_score, precision_score, recall_score, f1_score)
thresholds = [0.30, 0.50, 0.70]
metrics = []
for threshold in thresholds:
    predictions = (
        results["P_malignant"] < threshold
    ).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions
    ). ravel()

    metrics.append ({
        "Threshold": threshold,
        "TN": tn,
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1 Score": f1_score(y_test, predictions)
    })
    
metrics_table = pd. DataFrame(metrics)
print(metrics_table.round(4))


