import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import shap 

# These are what different features mean 
FEATURE_MEANINGS = {
    "jitter": "stability of your vocal pitch (shakiness)",
    "shimmer": "consistency of your vocal volume",
    "hnr": "clarity of your voice versus breathiness",
    "mfcc": "the unique texture and resonance of your voice",
    "intensity": "the strength and projection of your voice",
    "f1": "precision of your tongue and jaw movements",
    "f2": "precision of your tongue and jaw movements",
    "pitch": "the fundamental frequency and range of your voice"
}

def get_feature_meaning(feature_name):
    # Input the technical column name and get out the feature meaing 
    for key in FEATURE_MEANINGS:
        if key in feature_name.lower():
            return FEATURE_MEANINGS[key]
        
def train_model():
    # Preprocessing of the data
    df = pd.read_csv("audio_features.csv")
    X = df.drop(columns=['Sample ID', 'Label'])
    y = df['Label'] 

    # Normalisation of features. Doesn't affect RF but beneficial for SHAP 
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # 2. Split into Train/Test (80% train, 20% test)
    # We split here so we can test the final model on unseen data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Max depth not too large so the model won't overfit
    # This especially is a risk for our model since we don't have a very large dataset 
    temp_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    temp_model.fit(X_scaled, y)

    # Next, we use the previously trained model to do some feature selection
    # This is beneficial since it prevents overfitting. If our model has too many degrees of freedom
    # it might find some coincidental patterns that don't actually exist 
    # This is definitely a risk for us since we had over 50 features before
    # It was also pretty redundant since we had e.g. 5 different types of jitter 
    importances = temp_model.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    # feature_importance_df.to_csv('feature_importance.csv', index=False) <-- Only need to run once 

    # We get the best feature per category 
    best_per_category = {}
    for _, row in feature_importance_df.iterrows():
        for cat in FEATURE_MEANINGS.keys():
            if cat in row['Feature'].lower() and cat not in best_per_category:
                best_per_category[cat] = row['Feature']
    selected_features = list(best_per_category.values())

    # The above gave us 8 features
    # However, we still want to leave some space for other potentially very important features
    # So we are choosing to allow 15 features in total
    remaining_top = [f for f in feature_importance_df['Feature'] if f not in selected_features]
    selected_features.extend(remaining_top[:(15 - len(selected_features))])

    # print(f"Selected {len(selected_features)} features for the final model:")
    # print(selected_features)

    # Training of the final model and splitting into training and testing data to get accuracy 
    X_train_final = X_train[selected_features]
    X_test_final = X_test[selected_features] # We also filter the test set
    final_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    final_model.fit(X_train_final, y_train)

    y_pred = final_model.predict(X_test_final)
    accuracy = accuracy_score(y_test, y_pred)

    return final_model, scaler, selected_features, accuracy


def get_analysis(input_data, model, scaler, selected_features):
    # Prepare our input data
    input_df = pd.DataFrame([input_data])
    scaled_array = scaler.transform(input_df)
    scaled_df = pd.DataFrame(scaled_array, columns=scaler.feature_names_in_)
    model_input = scaled_df[selected_features]

    severity = model.predict_proba(model_input)[0][1] # First value was prob of healthy
    # This calculates the contribution of each of our features 
    explainer = shap.TreeExplainer(model)
    shap_output = explainer.shap_values(model_input)

    # Handle SHAP version logic (List vs Array vs Explanation)
    if isinstance(shap_output, list):
        shap_vals = np.array(shap_output[1]).flatten()
    elif hasattr(shap_output, "values"):
        vals = shap_output.values
        shap_vals = vals[0, :, 1] if vals.ndim == 3 else vals[0]
    else:
        shap_vals = shap_output[0, :, 1] if shap_output.ndim == 3 else shap_output[0]

    # Impact list will store pairs of feature and their impact score
    impact_list = []
    for name, val in zip(selected_features, shap_vals):
        impact_list.append((name, float(val)))
    top_reasons = sorted(impact_list, key=lambda x: x[1], reverse=True)[:5] # x[1] so sort based on score

    meanings = set([])
    for i, (name, score) in enumerate(top_reasons):
        meaning = get_feature_meaning(name)
        meanings.add(meaning)
    meanings = list(meanings)

    # top_reasons and primary indictors different dtype so use dict
    return {
        "severity_score": severity, 
        "top_indicators": meanings,
    }

final_model, model_scaler, selected_features, accuracy = train_model()
print(f"Accuracy: {accuracy*100:.1f}%\n")

# Use random sample of data (in this case the 40th row)
raw_data = pd.read_csv("audio_features.csv")
sample_row = raw_data.drop(columns=['Sample ID', 'Label']).iloc[40]
sample_dict = sample_row.to_dict()

analysis = get_analysis(sample_dict, final_model, model_scaler, selected_features)

print(f"Severity: {analysis['severity_score']*100:.1f}%")
print("\nMain indicator features:")
for meaning in analysis["top_indicators"]:
    print(f"Your {meaning} is a primary factor.")
