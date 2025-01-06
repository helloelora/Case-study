# Import libraries
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.model_selection import cross_val_score
import graphviz

# Load data
file_path = 'ShipSmall.csv'
df = pd.read_csv(file_path)

# Define high-value threshold
key_metrics = ['Monthly_Package_Count']
thresholds = df[key_metrics].quantile(0.75)

# Create High_Value_Client column
df['High_Value_Client'] = (
    (df['Customer_Satisfaction'] >= 4) & 
    (df['Monthly_Package_Count'] >= thresholds['Monthly_Package_Count'])
).astype(int)

# Feature Engineering
df['Cost_per_Package'] = df['Shipping_Cost'] / (df['Monthly_Package_Count'] + 1)

# Enhanced feature set
feature_cols = [
    'Age',  
    'Shipping_Cost',
    'Distance',
    'Cost_per_Package',
    'Account_Type',  # Categorical feature
]

X = df[feature_cols]
y = df['High_Value_Client']

categorical_features = ['Account_Type']

# Preprocessing pipeline (Only OneHotEncoding for categorical features)
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_features)  # Encode categorical features
    ],
    remainder='passthrough'  # Keep numerical features as they are
)

X_transformed = preprocessor.fit_transform(X)

# Get feature names after transformation
feature_names = preprocessor.get_feature_names_out()

# Remove the 'remainder__' prefix from numerical features
feature_names = [name.replace('remainder__', '') for name in feature_names]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42, stratify=y)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42, sampling_strategy=0.8)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Train DecisionTreeClassifier
dt_model = DecisionTreeClassifier(
    max_depth=9,  # Limit depth for better visualization
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight={0: 1, 1: 10},  # Balancing the classes
    random_state=42
)

dt_model.fit(X_resampled, y_resampled)

# Predictions
y_pred = dt_model.predict(X_test)

# Evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Non-High-Value', 'High-Value']))

scores = cross_val_score(dt_model, X_resampled, y_resampled, cv=5, scoring='recall')

print(scores)

# Export the decision tree to a DOT file
dot_data = export_graphviz(
    dt_model,
    out_file=None,  # Output directly as a string
    feature_names=feature_names,  # Feature names after transformation
    class_names=['Non-High-Value', 'High-Value'],
    filled=True,  # Add colors to nodes
    rounded=True,  # Round node edges
    special_characters=True
)

# Visualize the tree
graph = graphviz.Source(dot_data)
graph.render("Decision_Tree")  # Save the visualization to a file
graph.view()  # Open the visualization in a viewer

# Feature importance
importances = pd.Series(
    dt_model.feature_importances_,
    index=preprocessor.get_feature_names_out()
).sort_values(ascending=False)

print("\nTop important features:")
print(importances.head(10))
