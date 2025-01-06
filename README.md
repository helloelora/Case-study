# High-Value customer prediction for ShipSmall

## Context

ShipSmall specializes in delivering small parcels for e-commerce businesses. The goal of this analysis is to identify customers with the potential to become high-value clients (or "VIPs") and use these insights to guide acquisition and retention strategies.

## Methodology

### 1. Defining High-Value customers
- A customer is considered high-value if:
  - Their customer satisfaction score is 4 or higher.
  - They belong to the top 25% in terms of the number of monthly parcels shipped (75th percentile).

### 2. Data preparation
- **Feature engineering**:
  - `Cost_per_Package`: Shipping cost relative to the number of parcels (adjusted to avoid division by zero).
  - `Distance_per_Weight`: Logistic efficiency measured as distance traveled divided by parcel weight.
- **Categorical encoding**: Used **OneHotEncoder** for categorical features.
- **Handling class imbalance**: Applied **SMOTE** to address the imbalance between high-value and non-high-value customers.

### 3. Models tested
- **Logistic Regression**: Used as a baseline but produced unsatisfactory results
- **Random Forest**: Achieved better performance but lacked interpretability
- **Decision Tree Classifier**: Selected for its simplicity and ability to provide interpretable insights. Used graphviz for an interpretable result

### 4. Performance evaluation
- **Recall** prioritized to maximize identification of high-value customers.
- **5-fold cross-validation** used to ensure consistent performance and prevent overfitting.

## Results
### Decision Tree Classifier
- The model highlighted clear patterns for identifying high-value customers with low Gini impurity at key decision nodes.
- **Recall**: 95%
- **Precision**: 33%
- **Key Features Identified**:
  - Lower cost per package for high-value customers.
  - Account type, with business accounts representing a significant portion of high-value customers.
  - Shipping distance, typically under 400 miles for high-value clients.
  - Age, with younger adults (20–30 years old) being more likely to be high-value customers.

## Libraries Used

| **Library**      | **Role**                                                                                                  |
|------------------|----------------------------------------------------------------------------------------------------------|
| `scikit-learn`   | Tools for preprocessing, model training (Decision Tree, Random Forest), and performance evaluation.       |
| `pandas`         | Used for loading, cleaning, and manipulating tabular data.                                               |
| `imblearn`       | Enabled the use of **SMOTE** to balance classes and prevent bias toward the majority class.               |
| `graphviz`       | Generated clear and interpretable visualizations of decision trees to understand VIP characteristics.     |
| `numpy`          | Provided mathematical operations and data structures for handling arrays and matrices.                   |

## Insights and Recommendations

1. **Acquire business accounts**:
   - Offer volume-based discounts or enhanced customer support to attract businesses.
2. **Optimize shipping costs**:
   - Highlight competitive costs while maintaining service quality.
3. **Demographic targeting**:
   - Focus on digital campaigns targeting younger adults (20–30 years old).
4. **Regional campaigns**:
   - Target regions where customers primarily ship within shorter distances.

## Visualizations

- Decision Tree visualized using **Graphviz**.
- Feature importance displayed in a table and graph.
