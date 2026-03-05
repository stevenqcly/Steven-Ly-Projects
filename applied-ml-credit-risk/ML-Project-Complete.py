import pandas as pd
import numpy as np

# ------ Step 1: Load and preprocess the data ------ #
sample_data = pd.read_csv(r"C:\Users\nxm240006\Downloads\sampled_train_data.csv")

# Convert 'S_2' column to datetime
sample_data['S_2'] = pd.to_datetime(sample_data['S_2'])
print("Columns with datetime dtype:", sample_data.select_dtypes(include=['datetime64']).columns)

# ------ Step 2: One-Hot Encoding ------ #
# Specify categorical columns for one-hot encoding
categorical_cols = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
sample_data_encoded = pd.get_dummies(sample_data, columns=categorical_cols, drop_first=True, dtype=int)

# Track newly created columns
newly_created_columns = [
    col for col in sample_data_encoded.columns.tolist()
    if col not in sample_data.columns
]

# ------ Step 3: Filtering Data ------ #
# Filter data for transactions up to April 2018
sample_data_filtered = sample_data[sample_data['S_2'] <= '2018-04-30']
sample_data_filtered = sample_data_filtered.sort_values(by=['customer_ID', 'S_2'])

# ------ Step 4: Aggregation-Based Features ------ #
df = sample_data_filtered.copy()

# Identify numerical features
num_features = [
    col for col in df.columns
    if col not in ['S_2', 'customer_ID', 'target'] + categorical_cols + newly_created_columns
]

# Aggregate numerical features
aggregated_features = df.set_index('customer_ID').groupby("customer_ID")[num_features].agg(['mean', 'sum', 'min', 'max', 'std'])
aggregated_features.columns = ['_'.join(col) for col in aggregated_features.columns]
print(aggregated_features.head())
print(len(aggregated_features))



# ------ Step 5: Rolling Features ------ #
reference_date = pd.Timestamp("2018-03-31")
 
# Function to calculate stats for last N months
def last_n_months_stats(df, months, num_columns):
    start_date = reference_date - pd.DateOffset(months=months)
   
    # Filter only data within the required time range
    df_filtered = df[(df["S_2"] > start_date) & (df["S_2"] <= reference_date)]
 
    # Aggregate per customer
    agg_df = df_filtered.groupby("customer_ID")[num_columns].agg(["mean", "min", "max", "std"])
 
    # Flatten column names (remove sub-columns)
    agg_df.columns = [f"{col}_{stat}_{months}m" for col, stat in agg_df.columns]
 
    return agg_df.reset_index()
 
# Compute stats for 3, 6, 9, and 12 months
stats_3m = last_n_months_stats(df, 3, num_features)
stats_6m = last_n_months_stats(df, 6, num_features)
stats_9m = last_n_months_stats(df, 9, num_features)
stats_12m = last_n_months_stats(df, 12, num_features)
 
# Merge all results into one final dataset (1 row per customer)
df_rolling = stats_3m.merge(stats_6m, on="customer_ID", how="left")\
                    .merge(stats_9m, on="customer_ID", how="left")\
                    .merge(stats_12m, on="customer_ID", how="left")
print(df_rolling.head())

# Merge rolling features with aggregated features

final_file = pd.merge(aggregated_features, df_rolling, on='customer_ID', how="inner")
print(len(final_file))
print(final_file.head())





# ------ Step 6: Recency-Based Features ------ #
REFERENCE_DATE = pd.Timestamp('2018-03-31')

last_statement_date = df.groupby('customer_ID')['S_2'].max()
days_since_last_statement = REFERENCE_DATE - last_statement_date
days_since_last_statement = days_since_last_statement.dt.days
days_since_last_statement = days_since_last_statement.reset_index()
days_since_last_statement = days_since_last_statement.add_suffix('_days_since_last_statement')
days_since_last_statement.rename(columns={'customer_ID_days_since_last_statement': 'customer_ID'}, inplace=True)
print(days_since_last_statement.head())
print(len(days_since_last_statement))
# Step 3: Merge this information back into the original dataframe
final_file = final_file.merge(days_since_last_statement, on='customer_ID', how='inner')
print(final_file.head())


# ------ Step 7: Delinquency-Risk Ratio ------ #
d_col = df.filter(regex='^D_').select_dtypes(include=[np.number]).columns
r_col = df.filter(regex='^R_').select_dtypes(include=[np.number]).columns

# Group by 'customer_ID' and calculate the mean for D_ and R_ columns
df_grouped = df.groupby('customer_ID')[list(d_col)].mean().reset_index()

# Create a dynamic column name for the 'D_avg' column
df_grouped['D_avg'] = df_grouped[d_col].mean(axis=1)

# Calculate the mean for the R_ columns and create 'R_avg' column
df_grouped['R_avg'] = df.groupby('customer_ID')[list(r_col)].mean().reset_index()[r_col].mean(axis=1)

print(df_grouped['R_avg'].head())
print(df_grouped['D_avg'].head())
print(df_grouped.head())


# Merge the grouped means back into the original DataFrame
final_file = final_file.merge(df_grouped[['customer_ID', 'D_avg', 'R_avg']], on='customer_ID', how='left')
print(final_file.head())

# Calculate the D_To_R_Ratio column
final_file['D_To_R_Ratio'] = final_file['D_avg'] / final_file['R_avg']
print(final_file.head())

final_file_copy = final_file.copy()


# ------ Step 8: Response Rate Features ------ #

def last_n_months_categorical_stats(df, months, cat_columns):
    start_date = reference_date - pd.DateOffset(months=months)
   
    # Filter only data within the required date range
    df_filtered = df[(df["S_2"] > start_date) & (df["S_2"] <= reference_date)]
   
    # Compute response rate (percentage of times the value is 1)
    response_rate = df_filtered.groupby("customer_ID")[cat_columns].sum() / df_filtered.groupby("customer_ID")[cat_columns].count()
   
    # Compute ever response (whether 1 appeared at least once)
    ever_response = df_filtered.groupby("customer_ID")[cat_columns].apply(lambda x: (x.sum() > 0).astype(int))
   
    # Rename columns
    response_rate.columns = [f"{col}_Response_Rate_{months}m" for col in response_rate.columns]
    ever_response.columns = [f"{col}_Ever_Response_{months}m" for col in ever_response.columns]
   
    # Merge response rate and ever response
    agg_df = response_rate.merge(ever_response, on="customer_ID")
   
    return agg_df.reset_index()
 
# Compute stats for 3, 6, 9, and 12 months

stats_3m = last_n_months_categorical_stats(sample_data_encoded, 3, newly_created_columns)
stats_6m = last_n_months_categorical_stats(sample_data_encoded, 6, newly_created_columns)
stats_9m = last_n_months_categorical_stats(sample_data_encoded, 9, newly_created_columns)
stats_12m = last_n_months_categorical_stats(sample_data_encoded, 12, newly_created_columns)

df_response = stats_3m.merge(stats_6m, on="customer_ID", how="left")\
                    .merge(stats_9m, on="customer_ID", how="left")\
                    .merge(stats_12m, on="customer_ID", how="left")
 

print(df_response['D_68_6.0_Response_Rate_12m'].describe())
print(len(df_response))


final_file1 = pd.merge(final_file, df_response, on='customer_ID', how="inner")


#Ensuring About the NaN values
final_file1 = final_file1.replace({None: np.nan})

#Creating the Target Variable as a DataFrame
df1 = df.drop_duplicates(subset='customer_ID')
final_file1 = final_file1.merge(df1[['customer_ID', 'target']], on='customer_ID', how="left")

#Removing the customer_ID Column
final_file1 = final_file1.drop(columns='customer_ID')


#Spliting Data Into Train and Test
import xgboost as xgb
from sklearn.model_selection import train_test_split

train_df, temp_df = train_test_split(final_file1, test_size=0.3, random_state=42)
test1_df, test2_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print(len(train_df))
print(len(test1_df))
print(len(test2_df))


train_y = train_df['target']
train_df = train_df.drop(columns='target')

test1_y = test1_df['target']
test1_df = test1_df.drop(columns='target')

test2_y = test2_df['target']
test2_df = test2_df.drop(columns='target')

#Initial XGBoost
model_1 = xgb.XGBClassifier()
model_1.fit(train_df, train_y)

#Finding the Best Features1
importances_1 = model_1.feature_importances_

feature_importance_df_1 = pd.DataFrame({
    'feature': train_df.columns,
    'importance': importances_1
})

feature_importance_df_1 = feature_importance_df_1.sort_values(by='importance', ascending=False)

feature_importance_df_1.to_csv(r"C:\Users\nxm240006\Downloads\feature_importance_1.csv", index=False)

#Second Model
model_2 = xgb.XGBClassifier(
    n_estimators=300,  # 300 trees
    learning_rate=0.5,  # Learning rate of 0.5
    max_depth=4,  # Maximum depth of trees
    subsample=0.5,  # Use 50% of observations for each tree
    colsample_bytree=0.5,  # Use 50% of features for each tree
    scale_pos_weight=5,  # Weight of 5 for the default (minority class)
    missing=np.nan,  # Handle missing values as NaN
    use_label_encoder=False  # Avoid deprecated warning for label encoding
)

model_2.fit(train_df, train_y)

#Finding the Best Features2

importances_2 = model_2.feature_importances_
feature_importance_df_2 = pd.DataFrame({
    'feature': train_df.columns,
    'importance': importances_2
})
feature_importance_df_2 = feature_importance_df_2.sort_values(by='importance', ascending=False)
feature_importance_df_2.to_csv(r"C:\Users\nxm240006\Downloads\feature_importance_2.csv", index=False)

combined_importance = pd.concat([feature_importance_df_1.set_index('feature'),
                                feature_importance_df_2.set_index('feature')],
                               axis=1, keys=['model_1', 'model_2'])

combined_importance['max_importance'] = combined_importance.max(axis=1)

important_features = combined_importance[combined_importance['max_importance'] > 0.005].index.tolist()

print(len(important_features))
print(important_features)

#Keeping the Best Featurs

train_df = train_df[important_features]
test1_df = test1_df[important_features]
test2_df = test2_df[important_features]

train_df.to_csv(r"C:\Users\nxm240006\Downloads\train_df.csv", index=False)
test1_df.to_csv(r"C:\Users\nxm240006\Downloads\test1_df.csv", index=False)
train_y.to_csv(r"C:\Users\nxm240006\Downloads\train_y.csv", index=False)
test1_y.to_csv(r"C:\Users\nxm240006\Downloads\test1_y.csv", index=False)
test2_y.to_csv(r"C:\Users\nxm240006\Downloads\test2_y.csv", index=False)


#Final Model
from sklearn.metrics import roc_auc_score
results = []

# Loop over the combinations manually (without using a param_grid dictionary)
for n_estimators in [50, 100, 300]:  # Number of trees
    for learning_rate in [0.01, 0.1]:  # Learning rates
        for subsample in [0.5, 0.8]:  # Percentage of observations used in each tree
            for colsample_bytree in [0.5, 1.0]:  # Percentage of features used in each tree
                for scale_pos_weight in [1, 5, 10]:  # Weight of default observations
                    
                    # Set up the model with the current hyperparameters
                    model = xgb.XGBClassifier(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        scale_pos_weight=scale_pos_weight,
                        missing=np.nan,
                        use_label_encoder=False
                    )

                    # Train the model
                    model.fit(train_df, train_y)

                    # Get the AUC score on training and both testing sets
                    y_train_pred = model.predict_proba(train_df)[:, 1]  # Predicted probabilities for AUC
                    y_test1_pred = model.predict_proba(test1_df)[:, 1]  # Predicted probabilities for AUC (Test 1)
                    y_test2_pred = model.predict_proba(test2_df)[:, 1]  # Predicted probabilities for AUC (Test 2)

                    auc_train = roc_auc_score(train_y, y_train_pred)
                    auc_test1 = roc_auc_score(test1_y, y_test1_pred)
                    auc_test2 = roc_auc_score(test2_y, y_test2_pred)

                    # Save the results for the current iteration
                    results.append({
                        'Trees': n_estimators,
                        'LR': learning_rate,
                        'Subsample': f'{int(subsample*100)}%',
                        '% Features': f'{int(colsample_bytree*100)}%',
                        'Weight of Default': scale_pos_weight,
                        'AUC Train': auc_train,
                        'AUC Test 1': auc_test1,
                        'AUC Test 2': auc_test2
                    })


                    # Save the results to CSV after each iteration to avoid data loss
                    results_df = pd.DataFrame(results)
                    results_df.to_csv(r"C:\Users\nxm240006\Downloads\grid_search_results.csv", index=False)

grid_search_results = pd.read_csv(r"C:\Users\nxm240006\Downloads\grid_search_results.csv")

#---------------------- BIAS ----------------------#
# Calculate the variance of AUC test scores for each model
grid_search_results["Avg AUC Test"] = grid_search_results[["AUC Test 1", "AUC Test 2"]].mean(axis=1)
grid_search_results["AUC Test Variance"] = grid_search_results[["AUC Train","AUC Test 1", "AUC Test 2"]].var(axis=1)

# Retrieve variance for the previously selected best model
best_model_variance = grid_search_results.loc[grid_search_results["Avg AUC Test"].idxmax(), "AUC Test Variance"]

# Identify the most stable model (lowest variance) among the top-performing ones
top_models = grid_search_results.nlargest(5, "Avg AUC Test")
most_stable_model = top_models.loc[top_models["AUC Test Variance"].idxmin()]

print(most_stable_model)



#Final XGBoost Model

model_3 = xgb.XGBClassifier(
    n_estimators=300,  # 300 trees
    learning_rate=0.01,  # Learning rate of 0.5
    subsample=0.5,  # Use 50% of observations for each tree
    colsample_bytree=1,  # Use 50% of features for each tree
    scale_pos_weight=1,  # Weight of 5 for the default (minority class)
    missing=np.nan,  # Handle missing values as NaN
    use_label_encoder=False  # Avoid deprecated warning for label encoding
)
model_3.fit(train_df, train_y)
import joblib
joblib.dump(model, r"C:\Users\nxm240006\Downloads\xgb_model.pkl")


#------------------------------------- Shap Analysis ----------------------------------------------#

import shap
import pandas as pd

explainer = shap.Explainer(model_3)
shap_values = explainer(train_df)

# Get the feature importance based on the mean absolute SHAP values
shap_importance = pd.DataFrame({
    'Feature': train_df.columns,
    'Mean SHAP': abs(shap_values.values).mean(axis=0)
})

# Sort features by the mean absolute SHAP values
shap_importance = shap_importance.sort_values(by='Mean SHAP', ascending=False)

# Get the top 5 features
top_5_features = shap_importance.head(5)
print(top_5_features)

# Get the names of the top 5 features
top_5_feature_names = top_5_features['Feature'].tolist()

# Extract the data for these top 5 features
top_5_data = train_df[top_5_feature_names]
print(top_5_data.head())

# Calculate summary statistics including 1st, 5th, 95th, and 99th percentiles
summary_statistics = top_5_data.describe(percentiles=[.01, .05, .95, .99])

# Calculate percentage of missing values
missing_percent = top_5_data.isnull().mean() * 100
summary_statistics.loc['% Missing'] = missing_percent

# Print or display the summary statistics
print("Summary Statistics for Top 5 Features with Highest SHAP Values (including % missing):")
print(summary_statistics)

# Save to CSV
summary_statistics.to_csv(r"C:\Users\nxm240006\Downloads\Shap_Analysis.csv")


#---------------------------------------- Plots for Shap Analysis --------------------------------------------#
#----------------------------------------      Beeswarm      -----------------------------------------------#

import matplotlib.pyplot as plt
model = joblib.load(r"C:\Users\nxm240006\Downloads\xgb_model.pkl")
test2_df = pd.read_csv(r"C:\Users\nxm240006\Downloads\test2_df.csv")
explainer = shap.Explainer(model)
shap_values_test2 = explainer(test2_df)

shap.plots.beeswarm(shap_values_test2, max_display=20)  # adjust number of features if needed


#----------------------------------------      Waterfall      -----------------------------------------------#

shap.plots.waterfall(shap_values_test2[8])











#--------------------------------------    Strategy        -------------------------------------------------#

#---------------------------- Creating the 4-Column dataframe --------------------------------#
pd_values = model.predict_proba(train_df)[:, 1]  # Get probability of default (class 1)
train_df['Y_Hat'] = pd_values
train_df.to_csv(r"C:\Users\nxm240006\Downloads\train_df_PD.csv", index=False)


customer_id_col = "customer_ID"
date_col = "S_2"  # should be in datetime format
spend_col = "S_18"
balance_col = "B_15"

# Convert to datetime if not already
df[date_col] = pd.to_datetime(df[date_col])

# Filter the last 6 months of data — for example: Nov 2017 to Apr 2018
start_date = "2017-10-01"
end_date = "2018-03-31"

df_6m = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)].copy()

# Group by customer and calculate average
avg_df = df_6m.groupby(customer_id_col)[[spend_col, balance_col]].mean().reset_index()

# Rename for clarity
avg_df.rename(columns={
    spend_col: "S_Avg",
    balance_col: "B_Avg"
}, inplace=True)

# Show sample

new_cols = pd.concat([train_y, train_df['Y_Hat']], axis=1)
print(new_cols.head())
df_merged = pd.concat([avg_df, new_cols], axis=1)
df_merged.drop(columns='customer_ID')
print(df_merged.head())



#---------------------------------------------- Function ---------------------------------------------------#
def calculate_default_and_revenue(
    df: pd.DataFrame,
    target_col: str,
    pd_col: str,
    balance_col: str,
    spend_col: str,
    threshold: float
):
    # Step 1: Filter accepted customers (those below the PD threshold)
    accepted = df[df[pd_col] < threshold].copy()
    
    if accepted.empty:
        return 0.0, 0.0

    # Step 2: Calculate default rate
    total_accepted = len(accepted)
    defaults = accepted[target_col].sum()
    default_rate = defaults / total_accepted

    # Step 3: Calculate monthly revenue
    accepted['B_Avg'] = accepted[[col for col in df.columns if balance_col in col]].mean(axis=1)
    accepted['S_Avg'] = accepted[[col for col in df.columns if spend_col in col]].mean(axis=1)

    accepted['MonthlyRevenue'] = accepted['B_Avg'] * 0.02 + accepted['S_Avg'] * 0.001

    # Step 4: Set revenue to 0 for defaulters
    accepted['ExpectedRevenue'] = accepted.apply(
        lambda row: 12 * row['MonthlyRevenue'] if row[target_col] == 0 else 0,
        axis=1
    )

    # Step 5: Sum portfolio expected revenue
    total_expected_revenue = accepted['ExpectedRevenue'].sum()

    return default_rate, total_expected_revenue




print(calculate_default_and_revenue(df_merged, 'target', 'Y_Hat', 'B_Avg', 'S_Avg', 0.65))

#--------- Aggressive Treshhold = 0.93 ----------#
#--------- Conservative Treshhold = 0.6 ----------#












#---------------------- Neural Network ------------------------#
import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K

# Step 1: Handle missing values and replace them with 0 (SimpleImputer)

imputer = SimpleImputer(strategy='constant', fill_value=0)
X_train_imputed = imputer.fit_transform(train_df)
X_test1_imputed = imputer.transform(test1_df)
X_test2_imputed = imputer.transform(test2_df)

# === Grid Search Parameters ===
nn_hidden_layers_list = [2, 4]
nn_nodes_list = [4, 6]
nn_activation_list = ['relu', 'tanh']
nn_dropout_list = [0.5, 0.0]  # 50%, 100% (no dropout)
nn_batch_size_list = [100, 10000]
nn_epochs = 20
nn_results_path = r'C:\Users\nxm240006\Downloads\nn_grid_results.csv'
 
# === Load Previous Results if Exist ===
if os.path.exists(nn_results_path):
     result_df_nn = pd.read_csv(nn_results_path)
else:
     result_df_nn = pd.DataFrame(columns=[
         'HL', '# Node', 'Activation Function', 'Dropout', 'Batch Size',
         'AUC Train', 'AUC Test1', 'AUC Test2'
     ])
 
# === Build Neural Network Model ===
def build_model_nn(n_hidden, n_nodes, activation, dropout, input_dim):
     model_nn = Sequential()
     model_nn.add(Dense(n_nodes, activation=activation, input_dim=input_dim))
     if dropout > 0:
         model_nn.add(Dropout(dropout))
     for _ in range(n_hidden - 1):
         model_nn.add(Dense(n_nodes, activation=activation))
         if dropout > 0:
             model_nn.add(Dropout(dropout))
     model_nn.add(Dense(1, activation='sigmoid'))
     model_nn.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=[])
     return model_nn
 

# === Begin Grid Search ===
for nn_hl in nn_hidden_layers_list:
    for nn_node in nn_nodes_list:
        for nn_act in nn_activation_list:
            for nn_drop in nn_dropout_list:
                for nn_bs in nn_batch_size_list:
                    # ... training & evaluation code ...

                    try:
                        auc_train_nn = roc_auc_score(train_y, model_nn.predict(X_train_imputed).ravel())
                        auc_test1_nn = roc_auc_score(test1_y, model_nn.predict(X_test1_imputed).ravel())
                        auc_test2_nn = roc_auc_score(test2_y, model_nn.predict(X_test2_imputed).ravel())
                    except Exception as e:
                        print(f"Error in AUC calculation: {e}")
                        auc_train_nn, auc_test1_nn, auc_test2_nn = np.nan, np.nan, np.nan

                    # Create and append the result immediately
                    new_row_nn = pd.DataFrame([{
                        'HL': nn_hl,
                        '# Node': nn_node,
                        'Activation Function': nn_act,
                        'Dropout': f"{int(nn_drop*100)}%",
                        'Batch Size': nn_bs,
                        'AUC Train': auc_train_nn,
                        'AUC Test1': auc_test1_nn,
                        'AUC Test2': auc_test2_nn
                    }])
                    
                    result_df_nn = pd.concat([result_df_nn, new_row_nn], ignore_index=True)


# Save progress
result_df_nn.to_csv(nn_results_path, index=False)
print("NN Grid Search Complete. Results saved to:", nn_results_path)

nn_grid_results = pd.read_csv(r"C:\Users\nxm240006\Downloads\nn_grid_results.csv")


# Calculate the variance of AUC test scores for each model
nn_grid_results["Avg AUC Test"] = nn_grid_results[["AUC Test1", "AUC Test2"]].mean(axis=1)
nn_grid_results["AUC Test Variance"] = nn_grid_results[["AUC Train","AUC Test1", "AUC Test2"]].var(axis=1)

# Retrieve variance for the previously selected best model
nn_best_model_variance = nn_grid_results.loc[nn_grid_results["Avg AUC Test"].idxmax(), "AUC Test Variance"]

# Identify the most stable model (lowest variance) among the top-performing ones
nn_top_models = nn_grid_results.nlargest(5, "Avg AUC Test")
nn_most_stable_model = nn_top_models.loc[nn_top_models["AUC Test Variance"].idxmin()]

print(nn_most_stable_model)


