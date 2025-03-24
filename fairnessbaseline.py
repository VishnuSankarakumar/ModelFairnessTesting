import tensorflow as tf
import numpy as np
import pandas as pd
import random
import scipy.stats as st

model = tf.keras.models.load_model("C:/Users/vishn/OneDrive/Documents/CS/Scripts/ise/model_processed_compas.h5")
fp1 = "C:/Users/vishn/OneDrive/Documents/CS/Scripts/ise/processed_compas.csv"
ds = pd.read_csv(fp1)
target_column = "Recidivism"
ds = ds.drop(columns=[target_column])

def perturb_features(case, sensitive_attr, perturb_prob=0.3, noise_level=0.05):
    perturbed = case.copy()
    for col in case.index:
        if col == "Sex" and sensitive_attr == "gender":
            continue
        if col == "Race" and sensitive_attr == "race":
            continue
        val = case[col]
        if np.issubdtype(type(val), np.number):
            if np.random.rand() < perturb_prob:
                noise = val * noise_level * np.random.randn()
                perturbed[col] = int(round(val + noise))
    return perturbed

sensitive_attribute = "race"

num_tests = 250  
biased_cases = 0 

if sensitive_attribute == "race":
    unique_races = ds["Race"].unique()

signed_differences = []

for i in range(num_tests):
    random_idx = random.randint(0, len(ds) - 1)
    original_case = ds.iloc[random_idx].copy()
    modified_case = original_case.copy()

    if sensitive_attribute == "race":
        modified_case["Race"] = random.choice([r for r in unique_races if r != original_case["Race"]]) #randomy assign different race
    else:
        print("Invalid sensitive attribute. Choose 'gender' or 'race'.")
        break
    
    original_case = perturb_features(original_case, sensitive_attribute)
    modified_case = perturb_features(modified_case, sensitive_attribute)
    
    original_input = np.array(original_case).reshape(1, -1)
    modified_input = np.array(modified_case).reshape(1, -1)

    original_pred = model.predict(original_input)[0][0]
    modified_pred = model.predict(modified_input)[0][0]
    
    diff = modified_pred - original_pred
    signed_differences.append(diff)
    
    if round(original_pred) != round(modified_pred): #case is biased only if final rounded prediction changed
        biased_cases += 1     

idi_ratio = biased_cases / num_tests
print(f"\nIDI Ratio: {idi_ratio:.4f} ({biased_cases}/{num_tests} biased cases)")

statistic, p_value = st.wilcoxon(signed_differences, alternative='two-sided')

print("\nWilcoxon signed-rank test results:")
print("Test statistic:", statistic)
print("p-value:", p_value)
median_value = np.median(signed_differences)
mean_value = np.mean(signed_differences)
print("Median of signed differences: ", median_value)
print("Average of signed differences: ", mean_value)
alpha = 0.05  #significance level
if p_value < alpha:
    print("Reject H0: Median not 0, predictions significantly altered")
else:
    print("Hold H0: No statistically significant effect found")
