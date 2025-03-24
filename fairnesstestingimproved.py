import tensorflow as tf
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import scipy.stats as st  # For hypothesis testing

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


model = tf.keras.models.load_model("C:/Users/vishn/OneDrive/Documents/CS/Scripts/ise/model_processed_compas.h5") #load model
fp1 = "C:/Users/vishn/OneDrive/Documents/CS/Scripts/ise/processed_compas.csv" #load dataset
ds = pd.read_csv(fp1)
target_column = "Recidivism"
ds = ds.drop(columns=[target_column])  #drop target column


sensitive_attribute = "race"  #attribute to test for bias
num_tests = 250  
biased_cases = 0 #to calculate idi ratio
threshold = 0.05 #threshold to be beaten for a case to be considered biased

if sensitive_attribute == "race": #how many unique options for the attribute?
    unique_races = ds["Race"].unique()
    
signed_differences = []  #needed for wilcoxon hyp test

for i in range(num_tests):
    random_idx = random.randint(0, len(ds) - 1) #pick random test case
    original_case = ds.iloc[random_idx].copy()
    modified_case = original_case.copy()
   
    if sensitive_attribute == "race":
        #use greedy heuristic to choose candidate with largest prediction difference.   
        original_input = np.array(original_case).reshape(1, -1)
        original_pred = model.predict(original_input)[0][0] #obtain original case prediction score         
        best_race = None
        max_diff = -1  #initialise with low value
        
        for candidate_race in [r for r in unique_races if r != original_case["Race"]]: #iterate over all races, find largest abs score difference
            temp_case = original_case.copy()
            temp_case["Race"] = candidate_race
            temp_input = np.array(temp_case).reshape(1, -1)
            candidate_pred = model.predict(temp_input)[0][0]            
            diff_candidate = abs(candidate_pred - original_pred)
            if diff_candidate > max_diff:
                max_diff = diff_candidate
                best_race = candidate_race        

        modified_case["Race"] = best_race #keep best candidate        
    else:
        print("Invalid sensitive attribute. Choose 'gender' or 'race'.") #throw error and break if invalid
        break
    
    original_case = perturb_features(original_case, sensitive_attribute) #random noise to nonsensitive attributes
    modified_case = perturb_features(modified_case, sensitive_attribute)

    original_input = np.array(original_case).reshape(1, -1) #convert to numpys in order to predict scores
    modified_input = np.array(modified_case).reshape(1, -1)
    
    original_pred = model.predict(original_input)[0][0] #obtain preds for both og and modified cases
    modified_pred = model.predict(modified_input)[0][0]    

    diff = modified_pred - original_pred  # signed difference
    abs_diff = abs(diff)  # absolute difference    
    signed_differences.append(diff)
    
    print(f"Test case {i}: Original: {original_pred:.4f}, Modified: {modified_pred:.4f}, Diff: {diff:.4f}") #for reference in terminal
    
    if abs_diff > threshold: #assign biased or not biased using preset threshold
        biased_cases += 1
        print("Bias detected (based on threshold)")
    else:
        print("No bias detected (based on threshold)")    
    

idi_ratio = biased_cases / num_tests #calculate idi
print(f"IDI Ratio: {idi_ratio:.4f} ({biased_cases}/{num_tests} biased cases)")

statistic, p_value = st.wilcoxon(signed_differences, alternative='two-sided') #h0: median is 0, reject h0 if median not 0

print("\nWilcoxon signed-rank test results:")
print("Test statistic:", statistic)
print("p-value:", p_value)
median_value = np.median(signed_differences)
print("Median of signed differences:", median_value)
mean_value = np.mean(signed_differences)
print("Average of signed differences: ", mean_value)

alpha = 0.05  #significance level
if p_value < alpha:
    print("Reject H0: Median not 0, predictions significantly altered")
else:
    print("Hold H0: No statistically significant effect found")

