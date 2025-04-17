import numpy as np

def extract_features_from_sample_battery_from_text(file_text: str):
    # Load the q_d_n values from the text, handling comma separation per line.
    q_d_n_values = []
    for line in file_text.splitlines():
        # Remove leading/trailing whitespace and trailing commas
        value_str = line.strip().rstrip(',')
        if value_str:
            q_d_n_values.append(float(value_str))
    
    # Convert the list to a numpy array
    q_d_n_array = np.array(q_d_n_values)
    
    # Trim trailing zeros from the q_d_n array (assumes zeros at the end indicate no data)
    trimmed_q_d_n = np.trim_zeros(q_d_n_array, 'b')
    
    # Compute total cycles as the length of the trimmed array
    total_cycles = len(trimmed_q_d_n)

    # Define k values for which features are computed
    k_values = [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    
    # Initialize the dictionary to hold features
    features = {}

    for k in k_values:
        # --- Last k cycles ---
        if total_cycles > k:
            slope_last = (trimmed_q_d_n[-1] - trimmed_q_d_n[-k]) / k
            grad_last = np.gradient(trimmed_q_d_n[-k:], 1)
            mean_grad_last = float(np.mean(grad_last))
        else:
            slope_last = np.nan
            mean_grad_last = np.nan
        
        features[f'slope_last_{k}_cycles'] = slope_last
        features[f'mean_grad_last_{k}_cycles'] = mean_grad_last

        # --- First k cycles ---
        if total_cycles > k:
            slope_first = (trimmed_q_d_n[k-1] - trimmed_q_d_n[0]) / k
            grad_first = np.gradient(trimmed_q_d_n[:k], 1)
            mean_grad_first = float(np.mean(grad_first))
        else:
            slope_first = np.nan
            mean_grad_first = np.nan

        features[f'slope_first_{k}_cycles'] = slope_first
        features[f'mean_grad_first_{k}_cycles'] = mean_grad_first

    # Add total cycles to the feature set
    features['total_cycles'] = total_cycles

    return features
