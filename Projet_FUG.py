import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import brentq

st.title('Multicomponent Distillation Calculator')

# Ask for the number of components
num_components = st.number_input('Number of Components', min_value=2, max_value=10, value=2)

# Initialize a DataFrame with placeholder values
data = {
    'Component': [f'Component {i+1}' for i in range(int(num_components))],
    'xD': [0.0] * int(num_components),
    'xW': [0.0] * int(num_components),
    'xF': [0.0] * int(num_components),
    'Ki': [1.0] * int(num_components)
}
df = pd.DataFrame(data)

# Use st.data_editor for user input
edited_df = st.data_editor(df, num_rows=int(num_components))

# Additional user input for feed thermal quality 'q'
q = st.number_input('Feed Thermal Quality (q)', min_value=0.0, max_value=1.0, value=1.0)

# Button to trigger calculations
if st.button('Calculate'):
    # Convert edited_df to float
    try:
        edited_df[['xD', 'xW', 'xF', 'Ki']] = edited_df[['xD', 'xW', 'xF', 'Ki']].astype(float)
    except ValueError:
        st.error('Please enter numerical values for all fields.')
        st.stop()
    
    # Check sum of compositions
    if not np.isclose(edited_df['xD'].sum(), 1.0):
        st.error('xD compositions must sum to 1.0')
        st.stop()
    if not np.isclose(edited_df['xW'].sum(), 1.0):
        st.error('xW compositions must sum to 1.0')
        st.stop()
    if not np.isclose(edited_df['xF'].sum(), 1.0):
        st.error('xF compositions must sum to 1.0')
        st.stop()
    
    # Check Ki values are positive
    if any(edited_df['Ki'] <= 0):
        st.error('Ki values must be positive.')
        st.stop()
    
    # Extract data for calculations
    xD = edited_df['xD'].values
    xW = edited_df['xW'].values
    xF = edited_df['xF'].values
    Ki = edited_df['Ki'].values
    
    # Find most and least volatile components
    most_volatile_index = np.argmax(Ki)
    least_volatile_index = np.argmin(Ki)
    
    # Calculate alpha_km
    alpha_km = Ki[most_volatile_index] / Ki[least_volatile_index]
    
    # Calculate ratio
    ratio = (xD[most_volatile_index] / xW[most_volatile_index]) * (xW[least_volatile_index] / xD[least_volatile_index])
    
    # Calculate Nmin using Fenske equation
    Nmin = np.log(ratio) / np.log(alpha_km)
    st.write(f'Minimum Stages (Nmin) = {Nmin:.2f} stages')
    
    # Underwood equation for θ
    def underwood_theta(theta):
        return np.sum(xF * (Ki - 1) / (Ki - theta)) - 1 + q
    
    # Find θ using brentq method
    try:
        theta = brentq(underwood_theta, a=np.min(Ki) + 1e-6, b=np.max(Ki) - 1e-6)
    except:
        st.error('Unable to find θ. Check Ki and feed compositions.')
        st.stop()
    
    # Calculate Rmin using Underwood equation
    Rmin = np.sum(xD * (Ki - 1) / (Ki - theta)) - 1
    st.write(f'Minimum Reflux (Rmin) = {Rmin:.2f}')
    
    # Gilliland correlation for N
    R = 1.1 * Rmin  # Safety factor
    X = (R - Rmin) / (R + 1)
    Y = 1 - np.exp((1 + 54.4 * X) / (11 + 117.2 * X) * (X - 1) / np.sqrt(X))
    N = Nmin + Y * (1 + Nmin)
    st.write(f'Actual Stages (N) = {N:.2f} stages')
    
    # Feed stage location using Kirkbride equation
    B = np.sum(xW)  # Bottom flow rate
    D = np.sum(xD)  # Distillate flow rate
    xF_k = xF[most_volatile_index]
    xW_k = xW[most_volatile_index]
    xD_k = xD[most_volatile_index]
    NF = int(np.round((N - 1) * (B / D) ** 0.206 * (xF_k / xW_k) ** 0.5 * (xW_k / xD_k) ** 0.5))
    
    # Ensure feed stage is within the range of 1 to N
    if NF < 1 or NF > N:
        st.warning('Calculated feed stage is outside the expected range. Please check input values.')
    else:
        st.write(f'Feed stage is stage {NF:.2f}')
    
    # Display results in a table
    results = {
        'Minimum Stages (Nmin)': Nmin,
        'Minimum Reflux (Rmin)': Rmin,
        'Actual Stages (N)': N,
        'Feed Stage': NF
    }
    st.table(results)

    