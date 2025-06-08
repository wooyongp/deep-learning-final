# %%
import pandas as pd
import numpy as np
np.random.seed(42)

# %%
data = pd.read_csv("https://drive.google.com/uc?id=1wD8h8pjCLDy1RbuPDZBSa3zH45TZL7ha&export=download") 

# %%
data

# %%
data = data.drop(columns=['Unnamed: 0'])

# %%
# Select only numeric columns for summary statistics
summary_stats = data.describe().T

# Optionally, round for better LaTeX formatting
summary_stats = summary_stats.round(2)

# Export to LaTeX
latex_table = summary_stats.to_latex()
with open("../doc/summary_statistics.tex", "w") as f:
    f.write(latex_table)

# Display the summary statistics table
summary_stats

# %%
from scipy.spatial import distance

# Define pre-treatment period
pre_treatment = data['year'] < 1989

# Define treated and control groups
treated = data['state'] == 3
control = data['state'] != 3

# Select covariates for balance check (excluding identifiers and outcome)
covariates = ['lnincome', 'beer', 'age15to24', 'retprice']

# Compute means for treated and control in pre-treatment period
treated_means = data.loc[pre_treatment & treated, covariates].mean()
control_means = data.loc[pre_treatment & control, covariates].mean()

# Compute pooled covariance matrix for pre-treatment period
pooled_cov = data.loc[pre_treatment, covariates].cov()

# Compute Mahalanobis distance
mahal_dist = distance.mahalanobis(treated_means, control_means, np.linalg.inv(pooled_cov))
# Compute Mahalanobis distance between treated (state 3) and each other state in pre-treatment period
state_ids = data.loc[pre_treatment & control, 'state'].unique()
mahal_dist_by_state = {}

for s in state_ids:
    state_means = data.loc[pre_treatment & (data['state'] == s), covariates].mean()
    dist = distance.mahalanobis(treated_means, state_means, np.linalg.inv(pooled_cov))
    mahal_dist_by_state[s] = dist

print("Mahalanobis distance (pre-1989, state 3 vs each state):")
for s, dist in mahal_dist_by_state.items():
    print(f"State {s}: {dist:.4f}")

# %%
# Create a DataFrame with state, Mahalanobis distance, and ranking
mahal_df = pd.DataFrame({
    'state': list(mahal_dist_by_state.keys()),
    'Mahalanobis distance': list(mahal_dist_by_state.values())
})

# Rank states by closeness (lower distance = closer, rank 1 is closest)
mahal_df['Rank'] = mahal_df['Mahalanobis distance'].rank(method='min').astype(int)

# Optional: sort by distance
mahal_df = mahal_df.sort_values('Mahalanobis distance')

# If you have a mapping from state number to state name, you can add it here.
# For now, just use the state number as "state name"
mahal_df = mahal_df.rename(columns={'state': 'state name'})

mahal_df.reset_index(drop=True, inplace=True)
mahal_df

# %%
# Merge the rank from mahal_df into the original data by state using join
# First, set the index of mahal_df to 'state name' for joining
mahal_df_indexed = mahal_df.set_index('state name')[['Rank']]

# Join the 'Rank' column to the original data based on 'state'
data_with_rank = data.join(mahal_df_indexed, on='state')

data_with_rank

# %%
# Select relevant columns
subset = data_with_rank[['state', 'year', 'cigsale', 'Rank']].dropna(subset=['Rank'])

# Ensure Rank is integer for sorting
subset['Rank'] = subset['Rank'].astype(int)

# Get unique years and sort them
years_sorted = sorted(subset['year'].unique())

# Sort states by Rank and get their order
states_by_rank = subset[['state', 'Rank']].drop_duplicates().sort_values('Rank')['state'].tolist()

# %%
# Pivot to create the matrix
cigsale_matrix = subset.pivot_table(index='state', columns='year', values='cigsale')

# Reindex rows and columns to match desired order
cigsale_matrix = cigsale_matrix.reindex(index=states_by_rank, columns=years_sorted)

cigsale_matrix

# %%
# Extract cigsale for state 3 for all years in years_sorted
state3_row = data[(data['state'] == 3) & (data['year'].isin(years_sorted))].sort_values('year')
state3_cigsale = state3_row.set_index('year').reindex(years_sorted)['cigsale']
state3_cigsale

# %%
# Add state 3 as the first row in cigsale_matrix
cigsale_matrix_with_california = pd.concat(
    [pd.DataFrame([state3_cigsale.values], index=[3], columns=years_sorted), cigsale_matrix]
)

cigsale_matrix_with_california

# %%
# Mask (set to NaN) the values of state 3 from 1989 onward
cigsale_matrix_with_california.loc[3, 1989:] = np.nan
cigsale_matrix_with_california

# %%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Prepare the data
cigsale_np = cigsale_matrix.values.astype(np.float32)
mask = ~np.isnan(cigsale_np)
mean_val = np.nanmean(cigsale_np)
cigsale_np_filled = np.where(mask, cigsale_np, mean_val)

# Randomly mask 20% of observed entries for training
rng = np.random.rand(*mask.shape)
train_mask = (rng > 0.2) & mask  # 80% observed, 20% masked

# Prepare input and target tensors
input_tensor = cigsale_np_filled[None, ..., None]  # shape (1, states, years, 1)
target_tensor = cigsale_np_filled[None, ..., None]
train_mask_tensor = train_mask[None, ..., None]

# Build the model
model = keras.Sequential([
    layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=input_tensor.shape[1:]),
    # layers.BatchNormalization(),
    layers.Conv2D(1, 3, padding='same')
])

model.compile(optimizer='adam', loss='mse')

# Custom training loop to mask loss
class MaskedLoss(keras.losses.Loss):
    def call(self, y_true, y_pred):
        mask = tf.cast(train_mask_tensor, tf.float32)
        return tf.reduce_sum(mask * tf.square(y_true - y_pred)) / tf.reduce_sum(mask)

masked_loss = MaskedLoss()

# Train the model
model.compile(optimizer='adam', loss=masked_loss)
model.fit(input_tensor, target_tensor, epochs=500, verbose=2)

# Predict missing entries in cigsale_matrix
pred_matrix = model.predict(input_tensor)[0, ..., 0]

# Now apply to cigsale_matrix_with_california to predict state 3 post-1989
cali_np = cigsale_matrix_with_california.values.astype(np.float32)
cali_mask = ~np.isnan(cali_np)
cali_np_filled = np.where(cali_mask, cali_np, mean_val)
cali_input = cali_np_filled[None, ..., None]

cali_pred = model.predict(cali_input)[0, ..., 0]

# Fill in the masked values for state 3, years >= 1989
state3_idx = list(cigsale_matrix_with_california.index).index(3)
year_indices = [i for i, y in enumerate(years_sorted) if y >= 1989]
for yidx in year_indices:
    cigsale_matrix_with_california.iloc[state3_idx, yidx] = cali_pred[state3_idx, yidx]

cigsale_matrix_with_california


# %%
data

# %%
# Create a DataFrame with year, actual cigsale of state 3, and imputed cigsale from cigsale_matrix_with_california
years = years_sorted
actual = state3_cigsale.values
imputed = cigsale_matrix_with_california.loc[3, years].values

comparison_df = pd.DataFrame({
    'year': years,
    'actual_cigsale': actual,
    'imputed_cigsale': imputed
})

comparison_df

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(comparison_df['year'], comparison_df['actual_cigsale'], color='black', label='Actual', linestyle='-')
plt.plot(comparison_df['year'], comparison_df['imputed_cigsale'], color='black', label='Imputed', linestyle='--')
plt.xlabel('Year')
plt.ylabel('Cigarette Sales')
plt.title('Actual vs Imputed Cigarette Sales for State 3')
plt.legend()
plt.axvline(x=1988, color='red', linestyle=':', linewidth=2)
plt.text(1988 + 0.2, plt.ylim()[1]*0.95, "Passage of Proposition 99", color='red', rotation=90, va='top')
plt.tight_layout()
plt.show()

# %%
# Placebo test: repeat synthetic control for states 9, 4, 36, 6, 39 as "treated"
placebo_results = {}

for placebo_state in [36, 6, 39]:
    # 1. Exclude state 3 from the donor pool
    placebo_states = [s for s in states_by_rank if s != 3 and s != placebo_state]
    # 2. Build cigsale matrix for placebo donor pool
    placebo_matrix = subset[subset['state'].isin(placebo_states + [placebo_state])].pivot_table(
        index='state', columns='year', values='cigsale'
    )
    placebo_matrix = placebo_matrix.reindex(index=[placebo_state] + placebo_states, columns=years_sorted)
    # 3. Mask post-1989 for placebo_state
    placebo_matrix.loc[placebo_state, 1989:] = np.nan
    # 4. Prepare data for model
    placebo_np = placebo_matrix.values.astype(np.float32)
    placebo_mask = ~np.isnan(placebo_np)
    placebo_mean = np.nanmean(placebo_np)
    placebo_np_filled = np.where(placebo_mask, placebo_np, placebo_mean)
    # 5. Randomly mask 20% for training
    placebo_rng = np.random.rand(*placebo_mask.shape)
    placebo_train_mask = (placebo_rng > 0.2) & placebo_mask
    placebo_input_tensor = placebo_np_filled[None, ..., None]
    placebo_target_tensor = placebo_np_filled[None, ..., None]
    placebo_train_mask_tensor = placebo_train_mask[None, ..., None]
    # 6. Build and train model
    placebo_model = keras.Sequential([
        layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=input_tensor.shape[1:]),
        # layers.BatchNormalization(),
        layers.Conv2D(1, 3, padding='same')
    ])
    class PlaceboMaskedLoss(keras.losses.Loss):
        def call(self, y_true, y_pred):
            mask = tf.cast(placebo_train_mask_tensor, tf.float32)
            return tf.reduce_sum(mask * tf.square(y_true - y_pred)) / tf.reduce_sum(mask)
    placebo_model.compile(optimizer='adam', loss=PlaceboMaskedLoss())
    placebo_model.fit(placebo_input_tensor, placebo_target_tensor, epochs=500, verbose=0)
    # 7. Predict
    placebo_pred = placebo_model.predict(placebo_input_tensor)[0, ..., 0]
    # 8. Fill in masked values for placebo_state, years >= 1989
    placebo_idx = 0  # first row is placebo_state
    placebo_year_indices = [i for i, y in enumerate(years_sorted) if y >= 1989]
    placebo_matrix_filled = placebo_matrix.copy()
    for yidx in placebo_year_indices:
        placebo_matrix_filled.iloc[placebo_idx, yidx] = placebo_pred[placebo_idx, yidx]
    # 9. Collect actual and imputed for placebo_state
    actual = subset[(subset['state'] == placebo_state) & (subset['year'].isin(years_sorted))].sort_values('year')['cigsale'].values

    imputed = placebo_matrix_filled.loc[placebo_state, years_sorted].values
    placebo_results[placebo_state] = pd.DataFrame({
        'year': years_sorted,
        'actual_cigsale': actual,
        'imputed_cigsale': imputed
    })


# %%
actual = subset[(subset['state'] == placebo_state) & (subset['year'].isin(years_sorted))].sort_values('year')['cigsale'].values
actual

# %%
plt.figure(figsize=(10, 6))
plt.plot(placebo_results[36]['year'], placebo_results[36]['actual_cigsale'], label='Actual', color='black', linestyle='-')
plt.plot(placebo_results[36]['year'], placebo_results[36]['imputed_cigsale'], label='Imputed', color='black', linestyle='--')
plt.xlabel('Year')
plt.ylabel('Cigarette Sales')
plt.title('Actual vs Imputed Cigarette Sales for State 36 (Placebo)')
plt.legend()
plt.axvline(x=1988, color='red', linestyle=':', linewidth=2)
plt.text(1988 + 0.2, plt.ylim()[1]*0.95, "Placebo Treatment", color='red', rotation=90, va='top')
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(10, 6))
plt.plot(placebo_results[6]['year'], placebo_results[6]['actual_cigsale'], label='Actual', color='black', linestyle='-')
plt.plot(placebo_results[6]['year'], placebo_results[6]['imputed_cigsale'], label='Imputed', color='black', linestyle='--')
plt.xlabel('Year')
plt.ylabel('Cigarette Sales')
plt.title('Actual vs Imputed Cigarette Sales for State 6 (Placebo)')
plt.legend()
plt.axvline(x=1988, color='red', linestyle=':', linewidth=2)
plt.text(1988 + 0.2, plt.ylim()[1]*0.95, "Placebo Treatment", color='red', rotation=90, va='top')
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(10, 6))
plt.plot(placebo_results[39]['year'], placebo_results[39]['actual_cigsale'], label='Actual', color='black', linestyle='-')
plt.plot(placebo_results[39]['year'], placebo_results[39]['imputed_cigsale'], label='Imputed', color='black', linestyle='--')
plt.xlabel('Year')
plt.ylabel('Cigarette Sales')
plt.title('Actual vs Imputed Cigarette Sales for State 39 (Placebo)')
plt.legend()
plt.axvline(x=1988, color='red', linestyle=':', linewidth=2)
plt.text(1988 + 0.2, plt.ylim()[1]*0.95, "Placebo Treatment", color='red', rotation=90, va='top')
plt.tight_layout()
plt.show()


# %%
import pandas as pd
import jax.numpy as jnp
import numpy as np
from causaltensor.cauest import MC_NNM_with_cross_validation

# %%
# !curl https://github.com/TianyiPeng/causaltensor/raw/main/tutorials/Synth.zip -L -o Synth.zip
# !unzip -o Synth.zip

# %%
O_raw = np.loadtxt('MLAB_data.txt')
O = O_raw[8:, :] ## remove features that are not relevant in this demo
O = O.T

# %%
import matplotlib.pyplot as plt

plt.plot(O[-1, :])

# %%
Z = np.zeros_like(O) # Z has the same shape as O
Z[-1, 19:] = 1 #Only California (the last row) used the intervention, which started in 1989

# %%
O

# %%
Z

# %%
M, a, b, tau = MC_NNM_with_cross_validation(O, 1-Z)

# %%
M_broadcasted = M + a + b.T
M_broadcasted

# %%
O

# %%
# Merge the last row of O and M into a new array with shape (2, O.shape[1])
merged_last_rows = np.vstack([O[-1, :], M_broadcasted[-1, :]])
merged_last_rows


# %%
import pandas as pd

merged_last_rows_df = pd.DataFrame({
    'O_california': merged_last_rows[0],
    'M_california': merged_last_rows[1]
})
merged_last_rows_df

# %%
plt.figure(figsize=(10, 6))
plt.plot(np.arange(1970,2001), merged_last_rows_df["O_california"], color='black', label='Actual', linestyle='-')
plt.plot(np.arange(1970,2001), merged_last_rows_df["M_california"], color='black', label='Imputed', linestyle='--')
plt.xlabel('Year')
plt.ylabel('Cigarette Sales')
plt.title('Actual vs Imputed Cigarette Sales for State 3')
plt.legend()
plt.axvline(x=1988, color='red', linestyle=':', linewidth=2)
plt.text(1988 + 0.2, plt.ylim()[1]*0.95, "Passage of Proposition 99", color='red', rotation=90, va='top')
plt.tight_layout()
plt.show()

# %%
# Placebo test for State 6 (4)

O_placebo = O[:-1]
Z = np.zeros_like(O_placebo) # Z has the same shape as O
Z[4, 19:] = 1 

# %%
M, a, b, tau = MC_NNM_with_cross_validation(O_placebo, 1-Z)
M_broadcasted = M + a + b.T

# %%
merged_last_rows = np.vstack([O[4, :], M_broadcasted[4, :]])
merged_last_rows
merged_last_rows_df = pd.DataFrame({
    'O_placebo': merged_last_rows[0],
    'M_placebo': merged_last_rows[1]
})
merged_last_rows_df

# %%
plt.figure(figsize=(10, 6))
plt.plot(np.arange(1970,2001), merged_last_rows_df["O_placebo"], color='black', label='Actual', linestyle='-')
plt.plot(np.arange(1970,2001), merged_last_rows_df["M_placebo"], color='black', label='Imputed', linestyle='--')
plt.xlabel('Year')
plt.ylabel('Cigarette Sales')
plt.title('Actual vs Imputed Cigarette Sales for State 6 (Placebo)')
plt.legend()
plt.axvline(x=1988, color='red', linestyle=':', linewidth=2)
plt.text(1988 + 0.2, plt.ylim()[1]*0.95, "Passage of Proposition 99", color='red', rotation=90, va='top')
plt.tight_layout()
plt.show()

# %%
# Placebo test for State 36 (34)
O_placebo = O[:-1]
Z = np.zeros_like(O_placebo) # Z has the same shape as O
Z[34, 19:] = 1

# %%
M, a, b, tau = MC_NNM_with_cross_validation(O_placebo, 1-Z)
M_broadcasted = M + a + b.T
merged_last_rows = np.vstack([O[34, :], M_broadcasted[34, :]])
merged_last_rows_df = pd.DataFrame({
    'O_placebo': merged_last_rows[0],
    'M_placebo': merged_last_rows[1]
})
merged_last_rows_df

# %%
plt.figure(figsize=(10, 6))
plt.plot(np.arange(1970,2001), merged_last_rows_df["O_placebo"], color='black', label='Actual', linestyle='-')
plt.plot(np.arange(1970,2001), merged_last_rows_df["M_placebo"], color='black', label='Imputed', linestyle='--')
plt.xlabel('Year')
plt.ylabel('Cigarette Sales')
plt.title('Actual vs Imputed Cigarette Sales for State 36 (Placebo)')
plt.legend()
plt.axvline(x=1988, color='red', linestyle=':', linewidth=2)
plt.text(1988 + 0.2, plt.ylim()[1]*0.95, "Passage of Proposition 99", color='red', rotation=90, va='top')
plt.tight_layout()
plt.show()

# %%
# Placebo test for State 39 (37)
O_placebo = O[:-1]
Z = np.zeros_like(O_placebo) # Z has the same shape as O
Z[37, 19:] = 1

# %%
M, a, b, tau = MC_NNM_with_cross_validation(O_placebo, 1-Z)
M_broadcasted = M + a + b.T
merged_last_rows = np.vstack([O[37, :], M_broadcasted[37, :]])
merged_last_rows_df = pd.DataFrame({
    'O_placebo': merged_last_rows[0],
    'M_placebo': merged_last_rows[1]
})
merged_last_rows_df

# %%
plt.figure(figsize=(10, 6))
plt.plot(np.arange(1970,2001), merged_last_rows_df["O_placebo"], color='black', label='Actual', linestyle='-')
plt.plot(np.arange(1970,2001), merged_last_rows_df["M_placebo"], color='black', label='Imputed', linestyle='--')
plt.xlabel('Year')
plt.ylabel('Cigarette Sales')
plt.title('Actual vs Imputed Cigarette Sales for State 39 (Placebo)')
plt.legend()
plt.axvline(x=1988, color='red', linestyle=':', linewidth=2)
plt.text(1988 + 0.2, plt.ylim()[1]*0.95, "Passage of Proposition 99", color='red', rotation=90, va='top')
plt.tight_layout()
plt.show()



# %%
import pandas as pd
import numpy as np
np.random.seed(42)

# %%
data = pd.read_csv("https://drive.google.com/uc?id=1wD8h8pjCLDy1RbuPDZBSa3zH45TZL7ha&export=download") 

# %%
data

# %%
data = data.drop(columns=['Unnamed: 0'])

# %%
from scipy.spatial import distance

# Define pre-treatment period
pre_treatment = data['year'] < 1989

# Define treated and control groups
treated = data['state'] == 3
control = data['state'] != 3

# Select covariates for balance check (excluding identifiers and outcome)
covariates = ['lnincome', 'beer', 'age15to24', 'retprice']

# Compute means for treated and control in pre-treatment period
treated_means = data.loc[pre_treatment & treated, covariates].mean()
control_means = data.loc[pre_treatment & control, covariates].mean()

# Compute pooled covariance matrix for pre-treatment period
pooled_cov = data.loc[pre_treatment, covariates].cov()

# Compute Mahalanobis distance
mahal_dist = distance.mahalanobis(treated_means, control_means, np.linalg.inv(pooled_cov))
# Compute Mahalanobis distance between treated (state 3) and each other state in pre-treatment period
state_ids = data.loc[pre_treatment & control, 'state'].unique()
mahal_dist_by_state = {}

for s in state_ids:
    state_means = data.loc[pre_treatment & (data['state'] == s), covariates].mean()
    dist = distance.mahalanobis(treated_means, state_means, np.linalg.inv(pooled_cov))
    mahal_dist_by_state[s] = dist

print("Mahalanobis distance (pre-1989, state 3 vs each state):")
for s, dist in mahal_dist_by_state.items():
    print(f"State {s}: {dist:.4f}")

# %%
# Create a DataFrame with state, Mahalanobis distance, and ranking
mahal_df = pd.DataFrame({
    'state': list(mahal_dist_by_state.keys()),
    'Mahalanobis distance': list(mahal_dist_by_state.values())
})

# Rank states by closeness (lower distance = closer, rank 1 is closest)
mahal_df['Rank'] = mahal_df['Mahalanobis distance'].rank(method='min').astype(int)

# Optional: sort by distance
mahal_df = mahal_df.sort_values('Mahalanobis distance')

# If you have a mapping from state number to state name, you can add it here.
# For now, just use the state number as "state name"
mahal_df = mahal_df.rename(columns={'state': 'state name'})

mahal_df.reset_index(drop=True, inplace=True)
mahal_df

# %%
# Merge the rank from mahal_df into the original data by state using join
# First, set the index of mahal_df to 'state name' for joining
mahal_df_indexed = mahal_df.set_index('state name')[['Rank']]

# Join the 'Rank' column to the original data based on 'state'
data_with_rank = data.join(mahal_df_indexed, on='state')

data_with_rank

# %%
# Select relevant columns
subset = data_with_rank[['state', 'year', 'cigsale', 'Rank']].dropna(subset=['Rank'])

# Ensure Rank is integer for sorting
subset['Rank'] = subset['Rank'].astype(int)

# Get unique years and sort them
years_sorted = sorted(subset['year'].unique())

# Sort states by Rank and get their order
states_by_rank = subset[['state', 'Rank']].drop_duplicates().sort_values('Rank')['state'].tolist()

# %%
# Pivot to create the matrix
cigsale_matrix = subset.pivot_table(index='state', columns='year', values='cigsale')

# Reindex rows and columns to match desired order
cigsale_matrix = cigsale_matrix.reindex(index=states_by_rank, columns=years_sorted)

cigsale_matrix

# %%
# Extract cigsale for state 3 for all years in years_sorted
state3_row = data[(data['state'] == 3) & (data['year'].isin(years_sorted))].sort_values('year')
state3_cigsale = state3_row.set_index('year').reindex(years_sorted)['cigsale']
state3_cigsale

# %%
# Add state 3 as the first row in cigsale_matrix
cigsale_matrix_with_california = pd.concat(
    [pd.DataFrame([state3_cigsale.values], index=[3], columns=years_sorted), cigsale_matrix]
)

cigsale_matrix_with_california

# %%
# Mask (set to NaN) the values of state 3 from 1989 onward
cigsale_matrix_with_california.loc[3, 1989:] = np.nan
cigsale_matrix_with_california

# %%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Prepare the data
cigsale_np = cigsale_matrix.values.astype(np.float32)
mask = ~np.isnan(cigsale_np)
mean_val = np.nanmean(cigsale_np)
cigsale_np_filled = np.where(mask, cigsale_np, mean_val)

# Randomly mask 20% of observed entries for training
rng = np.random.rand(*mask.shape)
train_mask = (rng > 0.2) & mask  # 80% observed, 20% masked

# Prepare input and target tensors
input_tensor = cigsale_np_filled[None, ..., None]  # shape (1, states, years, 1)
target_tensor = cigsale_np_filled[None, ..., None]
train_mask_tensor = train_mask[None, ..., None]

# Build the model
model = keras.Sequential([
    layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=input_tensor.shape[1:]),
    layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=input_tensor.shape[1:]),
    layers.Conv2D(1, 3, padding='same')
])

model.compile(optimizer='adam', loss='mse')

# Custom training loop to mask loss
class MaskedLoss(keras.losses.Loss):
    def call(self, y_true, y_pred):
        mask = tf.cast(train_mask_tensor, tf.float32)
        return tf.reduce_sum(mask * tf.square(y_true - y_pred)) / tf.reduce_sum(mask)

masked_loss = MaskedLoss()

# Train the model
model.compile(optimizer='adam', loss=masked_loss)
model.fit(input_tensor, target_tensor, epochs=500, verbose=2)

# Predict missing entries in cigsale_matrix
pred_matrix = model.predict(input_tensor)[0, ..., 0]

# Now apply to cigsale_matrix_with_california to predict state 3 post-1989
cali_np = cigsale_matrix_with_california.values.astype(np.float32)
cali_mask = ~np.isnan(cali_np)
cali_np_filled = np.where(cali_mask, cali_np, mean_val)
cali_input = cali_np_filled[None, ..., None]

cali_pred = model.predict(cali_input)[0, ..., 0]

# Fill in the masked values for state 3, years >= 1989
state3_idx = list(cigsale_matrix_with_california.index).index(3)
year_indices = [i for i, y in enumerate(years_sorted) if y >= 1989]
for yidx in year_indices:
    cigsale_matrix_with_california.iloc[state3_idx, yidx] = cali_pred[state3_idx, yidx]

cigsale_matrix_with_california


# %%
data

# %%
# Create a DataFrame with year, actual cigsale of state 3, and imputed cigsale from cigsale_matrix_with_california
years = years_sorted
actual = state3_cigsale.values
imputed = cigsale_matrix_with_california.loc[3, years].values

comparison_df = pd.DataFrame({
    'year': years,
    'actual_cigsale': actual,
    'imputed_cigsale': imputed
})

comparison_df

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(comparison_df['year'], comparison_df['actual_cigsale'], color='black', label='Actual', linestyle='-')
plt.plot(comparison_df['year'], comparison_df['imputed_cigsale'], color='black', label='Imputed', linestyle='--')
plt.xlabel('Year')
plt.ylabel('Cigarette Sales')
plt.title('Actual vs Imputed Cigarette Sales for State 3')
plt.legend()
plt.axvline(x=1988, color='red', linestyle=':', linewidth=2)
plt.text(1988 + 0.2, plt.ylim()[1]*0.95, "Passage of Proposition 99", color='red', rotation=90, va='top')
plt.tight_layout()
plt.show()

# %%
# Placebo test: repeat synthetic control for states 9, 4, 36, 6, 39 as "treated"
placebo_results = {}

for placebo_state in [36, 6, 39]:
    # 1. Exclude state 3 from the donor pool
    placebo_states = [s for s in states_by_rank if s != 3 and s != placebo_state]
    # 2. Build cigsale matrix for placebo donor pool
    placebo_matrix = subset[subset['state'].isin(placebo_states + [placebo_state])].pivot_table(
        index='state', columns='year', values='cigsale'
    )
    placebo_matrix = placebo_matrix.reindex(index=[placebo_state] + placebo_states, columns=years_sorted)
    # 3. Mask post-1989 for placebo_state
    placebo_matrix.loc[placebo_state, 1989:] = np.nan
    # 4. Prepare data for model
    placebo_np = placebo_matrix.values.astype(np.float32)
    placebo_mask = ~np.isnan(placebo_np)
    placebo_mean = np.nanmean(placebo_np)
    placebo_np_filled = np.where(placebo_mask, placebo_np, placebo_mean)
    # 5. Randomly mask 20% for training
    placebo_rng = np.random.rand(*placebo_mask.shape)
    placebo_train_mask = (placebo_rng > 0.2) & placebo_mask
    placebo_input_tensor = placebo_np_filled[None, ..., None]
    placebo_target_tensor = placebo_np_filled[None, ..., None]
    placebo_train_mask_tensor = placebo_train_mask[None, ..., None]
    # 6. Build and train model
    placebo_model = keras.Sequential([
        layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=input_tensor.shape[1:]),
        layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=input_tensor.shape[1:]),
        layers.Conv2D(1, 3, padding='same')
    ])
    class PlaceboMaskedLoss(keras.losses.Loss):
        def call(self, y_true, y_pred):
            mask = tf.cast(placebo_train_mask_tensor, tf.float32)
            return tf.reduce_sum(mask * tf.square(y_true - y_pred)) / tf.reduce_sum(mask)
    placebo_model.compile(optimizer='adam', loss=PlaceboMaskedLoss())
    placebo_model.fit(placebo_input_tensor, placebo_target_tensor, epochs=500, verbose=0)
    # 7. Predict
    placebo_pred = placebo_model.predict(placebo_input_tensor)[0, ..., 0]
    # 8. Fill in masked values for placebo_state, years >= 1989
    placebo_idx = 0  # first row is placebo_state
    placebo_year_indices = [i for i, y in enumerate(years_sorted) if y >= 1989]
    placebo_matrix_filled = placebo_matrix.copy()
    for yidx in placebo_year_indices:
        placebo_matrix_filled.iloc[placebo_idx, yidx] = placebo_pred[placebo_idx, yidx]
    # 9. Collect actual and imputed for placebo_state
    actual = subset[(subset['state'] == placebo_state) & (subset['year'].isin(years_sorted))].sort_values('year')['cigsale'].values

    imputed = placebo_matrix_filled.loc[placebo_state, years_sorted].values
    placebo_results[placebo_state] = pd.DataFrame({
        'year': years_sorted,
        'actual_cigsale': actual,
        'imputed_cigsale': imputed
    })


# %%
actual = subset[(subset['state'] == placebo_state) & (subset['year'].isin(years_sorted))].sort_values('year')['cigsale'].values
actual

# %%
plt.figure(figsize=(10, 6))
plt.plot(placebo_results[36]['year'], placebo_results[36]['actual_cigsale'], label='Actual', color='black', linestyle='-')
plt.plot(placebo_results[36]['year'], placebo_results[36]['imputed_cigsale'], label='Imputed', color='black', linestyle='--')
plt.xlabel('Year')
plt.ylabel('Cigarette Sales')
plt.title('Actual vs Imputed Cigarette Sales for State 36 (Placebo)')
plt.legend()
plt.axvline(x=1988, color='red', linestyle=':', linewidth=2)
plt.text(1988 + 0.2, plt.ylim()[1]*0.95, "Placebo Treatment", color='red', rotation=90, va='top')
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(10, 6))
plt.plot(placebo_results[6]['year'], placebo_results[6]['actual_cigsale'], label='Actual', color='black', linestyle='-')
plt.plot(placebo_results[6]['year'], placebo_results[6]['imputed_cigsale'], label='Imputed', color='black', linestyle='--')
plt.xlabel('Year')
plt.ylabel('Cigarette Sales')
plt.title('Actual vs Imputed Cigarette Sales for State 6 (Placebo)')
plt.legend()
plt.axvline(x=1988, color='red', linestyle=':', linewidth=2)
plt.text(1988 + 0.2, plt.ylim()[1]*0.95, "Placebo Treatment", color='red', rotation=90, va='top')
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(10, 6))
plt.plot(placebo_results[39]['year'], placebo_results[39]['actual_cigsale'], label='Actual', color='black', linestyle='-')
plt.plot(placebo_results[39]['year'], placebo_results[39]['imputed_cigsale'], label='Imputed', color='black', linestyle='--')
plt.xlabel('Year')
plt.ylabel('Cigarette Sales')
plt.title('Actual vs Imputed Cigarette Sales for State 39 (Placebo)')
plt.legend()
plt.axvline(x=1988, color='red', linestyle=':', linewidth=2)
plt.text(1988 + 0.2, plt.ylim()[1]*0.95, "Placebo Treatment", color='red', rotation=90, va='top')
plt.tight_layout()
plt.show()

# %%



# %%
import pandas as pd
import numpy as np
np.random.seed(42)

# %%
data = pd.read_csv("https://drive.google.com/uc?id=1wD8h8pjCLDy1RbuPDZBSa3zH45TZL7ha&export=download") 

# %%
data

# %%
data = data.drop(columns=['Unnamed: 0'])

# %%
from scipy.spatial import distance

# Define pre-treatment period
pre_treatment = data['year'] < 1989

# Define treated and control groups
treated = data['state'] == 3
control = data['state'] != 3

# Select covariates for balance check (excluding identifiers and outcome)
covariates = ['lnincome', 'beer', 'age15to24', 'retprice']

# Compute means for treated and control in pre-treatment period
treated_means = data.loc[pre_treatment & treated, covariates].mean()
control_means = data.loc[pre_treatment & control, covariates].mean()

# Compute pooled covariance matrix for pre-treatment period
pooled_cov = data.loc[pre_treatment, covariates].cov()

# Compute Mahalanobis distance
mahal_dist = distance.mahalanobis(treated_means, control_means, np.linalg.inv(pooled_cov))
# Compute Mahalanobis distance between treated (state 3) and each other state in pre-treatment period
state_ids = data.loc[pre_treatment & control, 'state'].unique()
mahal_dist_by_state = {}

for s in state_ids:
    state_means = data.loc[pre_treatment & (data['state'] == s), covariates].mean()
    dist = distance.mahalanobis(treated_means, state_means, np.linalg.inv(pooled_cov))
    mahal_dist_by_state[s] = dist

print("Mahalanobis distance (pre-1989, state 3 vs each state):")
for s, dist in mahal_dist_by_state.items():
    print(f"State {s}: {dist:.4f}")

# %%
# Create a DataFrame with state, Mahalanobis distance, and ranking
mahal_df = pd.DataFrame({
    'state': list(mahal_dist_by_state.keys()),
    'Mahalanobis distance': list(mahal_dist_by_state.values())
})

# Rank states by closeness (lower distance = closer, rank 1 is closest)
mahal_df['Rank'] = mahal_df['Mahalanobis distance'].rank(method='min').astype(int)

# Optional: sort by distance
mahal_df = mahal_df.sort_values('Mahalanobis distance')

# If you have a mapping from state number to state name, you can add it here.
# For now, just use the state number as "state name"
mahal_df = mahal_df.rename(columns={'state': 'state name'})

mahal_df.reset_index(drop=True, inplace=True)
mahal_df

# %%
# Merge the rank from mahal_df into the original data by state using join
# First, set the index of mahal_df to 'state name' for joining
mahal_df_indexed = mahal_df.set_index('state name')[['Rank']]

# Join the 'Rank' column to the original data based on 'state'
data_with_rank = data.join(mahal_df_indexed, on='state')

data_with_rank

# %%
# Select relevant columns
subset = data_with_rank[['state', 'year', 'cigsale', 'Rank']].dropna(subset=['Rank'])

# Ensure Rank is integer for sorting
subset['Rank'] = subset['Rank'].astype(int)

# Get unique years and sort them
years_sorted = sorted(subset['year'].unique())

# Sort states by Rank and get their order
states_by_rank = subset[['state', 'Rank']].drop_duplicates().sort_values('Rank')['state'].tolist()

# %%
# Pivot to create the matrix
cigsale_matrix = subset.pivot_table(index='state', columns='year', values='cigsale')

# Reindex rows and columns to match desired order
cigsale_matrix = cigsale_matrix.reindex(index=states_by_rank, columns=years_sorted)

cigsale_matrix

# %%
# Extract cigsale for state 3 for all years in years_sorted
state3_row = data[(data['state'] == 3) & (data['year'].isin(years_sorted))].sort_values('year')
state3_cigsale = state3_row.set_index('year').reindex(years_sorted)['cigsale']
state3_cigsale

# %%
# Add state 3 as the first row in cigsale_matrix
cigsale_matrix_with_california = pd.concat(
    [pd.DataFrame([state3_cigsale.values], index=[3], columns=years_sorted), cigsale_matrix]
)

cigsale_matrix_with_california

# %%
# Mask (set to NaN) the values of state 3 from 1989 onward
cigsale_matrix_with_california.loc[3, 1989:] = np.nan
cigsale_matrix_with_california

# %%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Prepare the data
cigsale_np = cigsale_matrix.values.astype(np.float32)
mask = ~np.isnan(cigsale_np)
mean_val = np.nanmean(cigsale_np)
cigsale_np_filled = np.where(mask, cigsale_np, mean_val)

# Randomly mask 20% of observed entries for training
rng = np.random.rand(*mask.shape)
train_mask = (rng > 0.2) & mask  # 80% observed, 20% masked

# Prepare input and target tensors
input_tensor = cigsale_np_filled[None, ..., None]  # shape (1, states, years, 1)
target_tensor = cigsale_np_filled[None, ..., None]
train_mask_tensor = train_mask[None, ..., None]

# Build the model
model = keras.Sequential([
    layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=input_tensor.shape[1:]),
    layers.BatchNormalization(),
    layers.Conv2D(1, 3, padding='same')
])

model.compile(optimizer='adam', loss='mse')

# Custom training loop to mask loss
class MaskedLoss(keras.losses.Loss):
    def call(self, y_true, y_pred):
        mask = tf.cast(train_mask_tensor, tf.float32)
        return tf.reduce_sum(mask * tf.square(y_true - y_pred)) / tf.reduce_sum(mask)

masked_loss = MaskedLoss()

# Train the model
model.compile(optimizer='adam', loss=masked_loss)
model.fit(input_tensor, target_tensor, epochs=500, verbose=2)

# Predict missing entries in cigsale_matrix
pred_matrix = model.predict(input_tensor)[0, ..., 0]

# Now apply to cigsale_matrix_with_california to predict state 3 post-1989
cali_np = cigsale_matrix_with_california.values.astype(np.float32)
cali_mask = ~np.isnan(cali_np)
cali_np_filled = np.where(cali_mask, cali_np, mean_val)
cali_input = cali_np_filled[None, ..., None]

cali_pred = model.predict(cali_input)[0, ..., 0]

# Fill in the masked values for state 3, years >= 1989
state3_idx = list(cigsale_matrix_with_california.index).index(3)
year_indices = [i for i, y in enumerate(years_sorted) if y >= 1989]
for yidx in year_indices:
    cigsale_matrix_with_california.iloc[state3_idx, yidx] = cali_pred[state3_idx, yidx]

cigsale_matrix_with_california


# %%
data

# %%
# Create a DataFrame with year, actual cigsale of state 3, and imputed cigsale from cigsale_matrix_with_california
years = years_sorted
actual = state3_cigsale.values
imputed = cigsale_matrix_with_california.loc[3, years].values

comparison_df = pd.DataFrame({
    'year': years,
    'actual_cigsale': actual,
    'imputed_cigsale': imputed
})

comparison_df

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(comparison_df['year'], comparison_df['actual_cigsale'], color='black', label='Actual', linestyle='-')
plt.plot(comparison_df['year'], comparison_df['imputed_cigsale'], color='black', label='Imputed', linestyle='--')
plt.xlabel('Year')
plt.ylabel('Cigarette Sales')
plt.title('Actual vs Imputed Cigarette Sales for State 3')
plt.legend()
plt.axvline(x=1988, color='red', linestyle=':', linewidth=2)
plt.text(1988 + 0.2, plt.ylim()[1]*0.95, "Passage of Proposition 99", color='red', rotation=90, va='top')
plt.tight_layout()
plt.show()

# %%
# Placebo test: repeat synthetic control for states 9, 4, 36, 6, 39 as "treated"
placebo_results = {}

for placebo_state in [36, 6, 39]:
    # 1. Exclude state 3 from the donor pool
    placebo_states = [s for s in states_by_rank if s != 3 and s != placebo_state]
    # 2. Build cigsale matrix for placebo donor pool
    placebo_matrix = subset[subset['state'].isin(placebo_states + [placebo_state])].pivot_table(
        index='state', columns='year', values='cigsale'
    )
    placebo_matrix = placebo_matrix.reindex(index=[placebo_state] + placebo_states, columns=years_sorted)
    # 3. Mask post-1989 for placebo_state
    placebo_matrix.loc[placebo_state, 1989:] = np.nan
    # 4. Prepare data for model
    placebo_np = placebo_matrix.values.astype(np.float32)
    placebo_mask = ~np.isnan(placebo_np)
    placebo_mean = np.nanmean(placebo_np)
    placebo_np_filled = np.where(placebo_mask, placebo_np, placebo_mean)
    # 5. Randomly mask 20% for training
    placebo_rng = np.random.rand(*placebo_mask.shape)
    placebo_train_mask = (placebo_rng > 0.2) & placebo_mask
    placebo_input_tensor = placebo_np_filled[None, ..., None]
    placebo_target_tensor = placebo_np_filled[None, ..., None]
    placebo_train_mask_tensor = placebo_train_mask[None, ..., None]
    # 6. Build and train model
    placebo_model = keras.Sequential([
        layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=input_tensor.shape[1:]),
        layers.BatchNormalization(),
        layers.Conv2D(1, 3, padding='same')
    ])
    class PlaceboMaskedLoss(keras.losses.Loss):
        def call(self, y_true, y_pred):
            mask = tf.cast(placebo_train_mask_tensor, tf.float32)
            return tf.reduce_sum(mask * tf.square(y_true - y_pred)) / tf.reduce_sum(mask)
    placebo_model.compile(optimizer='adam', loss=PlaceboMaskedLoss())
    placebo_model.fit(placebo_input_tensor, placebo_target_tensor, epochs=500, verbose=0)
    # 7. Predict
    placebo_pred = placebo_model.predict(placebo_input_tensor)[0, ..., 0]
    # 8. Fill in masked values for placebo_state, years >= 1989
    placebo_idx = 0  # first row is placebo_state
    placebo_year_indices = [i for i, y in enumerate(years_sorted) if y >= 1989]
    placebo_matrix_filled = placebo_matrix.copy()
    for yidx in placebo_year_indices:
        placebo_matrix_filled.iloc[placebo_idx, yidx] = placebo_pred[placebo_idx, yidx]
    # 9. Collect actual and imputed for placebo_state
    actual = subset[(subset['state'] == placebo_state) & (subset['year'].isin(years_sorted))].sort_values('year')['cigsale'].values

    imputed = placebo_matrix_filled.loc[placebo_state, years_sorted].values
    placebo_results[placebo_state] = pd.DataFrame({
        'year': years_sorted,
        'actual_cigsale': actual,
        'imputed_cigsale': imputed
    })


# %%
actual = subset[(subset['state'] == placebo_state) & (subset['year'].isin(years_sorted))].sort_values('year')['cigsale'].values
actual

# %%
plt.figure(figsize=(10, 6))
plt.plot(placebo_results[36]['year'], placebo_results[36]['actual_cigsale'], label='Actual', color='black', linestyle='-')
plt.plot(placebo_results[36]['year'], placebo_results[36]['imputed_cigsale'], label='Imputed', color='black', linestyle='--')
plt.xlabel('Year')
plt.ylabel('Cigarette Sales')
plt.title('Actual vs Imputed Cigarette Sales for State 36 (Placebo)')
plt.legend()
plt.axvline(x=1988, color='red', linestyle=':', linewidth=2)
plt.text(1988 + 0.2, plt.ylim()[1]*0.95, "Placebo Treatment", color='red', rotation=90, va='top')
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(10, 6))
plt.plot(placebo_results[6]['year'], placebo_results[6]['actual_cigsale'], label='Actual', color='black', linestyle='-')
plt.plot(placebo_results[6]['year'], placebo_results[6]['imputed_cigsale'], label='Imputed', color='black', linestyle='--')
plt.xlabel('Year')
plt.ylabel('Cigarette Sales')
plt.title('Actual vs Imputed Cigarette Sales for State 6 (Placebo)')
plt.legend()
plt.axvline(x=1988, color='red', linestyle=':', linewidth=2)
plt.text(1988 + 0.2, plt.ylim()[1]*0.95, "Placebo Treatment", color='red', rotation=90, va='top')
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(10, 6))
plt.plot(placebo_results[39]['year'], placebo_results[39]['actual_cigsale'], label='Actual', color='black', linestyle='-')
plt.plot(placebo_results[39]['year'], placebo_results[39]['imputed_cigsale'], label='Imputed', color='black', linestyle='--')
plt.xlabel('Year')
plt.ylabel('Cigarette Sales')
plt.title('Actual vs Imputed Cigarette Sales for State 39 (Placebo)')
plt.legend()
plt.axvline(x=1988, color='red', linestyle=':', linewidth=2)
plt.text(1988 + 0.2, plt.ylim()[1]*0.95, "Placebo Treatment", color='red', rotation=90, va='top')
plt.tight_layout()
plt.show()

# %%



