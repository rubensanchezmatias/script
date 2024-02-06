import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Read the CSV file
datos = pd.read_csv("breastcancer.csv")

# Remove non-predictors (id and last empty column)
df = datos.iloc[:, 1:-1]

# Convert diagnosis (B and M) to the values 0 and 1, respectively, and convert it to a factor
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0}).astype('category')

# Normalize the data
scaler = StandardScaler()
df.iloc[:, 1:] = scaler.fit_transform(df.iloc[:, 1:])

# Perform PCA
pca = PCA(n_components=len(df.columns)-1)
pca.fit(df.iloc[:, 1:])

# Print PCA results
print(pca)

# Scree plot
plt.plot(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_.cumsum(), marker='o')
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.savefig('plot.png')
plt.show()
