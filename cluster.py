import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("creditcard.csv")

df = df.iloc[:, :3]

print("\nDATA BEFORE KMEANS:\n", df.head(10))

# Preprocessing
df = df.fillna(df.mean(numeric_only=True))
df = df.apply(lambda x: x.astype('category').cat.codes)

# Model
model = KMeans(n_clusters=3)

model.fit(df)

# Add cluster labels
df['Cluster'] = model.labels_

print("\nDATA AFTER KMEANS:\n", df.head(10))

# -------- PERFORMANCE MEASURE --------
score = silhouette_score(df.drop('Cluster', axis=1), df['Cluster'])

print("\nSilhouette Score:", score)