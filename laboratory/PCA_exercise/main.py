import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

column_names = [
    'Tipo de Cliente',
    'Región',
    'Gasto anual en productos frescos',
    'Gasto anual en productos lácteos',
    'Gasto anual en productos de ultramarinos',
    'Gasto anual en productos congelados',
    'Gasto anual en detergentes y productos de papelería',
    'Gasto anual en productos delicatessen'
]

df = pd.read_csv("DatosCluster.csv", names=column_names, header=None)

cat_vars = ["Tipo de Cliente", "Región"]
num_vars = [col for col in df.columns if col not in cat_vars]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_vars),
        ('cat', OneHotEncoder(), cat_vars)])

X_preprocessed = preprocessor.fit_transform(df)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_preprocessed)

optimal_clusters = 4

kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_pca)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis")
plt.xlabel("Componente principal 1")
plt.ylabel("Componente principal 2")
plt.title("Clustering de clientes")
plt.colorbar(label="Cluster")
plt.show()

df['Cluster'] = clusters

cluster_means = df.groupby('Cluster').mean()

print(cluster_means)
