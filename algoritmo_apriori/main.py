from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# for market basket analysis
from mlxtend.frequent_patterns import apriori

# from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

# Configurar rutas
base_dir = Path(__file__).parent
input_dir = base_dir / "input"
output_dir = base_dir / "output"

# Crear directorio de salida si no existe
output_dir.mkdir(exist_ok=True)

# Leer datos de entrada
df = pd.read_csv(
    input_dir / "2.1.3 Market_Basket_Optimisation.csv", header=None
)

trx = []
data = {"transaccion": [], "producto": []}
for ix, row in df.iterrows():
    for item in row:
        data["producto"].append(item)
        data["transaccion"].append(ix + 1)
ndf = pd.DataFrame(data)


ndf = ndf.dropna(subset=["producto"])
ndf.head(25)

ndf.value_counts("producto")

# sns.histplot(data=ndf, x='producto', bins=15, kde=True)
plt.figure(figsize=(12, 4))  # 👈 más ancho que alto
order = ndf["producto"].value_counts().index
sns.countplot(x="producto", data=ndf, order=order)
plt.xticks(rotation=45)  # 👈 aquí rotas las etiquetas
plt.tight_layout()
plt.savefig(
    output_dir / "01_countplot_productos.png", dpi=300, bbox_inches="tight"
)
print(f"Gráfico guardado: {output_dir / '01_countplot_productos.png'}")
plt.close()

plt.figure(figsize=(12, 4))

order = ndf["producto"].value_counts().index
percentages = ndf["producto"].value_counts(normalize=True).reindex(order) * 100

sns.barplot(x=percentages.index, y=percentages.values)

plt.title("Distribución de Categorías (%)")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Porcentaje")

plt.tight_layout()
plt.savefig(
    output_dir / "02_barplot_porcentajes.png", dpi=300, bbox_inches="tight"
)
print(f"Gráfico guardado: {output_dir / '02_barplot_porcentajes.png'}")
plt.close()

trans = ndf.groupby("transaccion")["producto"].apply(list)
print(trans)

# https://rasbt.github.io/mlxtend/user_guide/preprocessing/TransactionEncoder/
te = TransactionEncoder()
tpdata = te.fit_transform(trans)
tpdata = pd.DataFrame(tpdata, columns=te.columns_)
print(tpdata)

freq_rules = apriori(tpdata, min_support=0.01, use_colnames=True)
print(freq_rules)

freq_rules["length"] = freq_rules["itemsets"].apply(lambda x: len(x))
print(freq_rules)

mask = freq_rules["length"] == 3
filtered_freq_rules = freq_rules.loc[mask]
filtered_freq_rules = filtered_freq_rules.sort_values(
    "support", ascending=False
)
print(filtered_freq_rules)

# ==============================================================================
# Visualización de la matriz tpdata (transacciones codificadas)
# ==============================================================================

# Opción 1: Muestra reducida (primeras transacciones + productos más frecuentes)
# Seleccionar los 30 productos más frecuentes
top_products = tpdata.sum().sort_values(ascending=False).head(30).index
sample_size = 100  # primeras 100 transacciones

tpdata_sample = tpdata.loc[: sample_size - 1, top_products]

plt.figure(figsize=(14, 8))
sns.heatmap(
    tpdata_sample.astype(int),  # Convertir bool a 0/1
    cmap=["white", "red"],  # False=blanco, True=rojo
    cbar=False,  # Sin barra de color
    xticklabels=True,
    yticklabels=False,  # Muchas transacciones, no mostrar todas
    linewidths=0,
)
plt.title(
    f"Matriz de Transacciones (Muestra: {sample_size} transacciones × {len(top_products)} productos más frecuentes)"
)
plt.xlabel("Productos")
plt.ylabel(f"Transacciones (0-{sample_size-1})")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(
    output_dir / "03_heatmap_transacciones_muestra.png",
    dpi=300,
    bbox_inches="tight",
)
print(
    f"Gráfico guardado: {output_dir / '03_heatmap_transacciones_muestra.png'}"
)
plt.close()

# Opción 2: Vista completa comprimida (sin etiquetas individuales)
plt.figure(figsize=(16, 10))
# Ordenar columnas por frecuencia
product_freq = tpdata.sum().sort_values(ascending=False)
tpdata_sorted = tpdata
# tpdata_sorted = tpdata[product_freq.index]

sns.heatmap(
    tpdata_sorted.astype(int),
    cmap=["white", "red"],
    cbar=False,  # Sin barra de color
    xticklabels=False,  # Demasiados productos
    yticklabels=False,  # Demasiadas transacciones
    linewidths=0,
)
plt.title(
    f"Matriz Completa de Transacciones ({tpdata.shape[0]} transacciones × {tpdata.shape[1]} productos)\nProductos ordenados por frecuencia (izq=más frecuente)"
)
plt.xlabel(f"Productos (1-{tpdata.shape[1]})")
plt.ylabel(f"Transacciones (1-{tpdata.shape[0]})")
plt.tight_layout()
plt.savefig(
    output_dir / "04_heatmap_transacciones_completo.png",
    dpi=300,
    bbox_inches="tight",
)
print(
    f"Gráfico guardado: {output_dir / '04_heatmap_transacciones_completo.png'}"
)
print(
    f"Dimensiones de la matriz: {tpdata.shape[0]} transacciones × {tpdata.shape[1]} productos"
)
plt.close()
