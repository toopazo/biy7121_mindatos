"""
Ejemplos de uso del algoritmo Apriori
"""

from abc import abstractmethod
from pathlib import Path
import os
from dotenv import load_dotenv

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# for market basket analysis
from mlxtend.frequent_patterns import apriori

# from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

# Cargar variables de entorno desde .env
load_dotenv()


class Algo:
    """Clase base para algoritmos de análisis de transacciones"""

    def __init__(self):
        self.base_dir = Path(__file__).parent
        # self.input_dir = self.base_dir / "input"
        self.output_dir = self.base_dir / "output"

        # Crear directorio de salida si no existe
        self.output_dir.mkdir(exist_ok=True)

        # # Borrar contedio del directorio de salida
        # for file in self.output_dir.glob("*"):
        #     file.unlink()

        self.df = self.load_dataset()

    @abstractmethod
    def load_dataset(self) -> pd.DataFrame:
        """
        Carga el dataset y devuelve un DataFrame con al menos las columnas:
        - transaccion: ID de la transacción
        - producto: Nombre del producto comprado
        """
        pass

    def run(self):
        # print(f"Directorio de entrada: {self.input_dir}")
        # print(f"Directorio de salida: {self.output_dir}")
        # print()

        self.df.value_counts("producto")

        cn = self.__class__.__name__

        self.plot_distribution(self.df, f"{cn}_01_distribucion_productos.png")

        tpdata, filtered_freq_rules = self.apriori_analysis(self.df)
        self.plot_patterns(tpdata, f"{cn}_02_patron_transacciones.png")
        self.plot_frequent_itemsets(
            filtered_freq_rules, f"{cn}_03_frequent_itemsets.png"
        )

    def apriori_analysis(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        trans = df.groupby("transaccion")["producto"].apply(list)
        print("Transacciones agrupadas por ID de transacción:")
        print(trans)

        # https://rasbt.github.io/mlxtend/user_guide/preprocessing/TransactionEncoder/
        te = TransactionEncoder()
        tpdata = te.fit_transform(trans)
        tpdata = pd.DataFrame(tpdata, columns=te.columns_)
        print("TransactionEncoder")
        print(tpdata.shape)
        print(tpdata)

        min_support = 0.01
        freq_rules = apriori(tpdata, min_support=min_support, use_colnames=True)
        # print("Frequent Itemsets")
        # print(freq_rules)

        freq_rules["length"] = freq_rules["itemsets"].apply(lambda x: len(x))
        print(f"Frequent Itemsets (min_support {min_support})")
        print(freq_rules.shape)
        print(freq_rules)

        mask = freq_rules["length"] == 3
        filtered_freq_rules = freq_rules.loc[mask]
        filtered_freq_rules = filtered_freq_rules.sort_values(
            "support", ascending=False
        )
        print(f"Filtered Frequent Itemsets (min_support {min_support})")
        print(filtered_freq_rules.shape)
        print(filtered_freq_rules)

        return tpdata, filtered_freq_rules

    def plot_patterns(self, tpdata: pd.DataFrame, fig_name: str):
        # ==============================================================================
        # Visualización de la matriz tpdata (transacciones codificadas)
        # ==============================================================================

        # # Opción 1: Muestra reducida (primeras transacciones + productos más frecuentes)
        # # Seleccionar los 30 productos más frecuentes
        # top_products = tpdata.sum().sort_values(ascending=False).head(30).index
        # sample_size = 100  # primeras 100 transacciones

        # tpdata_sample = tpdata.loc[: sample_size - 1, top_products]

        # plt.figure(figsize=(14, 8))
        # sns.heatmap(
        #     tpdata_sample.astype(int),  # Convertir bool a 0/1
        #     cmap=["white", "red"],  # False=blanco, True=rojo
        #     cbar=False,  # Sin barra de color
        #     xticklabels=True,
        #     yticklabels=False,  # Muchas transacciones, no mostrar todas
        #     linewidths=0,
        # )
        # plt.title(
        #     f"Matriz de Transacciones (Muestra: {sample_size} transacciones × {len(top_products)} productos más frecuentes)"
        # )
        # plt.xlabel("Productos")
        # plt.ylabel(f"Transacciones (0-{sample_size-1})")
        # plt.xticks(rotation=45, ha="right")
        # plt.tight_layout()
        # plt.savefig(
        #     self.output_dir / fig_name,
        #     dpi=300,
        #     bbox_inches="tight",
        # )
        # print(f"Gráfico guardado: {self.output_dir / fig_name}")
        # plt.close()

        # Opción 2: Vista completa comprimida (sin etiquetas individuales)
        plt.figure(figsize=(16, 10))
        # Ordenar columnas por frecuencia
        product_freq = tpdata.sum().sort_values(ascending=False)
        tpdata_sorted = tpdata
        # tpdata_sorted = tpdata[product_freq.index]
        _ = product_freq

        sns.heatmap(
            tpdata_sorted.astype(int),
            cmap=["white", "red"],
            cbar=False,  # Sin barra de color
            xticklabels=False,  # Demasiados productos
            yticklabels=False,  # Demasiadas transacciones
            linewidths=0,
        )
        plt.title(
            f"Matriz Completa de Transacciones ({tpdata.shape[0]} "
            f"transacciones x {tpdata.shape[1]} productos)\nProductos ordenados"
            " por frecuencia (izq=más frecuente)"
        )
        plt.xlabel(f"Productos (1-{tpdata.shape[1]})")
        plt.ylabel(f"Transacciones (1-{tpdata.shape[0]})")
        plt.tight_layout()
        plt.savefig(
            self.output_dir / fig_name,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def plot_frequent_itemsets(self, freq_rules: pd.DataFrame, fig_name: str):
        """
        Genera una imagen PNG con la tabla de frequent itemsets
        """
        # Crear una copia para formatear sin modificar el original
        df_display = freq_rules.copy()

        # Convertir itemsets de frozenset a string legible
        df_display["itemsets"] = df_display["itemsets"].apply(
            lambda x: ", ".join(sorted(list(x)))
        )

        # Formatear support a 4 decimales
        df_display["support"] = df_display["support"].apply(
            lambda x: f"{x:.4f}"
        )

        # Resetear índice para mostrarlo como columna
        df_display = df_display.reset_index(drop=True)

        # Calcular dimensiones de la figura según cantidad de filas
        num_rows = len(df_display)
        row_height = 0.4
        fig_height = max(6, (num_rows + 1) * row_height)

        # Crear figura
        fig, ax = plt.subplots(figsize=(16, fig_height))
        ax.axis("tight")
        ax.axis("off")
        _ = fig

        # Crear tabla
        table = ax.table(
            cellText=df_display.values,
            colLabels=df_display.columns,
            cellLoc="left",
            loc="center",
            colWidths=[0.6, 0.2, 0.2],  # Anchos relativos de columnas
        )

        # Estilizar tabla
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)  # Escalar para mejor legibilidad

        # Estilizar encabezados
        for i in range(len(df_display.columns)):
            cell = table[(0, i)]
            cell.set_facecolor("#40466e")
            cell.set_text_props(weight="bold", color="white")

        # Alternar colores en filas
        for i in range(1, len(df_display) + 1):
            for j in range(len(df_display.columns)):
                cell = table[(i, j)]
                if i % 2 == 0:
                    cell.set_facecolor("#f0f0f0")
                else:
                    cell.set_facecolor("white")

        plt.title(
            f"Frequent Itemsets (Total: {num_rows} itemsets)",
            fontsize=14,
            weight="bold",
            pad=20,
        )

        plt.savefig(self.output_dir / fig_name, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_distribution(self, df: pd.DataFrame, fig_name: str):
        # Configurar tamaños de fuente
        plt.rcParams.update(
            {
                "font.size": 12,
                "axes.titlesize": 14,
                "axes.labelsize": 12,
                "xtick.labelsize": 11,
                "ytick.labelsize": 11,
            }
        )

        # Crear figura con 2 subplots verticales
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9))
        _ = fig

        # Subplot 1: Countplot de productos
        order = df["producto"].value_counts().index
        counts = df["producto"].value_counts().values
        info = f"num {len(order)} max {order[0]} {counts[0]} "
        info += f"min {order[-1]} {counts[-1]}"

        sns.countplot(x="producto", data=df, order=order, ax=ax1)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90, ha="right")
        ax1.set_title(f"Conteo - Absoluto - {info}")

        # Subplot 2: Barplot de porcentajes
        percentages = (
            df["producto"].value_counts(normalize=True).reindex(order) * 100
        )
        info = f"num {len(percentages)} max {order[0]} {percentages.max():.2f}%"
        info += f" min {order[-1]} {percentages.min():.2f}%"

        sns.barplot(x=percentages.index, y=percentages.values, ax=ax2)
        ax2.set_title(f"Conteo - Porcentaje - {info}")
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=90, ha="right")
        ax2.set_ylabel("Porcentaje")

        # Guardar figura completa
        plt.tight_layout()
        plt.savefig(
            self.output_dir / fig_name,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


class AlgoOnlineRetailDataset(Algo):
    """Algoritmo para analizar el dataset Online Retail"""

    def load_dataset(self):
        csv_path_str = os.getenv("ONLINE_RETAIL_CSV")
        if csv_path_str is None:
            raise ValueError(
                "Variable de entorno ONLINE_RETAIL_CSV no encontrada. "
                "Crea un archivo .env con la ruta al dataset."
            )

        csv_path = Path(csv_path_str)
        assert csv_path.exists(), f"Archivo no encontrado: {csv_path}"

        df = pd.read_csv(csv_path)

        df = df[["InvoiceNo", "StockCode"]].copy()
        df = df.rename(
            columns={"InvoiceNo": "transaccion", "StockCode": "producto"}
        )

        print("Dataset Online Retail:")
        print(df.head(30))
        print(df.info())
        print()

        return df


class AlgoMarketBasketDataset(Algo):
    """Algoritmo para analizar el dataset Market Basket"""

    def load_dataset(self):

        csv_path = Path(__file__).parent / Path(
            "input/2.1.3 Market_Basket_Optimisation.csv"
        )
        assert csv_path.exists(), f"Archivo no encontrado: {csv_path}"

        df = pd.read_csv(csv_path, header=None)

        data: dict[str, list] = {"transaccion": [], "producto": []}
        for ix, row in df.iterrows():
            for item in row:
                data["producto"].append(item)
                data["transaccion"].append(int(ix) + 1)
        ndf = pd.DataFrame(data)

        ndf = ndf.dropna(subset=["producto"])

        print("Dataset Market Basket:")
        print(ndf.head(30))
        print(ndf.info())
        print()

        return ndf


if __name__ == "__main__":
    algo1 = AlgoMarketBasketDataset()
    algo1.run()

    algo2 = AlgoOnlineRetailDataset()
    algo2.run()
