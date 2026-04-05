from abc import abstractmethod
from pathlib import Path
import os
from dotenv import load_dotenv
from datetime import datetime

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar variables de entorno desde .env
load_dotenv()


class AlgoRFM:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.output_dir = self.base_dir / "output"

        # Crear directorio de salida si no existe
        self.output_dir.mkdir(exist_ok=True)

        self.df = self.load_dataset()
        self.rfm_df = None

    @abstractmethod
    def load_dataset(self) -> pd.DataFrame:
        """
        Carga el dataset y devuelve un DataFrame con al menos:
        - customer_id: ID del cliente
        - fecha: Fecha de la transacción
        - monto: Monto de la compra
        """
        pass

    def run(self):
        cn = self.__class__.__name__

        # Calcular métricas RFM
        self.rfm_df = self.calculate_rfm(self.df)

        # Segmentar clientes
        self.rfm_df = self.segment_customers(self.rfm_df)

        # Visualizaciones
        self.plot_rfm_distribution(
            self.rfm_df, f"{cn}_01_rfm_distribution.png"
        )
        self.plot_segments(self.rfm_df, f"{cn}_02_segments.png")
        self.plot_rfm_heatmap(self.rfm_df, f"{cn}_03_rfm_heatmap.png")
        self.plot_segment_summary(
            self.rfm_df, f"{cn}_04_segment_summary.png"
        )

    def calculate_rfm(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula métricas RFM para cada cliente
        """
        print("Calculando métricas RFM...")

        # Fecha de referencia (última fecha + 1 día)
        reference_date = df["fecha"].max() + pd.Timedelta(days=1)

        rfm = (
            df.groupby("customer_id")
            .agg(
                {
                    "fecha": lambda x: (reference_date - x.max()).days,
                    "customer_id": "count",
                    "monto": "sum",
                }
            )
            .rename(
                columns={
                    "fecha": "Recency",
                    "customer_id": "Frequency",
                    "monto": "Monetary",
                }
            )
        )

        print("\nRFM Metrics:")
        print(rfm.describe())
        print()

        return rfm

    def segment_customers(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Segmenta clientes usando quartiles de RFM
        """
        print("Segmentando clientes...")

        # Calcular quartiles (invertir Recency: menor es mejor)
        rfm_df["R_Score"] = pd.qcut(
            rfm_df["Recency"], 4, labels=[4, 3, 2, 1]
        )
        rfm_df["F_Score"] = pd.qcut(
            rfm_df["Frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]
        )
        rfm_df["M_Score"] = pd.qcut(
            rfm_df["Monetary"], 4, labels=[1, 2, 3, 4]
        )

        # Score total
        rfm_df["RFM_Score"] = (
            rfm_df["R_Score"].astype(str)
            + rfm_df["F_Score"].astype(str)
            + rfm_df["M_Score"].astype(str)
        )

        # Segmentos basados en RFM Score
        def rfm_segment(row):
            r, f, m = int(row["R_Score"]), int(row["F_Score"]), int(
                row["M_Score"]
            )

            if r >= 4 and f >= 4 and m >= 4:
                return "Champions"
            elif r >= 3 and f >= 3:
                return "Loyal Customers"
            elif r >= 4 and f <= 2:
                return "Promising"
            elif r >= 3 and f <= 2 and m >= 3:
                return "Big Spenders"
            elif r <= 2 and f >= 3:
                return "At Risk"
            elif r <= 2 and f <= 2:
                return "Lost"
            else:
                return "Others"

        rfm_df["Segment"] = rfm_df.apply(rfm_segment, axis=1)

        print("\nDistribución de segmentos:")
        print(rfm_df["Segment"].value_counts())
        print()

        return rfm_df

    def plot_rfm_distribution(self, rfm_df: pd.DataFrame, fig_name: str):
        """
        Visualiza la distribución de Recency, Frequency y Monetary
        """
        plt.rcParams.update(
            {
                "font.size": 12,
                "axes.titlesize": 14,
                "axes.labelsize": 12,
            }
        )

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Recency
        axes[0].hist(rfm_df["Recency"], bins=30, color="skyblue", edgecolor="black")
        axes[0].set_title("Distribución de Recency")
        axes[0].set_xlabel("Días desde última compra")
        axes[0].set_ylabel("Cantidad de clientes")

        # Frequency
        axes[1].hist(
            rfm_df["Frequency"], bins=30, color="lightgreen", edgecolor="black"
        )
        axes[1].set_title("Distribución de Frequency")
        axes[1].set_xlabel("Número de compras")
        axes[1].set_ylabel("Cantidad de clientes")

        # Monetary
        axes[2].hist(
            rfm_df["Monetary"], bins=30, color="salmon", edgecolor="black"
        )
        axes[2].set_title("Distribución de Monetary")
        axes[2].set_xlabel("Valor total de compras")
        axes[2].set_ylabel("Cantidad de clientes")

        plt.tight_layout()
        plt.savefig(
            self.output_dir / fig_name, dpi=300, bbox_inches="tight"
        )
        plt.close()

    def plot_segments(self, rfm_df: pd.DataFrame, fig_name: str):
        """
        Visualiza los segmentos de clientes
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Pie chart de segmentos
        segment_counts = rfm_df["Segment"].value_counts()
        colors = plt.cm.Set3(range(len(segment_counts)))
        ax1.pie(
            segment_counts,
            labels=segment_counts.index,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
        )
        ax1.set_title("Distribución de segmentos de clientes")

        # Scatter de R vs F coloreado por segmento
        for segment in rfm_df["Segment"].unique():
            segment_data = rfm_df[rfm_df["Segment"] == segment]
            ax2.scatter(
                segment_data["Recency"],
                segment_data["Frequency"],
                label=segment,
                alpha=0.6,
                s=50,
            )

        ax2.set_xlabel("Recency (días)")
        ax2.set_ylabel("Frequency (compras)")
        ax2.set_title("Segmentos: Recency vs Frequency")
        ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / fig_name, dpi=300, bbox_inches="tight"
        )
        plt.close()

    def plot_rfm_heatmap(self, rfm_df: pd.DataFrame, fig_name: str):
        """
        Mapa de calor de correlación entre R, F, M
        """
        plt.figure(figsize=(8, 6))

        corr_matrix = rfm_df[["Recency", "Frequency", "Monetary"]].corr()

        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
        )

        plt.title("Correlación entre métricas RFM")
        plt.tight_layout()
        plt.savefig(
            self.output_dir / fig_name, dpi=300, bbox_inches="tight"
        )
        plt.close()

    def plot_segment_summary(self, rfm_df: pd.DataFrame, fig_name: str):
        """
        Tabla resumen de segmentos
        """
        # Calcular estadísticas por segmento
        summary = (
            rfm_df.groupby("Segment")
            .agg(
                {
                    "Recency": ["mean", "median"],
                    "Frequency": ["mean", "median"],
                    "Monetary": ["mean", "median"],
                }
            )
            .round(2)
        )

        # Agregar conteo
        summary["Count"] = rfm_df.groupby("Segment").size()

        # Resetear índice para tabla
        summary_display = summary.reset_index()

        # Crear figura
        fig, ax = plt.subplots(figsize=(14, len(summary_display) * 0.6 + 2))
        ax.axis("tight")
        ax.axis("off")

        # Formatear columnas
        col_labels = [
            "Segment",
            "R_Mean",
            "R_Median",
            "F_Mean",
            "F_Median",
            "M_Mean",
            "M_Median",
            "Count",
        ]

        # Crear tabla
        table = ax.table(
            cellText=summary_display.values,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
            colWidths=[0.15] * len(col_labels),
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Estilizar encabezados
        for i in range(len(col_labels)):
            cell = table[(0, i)]
            cell.set_facecolor("#40466e")
            cell.set_text_props(weight="bold", color="white")

        # Alternar colores en filas
        for i in range(1, len(summary_display) + 1):
            for j in range(len(col_labels)):
                cell = table[(i, j)]
                if i % 2 == 0:
                    cell.set_facecolor("#f0f0f0")
                else:
                    cell.set_facecolor("white")

        plt.title(
            "Resumen de segmentos RFM", fontsize=14, weight="bold", pad=20
        )

        plt.savefig(
            self.output_dir / fig_name, dpi=300, bbox_inches="tight"
        )
        plt.close()


class AlgoOnlineRetailRFM(AlgoRFM):
    """
    Análisis RFM usando el dataset Online Retail de Kaggle
    """

    def __init__(self):
        super().__init__()

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

        # Limpiar y preparar datos
        df = df[["CustomerID", "InvoiceDate", "Quantity", "UnitPrice"]].copy()
        df = df.dropna(subset=["CustomerID"])
        df["CustomerID"] = df["CustomerID"].astype(int)

        # Calcular monto total
        df["monto"] = df["Quantity"] * df["UnitPrice"]

        # Filtrar montos positivos
        df = df[df["monto"] > 0]

        # Convertir fecha
        df["fecha"] = pd.to_datetime(df["InvoiceDate"])

        # Renombrar para compatibilidad
        df = df.rename(columns={"CustomerID": "customer_id"})

        df = df[["customer_id", "fecha", "monto"]]

        print("Dataset Online Retail para RFM:")
        print(df.head(20))
        print(df.info())
        print()

        return df


class AlgoSyntheticRFM(AlgoRFM):
    """
    Análisis RFM usando datos sintéticos para demostración
    """

    def __init__(self):
        super().__init__()

    def load_dataset(self):
        print("Generando dataset sintético para RFM...")

        np.random.seed(42)
        n_customers = 500
        n_transactions = 3000

        # Generar transacciones
        data = {
            "customer_id": np.random.randint(1, n_customers + 1, n_transactions),
            "fecha": pd.date_range(
                end=datetime.now(), periods=n_transactions, freq="H"
            )
            + pd.to_timedelta(np.random.randint(0, 365, n_transactions), unit="D"),
            "monto": np.random.gamma(shape=2, scale=50, size=n_transactions),
        }

        df = pd.DataFrame(data)
        df = df.sort_values("fecha")

        print("Dataset sintético generado:")
        print(df.head(20))
        print(df.info())
        print()

        return df


if __name__ == "__main__":
    # Ejemplo con datos sintéticos (siempre disponible)
    algo1 = AlgoSyntheticRFM()
    algo1.run()

    # Ejemplo con Online Retail (requiere .env configurado)
    # algo2 = AlgoOnlineRetailRFM()
    # algo2.run()
