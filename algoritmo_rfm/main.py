from abc import abstractmethod
from pathlib import Path
import os
from dotenv import load_dotenv
from datetime import datetime

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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
        self.plot_rfm_distribution(self.rfm_df, f"{cn}_01_rfm_distribution.png")
        self.plot_segments(self.rfm_df, f"{cn}_02_segments.png")
        # self.plot_rfm_heatmap(self.rfm_df, f"{cn}_03_rfm_heatmap.png")
        # self.plot_segment_summary(self.rfm_df, f"{cn}_04_segment_summary.png")
        self.plot_rfm_score_grid(self.rfm_df, f"{cn}_03_rfm_score_grid.png")
        self.plot_3d_rfm_score_grid(
            self.rfm_df, f"{cn}_04_3d_rfm_score_grid.png"
        )
        self.plot_3d_rfm_monetary_grid(
            self.rfm_df, f"{cn}_05_3d_rfm_monetary_grid.png"
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
        rfm_df["R_Score"] = pd.qcut(rfm_df["Recency"], 4, labels=[4, 3, 2, 1])
        rfm_df["F_Score"] = pd.qcut(
            rfm_df["Frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]
        )
        rfm_df["M_Score"] = pd.qcut(rfm_df["Monetary"], 4, labels=[1, 2, 3, 4])

        # Score total
        rfm_df["RFM_Score"] = (
            rfm_df["R_Score"].astype(str)
            + rfm_df["F_Score"].astype(str)
            + rfm_df["M_Score"].astype(str)
        )

        rfm_df["Segment"] = rfm_df.apply(self.rfm_segment, axis=1)

        print("\nDistribución de segmentos:")
        print(rfm_df["Segment"].value_counts())
        print()

        return rfm_df

    def rfm_segment(self, row: pd.Series) -> str:
        r, f, m = (
            int(row["R_Score"]),
            int(row["F_Score"]),
            int(row["M_Score"]),
        )

        if r == 4 and f == 4:
            return "44X - Campeones"
        elif r == 1 and f == 4:
            return "14X - En riesgo"
        elif r == 4 and f == 1:
            return "41X - Nuevos"
        elif r == 1 and f == 1:
            return "11X - Perdidos"
        else:
            return "Otros"

    def plot_rfm_distribution(self, rfm_df: pd.DataFrame, fig_name: str):
        """
        Visualiza la distribución de Recency, Frequency y Monetary,
        marcando sus cuartiles con líneas verticales.
        """
        plt.rcParams.update(
            {
                "font.size": 12,
                "axes.titlesize": 14,
                "axes.labelsize": 12,
            }
        )

        _, axes = plt.subplots(1, 3, figsize=(16, 5))
        plot_config = [
            (
                "Recency",
                "skyblue",
                "Distribución de Recency",
                "Días desde última compra",
            ),
            (
                "Frequency",
                "lightgreen",
                "Distribución de Frequency",
                "Número de compras",
            ),
            (
                "Monetary",
                "salmon",
                "Distribución de Monetary",
                "Valor total de compras",
            ),
        ]
        quartile_styles = [
            (0.25, "Q1 (25%)", "-"),
            (0.50, "Q2 (50%)", "--"),
            (0.75, "Q3 (75%)", ":"),
            (1.00, "Q4 (100%)", "-."),
        ]

        for ax, (column, color, title, xlabel) in zip(axes, plot_config):
            ax.hist(rfm_df[column], bins=30, color=color, edgecolor="black")
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Cantidad de clientes")

            for quantile, label, linestyle in quartile_styles:
                value = rfm_df[column].quantile(quantile)
                ax.axvline(
                    value,
                    color="crimson",
                    linestyle=linestyle,
                    linewidth=2,
                    label=f"{label}: {value:.2f}",
                )

            ax.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(self.output_dir / fig_name, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_segments(self, rfm_df: pd.DataFrame, fig_name: str):
        """
        Visualiza los segmentos de clientes
        """
        _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

        # Pie chart y scatter con un mapeo de colores consistente
        segment_counts = rfm_df["Segment"].value_counts()
        segment_order = list(segment_counts.index)
        palette = sns.color_palette("Set3", n_colors=len(segment_order))
        color_map = dict(zip(segment_order, palette))

        ax1.pie(
            segment_counts,
            labels=None,
            autopct="%1.0f%%",
            pctdistance=1.20,
            colors=[color_map[segment] for segment in segment_counts.index],
            startangle=90,
        )
        ax1.set_title("Distribución de segmentos de clientes")

        # Scatter de R vs F coloreado por segmento usando el mismo color_map
        for segment in segment_order:
            segment_data = rfm_df[rfm_df["Segment"] == segment]
            ax2.scatter(
                segment_data["Recency"],
                segment_data["Frequency"],
                label=segment,
                color=color_map[segment],
                alpha=0.6,
                s=50,
            )

        ax2.set_xlabel("Recency (días)")
        ax2.set_ylabel("Frequency (compras)")
        ax2.set_title("Segmentos: Recency vs Frequency")
        # ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax2.grid(True, alpha=0.3)

        # Scatter de R vs F coloreado por segmento usando el mismo color_map
        for segment in segment_order:
            segment_data = rfm_df[rfm_df["Segment"] == segment]
            ax3.scatter(
                segment_data["Recency"],
                segment_data["Frequency"],
                label=segment,
                color=color_map[segment],
                alpha=0.6,
                s=50,
            )

        ax3.set_xlabel("Recency (días)")
        ax3.set_ylabel("Frequency (compras)")
        ax3.set_title("Segmentos: Recency vs Frequency (log scale)")
        ax3.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale("log")

        plt.tight_layout()
        plt.savefig(self.output_dir / fig_name, dpi=300, bbox_inches="tight")
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
        plt.savefig(self.output_dir / fig_name, dpi=300, bbox_inches="tight")
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

        plt.savefig(self.output_dir / fig_name, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_rfm_score_grid(self, rfm_df: pd.DataFrame, fig_name: str):
        """
        Visualiza la cantidad de clientes por combinación de R y F para cada
        valor de M_Score.
        """
        score_df = rfm_df.copy()
        for col in ["R_Score", "F_Score", "M_Score"]:
            score_df[col] = score_df[col].astype(int)

        count_series = score_df.groupby(
            ["M_Score", "R_Score", "F_Score"]
        ).size()
        vmax = int(count_series.max()) if not count_series.empty else 0

        fig, axes = plt.subplots(
            2, 2, figsize=(14, 12), sharex=True, sharey=True
        )

        for ax, m_score in zip(axes.flat, [1, 2, 3, 4]):
            matrix = (
                score_df[score_df["M_Score"] == m_score]
                .groupby(["R_Score", "F_Score"])
                .size()
                .unstack(fill_value=0)
                .reindex(
                    index=[4, 3, 2, 1],
                    columns=[1, 2, 3, 4],
                    fill_value=0,
                )
            )

            sns.heatmap(
                matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                linewidths=0.5,
                linecolor="white",
                square=True,
                vmin=0,
                vmax=vmax,
                ax=ax,
            )
            ax.set_title(f"M = {m_score}")
            ax.set_xlabel("F_Score")
            ax.set_ylabel("R_Score")

        fig.suptitle(
            "Cantidad de clientes por combinación RFM",
            fontsize=14,
            weight="bold",
        )
        plt.tight_layout(rect=(0, 0, 1, 0.97))
        plt.savefig(self.output_dir / fig_name, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_3d_rfm_score_grid(self, rfm_df: pd.DataFrame, fig_name: str):
        """
        Genera una versión estática PNG y otra interactiva HTML de la matriz
        RFM 4x4x4 usando F, R y M como ejes. La versión HTML se abre en el
        navegador para permitir rotar, hacer zoom y explorar el gráfico.
        """
        score_df = rfm_df.copy()
        for col in ["R_Score", "F_Score", "M_Score"]:
            score_df[col] = score_df[col].astype(int)

        grid = (
            score_df.groupby(["R_Score", "F_Score", "M_Score"])
            .size()
            .reindex(
                pd.MultiIndex.from_product(
                    [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
                    names=["R_Score", "F_Score", "M_Score"],
                ),
                fill_value=0,
            )
            .reset_index(name="Count")
        )

        max_count = max(int(grid["Count"].max()), 1)
        html_path = self.output_dir / Path(fig_name).with_suffix(".html").name
        interactive_sizes = 10 + (grid["Count"] / max_count) * 30
        interactive_text = [
            str(int(count)) if int(count) > 0 else "" for count in grid["Count"]
        ]

        interactive_fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=grid["F_Score"],
                    y=grid["R_Score"],
                    z=grid["M_Score"],
                    mode="markers+text",
                    text=interactive_text,
                    textposition="top center",
                    customdata=grid[["Count"]],
                    marker={
                        "size": interactive_sizes,
                        "color": grid["Count"],
                        "colorscale": "Viridis",
                        "opacity": 0.85,
                        "line": {"color": "black", "width": 1},
                        "colorbar": {"title": "Clientes"},
                    },
                    hovertemplate=(
                        "F_Score=%{x}<br>"
                        "R_Score=%{y}<br>"
                        "M_Score=%{z}<br>"
                        "Clientes=%{customdata[0]}<extra></extra>"
                    ),
                )
            ]
        )
        interactive_fig.update_layout(
            title="Matriz RFM 4x4x4 interactiva",
            scene={
                "xaxis": {"title": "F_Score", "tickvals": [1, 2, 3, 4]},
                "yaxis": {
                    "title": "R_Score",
                    "tickvals": [1, 2, 3, 4],
                    "autorange": "reversed",
                },
                "zaxis": {"title": "M_Score", "tickvals": [1, 2, 3, 4]},
            },
            margin={"l": 0, "r": 0, "b": 0, "t": 50},
        )

        auto_open_browser = os.environ.get("RFM_NO_BROWSER") != "1"
        interactive_fig.write_html(
            str(html_path),
            include_plotlyjs="cdn",
            auto_open=auto_open_browser,
        )
        print(f"Gráfico 3D interactivo guardado en: {html_path}")

    def plot_3d_rfm_monetary_grid(self, rfm_df: pd.DataFrame, fig_name: str):
        """
        Genera una visualización 3D interactiva donde cada burbuja representa
        la suma de dinero (`Monetary`) por combinación de scores R, F y M.
        """
        score_df = rfm_df.copy()
        for col in ["R_Score", "F_Score", "M_Score"]:
            score_df[col] = score_df[col].astype(int)

        totals = (
            score_df.groupby(["R_Score", "F_Score", "M_Score"], as_index=False)[
                "Monetary"
            ]
            .sum()
            .rename(columns={"Monetary": "TotalMonetary"})
        )

        grid = (
            pd.MultiIndex.from_product(
                [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
                names=["R_Score", "F_Score", "M_Score"],
            )
            .to_frame(index=False)
            .merge(totals, on=["R_Score", "F_Score", "M_Score"], how="left")
            .fillna({"TotalMonetary": 0.0})
        )

        max_total = max(float(grid["TotalMonetary"].max()), 1.0)
        html_path = self.output_dir / Path(fig_name).with_suffix(".html").name
        marker_sizes = 10 + (grid["TotalMonetary"] / max_total) * 35
        interactive_text = [
            f"${value:,.0f}" if value > 0 else ""
            for value in grid["TotalMonetary"]
        ]

        interactive_fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=grid["F_Score"],
                    y=grid["R_Score"],
                    z=grid["M_Score"],
                    mode="markers+text",
                    text=interactive_text,
                    textposition="top center",
                    customdata=grid[["TotalMonetary"]].to_numpy(),
                    marker={
                        "size": marker_sizes,
                        "color": grid["TotalMonetary"],
                        "colorscale": "Plasma",
                        "opacity": 0.85,
                        "line": {"color": "black", "width": 1},
                        "colorbar": {"title": "Monto total"},
                    },
                    hovertemplate=(
                        "F_Score=%{x}<br>"
                        "R_Score=%{y}<br>"
                        "M_Score=%{z}<br>"
                        "Monto total=$%{customdata[0]:,.2f}<extra></extra>"
                    ),
                )
            ]
        )
        interactive_fig.update_layout(
            title="Matriz RFM 4x4x4 interactiva por monto total",
            scene={
                "xaxis": {"title": "F_Score", "tickvals": [1, 2, 3, 4]},
                "yaxis": {
                    "title": "R_Score",
                    "tickvals": [1, 2, 3, 4],
                    "autorange": "reversed",
                },
                "zaxis": {"title": "M_Score", "tickvals": [1, 2, 3, 4]},
            },
            margin={"l": 0, "r": 0, "b": 0, "t": 50},
        )

        auto_open_browser = os.environ.get("RFM_NO_BROWSER") != "1"
        interactive_fig.write_html(
            str(html_path),
            include_plotlyjs="cdn",
            auto_open=auto_open_browser,
        )
        print(f"Gráfico 3D monetario guardado en: {html_path}")


class AlgoOnlineRetailDataset(AlgoRFM):
    """
    Análisis RFM usando el dataset Online Retail de Kaggle
    """

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

    def load_dataset(self):
        print("Generando dataset sintético para RFM...")

        np.random.seed(42)
        n_customers = 500
        n_transactions = 3000

        # Generar transacciones
        data = {
            "customer_id": np.random.randint(
                1, n_customers + 1, n_transactions
            ),
            "fecha": pd.date_range(
                end=datetime.now(), periods=n_transactions, freq="h"
            )
            + pd.to_timedelta(
                np.random.randint(0, 365, n_transactions), unit="D"
            ),
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
    algo2 = AlgoOnlineRetailDataset()
    algo2.run()
