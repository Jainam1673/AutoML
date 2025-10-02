"""Interactive dashboard for AutoML experimentation.

Streamlit-based UI for model training, monitoring, and analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

__all__ = ["AutoMLDashboard", "run_dashboard"]


class AutoMLDashboard:
    """Interactive dashboard for AutoML."""
    
    def __init__(self, experiment_dir: Path | None = None) -> None:
        """Initialize dashboard.
        
        Args:
            experiment_dir: Directory with experiment results
        """
        self.experiment_dir = experiment_dir or Path("./experiments")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
    
    def load_experiments(self) -> list[dict[str, Any]]:
        """Load all experiments.
        
        Returns:
            List of experiment metadata
        """
        experiments = []
        
        for exp_file in self.experiment_dir.glob("*/metadata.json"):
            with open(exp_file) as f:
                metadata = json.load(f)
                metadata["path"] = exp_file.parent
                experiments.append(metadata)
        
        return experiments
    
    def get_experiment_metrics(self, experiment_id: str) -> pd.DataFrame:
        """Get metrics for an experiment.
        
        Args:
            experiment_id: Experiment ID
        
        Returns:
            DataFrame with metrics
        """
        metrics_file = self.experiment_dir / experiment_id / "metrics.csv"
        
        if metrics_file.exists():
            return pd.read_csv(metrics_file)
        
        return pd.DataFrame()
    
    def compare_experiments(
        self,
        experiment_ids: list[str],
        metric: str = "accuracy",
    ) -> pd.DataFrame:
        """Compare multiple experiments.
        
        Args:
            experiment_ids: List of experiment IDs
            metric: Metric to compare
        
        Returns:
            Comparison DataFrame
        """
        comparisons = []
        
        for exp_id in experiment_ids:
            metrics = self.get_experiment_metrics(exp_id)
            if not metrics.empty and metric in metrics.columns:
                comparisons.append({
                    "experiment_id": exp_id,
                    "best_score": metrics[metric].max(),
                    "mean_score": metrics[metric].mean(),
                    "std_score": metrics[metric].std(),
                })
        
        return pd.DataFrame(comparisons)


def create_streamlit_app() -> None:
    """Create Streamlit dashboard application."""
    try:
        import streamlit as st
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError(
            "Streamlit and Plotly required for dashboard: "
            "pip install streamlit plotly"
        )
    
    st.set_page_config(
        page_title="AutoML Dashboard",
        page_icon="🤖",
        layout="wide",
    )
    
    st.title("🤖 AutoML Dashboard")
    st.markdown("Interactive ML experiment tracking and model comparison")
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Overview", "Experiments", "Model Comparison", "Training"],
    )
    
    dashboard = AutoMLDashboard()
    
    if page == "Overview":
        show_overview(st, dashboard)
    elif page == "Experiments":
        show_experiments(st, dashboard, px, go)
    elif page == "Model Comparison":
        show_comparison(st, dashboard, px)
    elif page == "Training":
        show_training(st, dashboard)


def show_overview(st: Any, dashboard: AutoMLDashboard) -> None:
    """Show overview page."""
    st.header("Overview")
    
    experiments = dashboard.load_experiments()
    
    # Stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Experiments", len(experiments))
    
    with col2:
        st.metric("Active Models", "N/A")  # Would query from registry
    
    with col3:
        st.metric("Today's Runs", "N/A")  # Would filter by date
    
    with col4:
        st.metric("Success Rate", "N/A")  # Would calculate from logs
    
    # Recent experiments
    st.subheader("Recent Experiments")
    
    if experiments:
        df = pd.DataFrame(experiments)
        st.dataframe(df[["path", "status"]].head(10), use_container_width=True)
    else:
        st.info("No experiments found. Start training to see results!")


def show_experiments(st: Any, dashboard: AutoMLDashboard, px: Any, go: Any) -> None:
    """Show experiments page."""
    st.header("Experiments")
    
    experiments = dashboard.load_experiments()
    
    if not experiments:
        st.warning("No experiments found.")
        return
    
    # Select experiment
    exp_ids = [str(exp["path"].name) for exp in experiments]
    selected_exp = st.selectbox("Select Experiment", exp_ids)
    
    if selected_exp:
        metrics_df = dashboard.get_experiment_metrics(selected_exp)
        
        if not metrics_df.empty:
            # Metrics over time
            st.subheader("Training Metrics")
            
            if "iteration" in metrics_df.columns:
                fig = px.line(
                    metrics_df,
                    x="iteration",
                    y=[col for col in metrics_df.columns if col != "iteration"],
                    title="Metrics over Iterations",
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Best scores
            st.subheader("Best Scores")
            
            numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
            best_scores = {col: metrics_df[col].max() for col in numeric_cols}
            
            col1, col2, col3 = st.columns(3)
            
            for idx, (metric, score) in enumerate(best_scores.items()):
                with [col1, col2, col3][idx % 3]:
                    st.metric(metric.upper(), f"{score:.4f}")


def show_comparison(st: Any, dashboard: AutoMLDashboard, px: Any) -> None:
    """Show model comparison page."""
    st.header("Model Comparison")
    
    experiments = dashboard.load_experiments()
    
    if not experiments:
        st.warning("No experiments to compare.")
        return
    
    # Select experiments to compare
    exp_ids = [str(exp["path"].name) for exp in experiments]
    selected_exps = st.multiselect(
        "Select Experiments to Compare",
        exp_ids,
        default=exp_ids[:min(3, len(exp_ids))],
    )
    
    if selected_exps:
        # Select metric
        metric = st.selectbox(
            "Metric",
            ["accuracy", "f1", "precision", "recall", "auc"],
        )
        
        # Compare
        comparison = dashboard.compare_experiments(selected_exps, metric)
        
        if not comparison.empty:
            # Bar chart
            fig = px.bar(
                comparison,
                x="experiment_id",
                y="best_score",
                title=f"{metric.upper()} Comparison",
                error_y="std_score",
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Table
            st.dataframe(comparison, use_container_width=True)


def show_training(st: Any, dashboard: AutoMLDashboard) -> None:
    """Show training page."""
    st.header("Train New Model")
    
    st.markdown("Configure and launch a new AutoML experiment")
    
    # Dataset selection
    st.subheader("1. Dataset")
    
    dataset_option = st.radio(
        "Dataset Source",
        ["Upload CSV", "Use Built-in Dataset", "Connect to Database"],
    )
    
    if dataset_option == "Upload CSV":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write("Preview:", df.head())
            
            target_col = st.selectbox("Target Column", df.columns)
    
    elif dataset_option == "Use Built-in Dataset":
        dataset_name = st.selectbox(
            "Dataset",
            ["iris", "wine", "breast_cancer", "diabetes"],
        )
    
    # Task type
    st.subheader("2. Task Configuration")
    
    task_type = st.selectbox(
        "Task Type",
        ["classification", "regression"],
    )
    
    # Model selection
    st.subheader("3. Model Selection")
    
    model_types = st.multiselect(
        "Models to Try",
        ["random_forest", "xgboost", "lightgbm", "catboost", "neural_net"],
        default=["random_forest", "xgboost"],
    )
    
    # Optimization
    st.subheader("4. Optimization Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_trials = st.number_input("Number of Trials", min_value=10, max_value=1000, value=100)
        timeout = st.number_input("Timeout (seconds)", min_value=60, value=3600)
    
    with col2:
        optimizer = st.selectbox("Optimizer", ["optuna", "ray_tune", "random_search"])
        metric = st.selectbox("Optimization Metric", ["accuracy", "f1", "auc", "rmse"])
    
    # Launch button
    if st.button("🚀 Launch Training", type="primary"):
        with st.spinner("Training in progress..."):
            st.success("Training started! Check the Experiments page for results.")
            
            # In real implementation, would launch training job
            st.info(
                "In production, this would launch a distributed training job "
                "with the specified configuration."
            )


def run_dashboard(host: str = "0.0.0.0", port: int = 8501) -> None:
    """Run the dashboard server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
    """
    import subprocess
    import sys
    
    # Write app to temp file
    import tempfile
    
    app_code = """
import sys
sys.path.insert(0, '.')

from automl.dashboard.app import create_streamlit_app

if __name__ == '__main__':
    create_streamlit_app()
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(app_code)
        temp_file = f.name
    
    # Run streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            temp_file,
            "--server.address", host,
            "--server.port", str(port),
        ])
    finally:
        Path(temp_file).unlink()


if __name__ == "__main__":
    create_streamlit_app()
