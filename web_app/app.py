"""
Streamlit Web Interface for Stance Detection

This module provides a user-friendly web interface for the stance detection system.
Users can input text and targets to get stance predictions, view model performance,
and explore the synthetic dataset.

Author: AI Assistant
Date: 2024
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from stance_detector import StanceDetector


def load_config():
    """Load configuration from JSON file."""
    config_path = Path(__file__).parent.parent / "config" / "config.json"
    with open(config_path, "r") as f:
        return json.load(f)


def initialize_detector():
    """Initialize the stance detector."""
    if "detector" not in st.session_state:
        config = load_config()
        st.session_state.detector = StanceDetector(
            model_name=config["model"]["default_model"]
        )
        st.session_state.detector.load_model()


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Stance Detection System",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 Stance Detection System")
    st.markdown("Detect the stance (Favor, Against, or Neutral) of text toward a given target.")
    
    # Initialize detector
    with st.spinner("Loading stance detection model..."):
        initialize_detector()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Single Prediction", "Batch Prediction", "Model Evaluation", "Dataset Explorer"]
    )
    
    if page == "Single Prediction":
        single_prediction_page()
    elif page == "Batch Prediction":
        batch_prediction_page()
    elif page == "Model Evaluation":
        model_evaluation_page()
    elif page == "Dataset Explorer":
        dataset_explorer_page()


def single_prediction_page():
    """Single prediction interface."""
    st.header("Single Stance Prediction")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input")
        target = st.text_input(
            "Target Topic",
            value="climate change",
            help="The topic or subject that the text expresses a stance about"
        )
        
        text = st.text_area(
            "Text to Analyze",
            value="I believe that the evidence strongly supports the need for immediate action on climate change.",
            height=100,
            help="The text whose stance you want to detect"
        )
        
        if st.button("Detect Stance", type="primary"):
            if target and text:
                with st.spinner("Analyzing stance..."):
                    result = st.session_state.detector.predict_stance(text, target)
                
                with col2:
                    st.subheader("Results")
                    
                    # Display result
                    stance = result["label"]
                    confidence = result["confidence"]
                    
                    # Color coding for stance
                    if stance == "FAVOR":
                        color = "🟢"
                        bg_color = "#d4edda"
                    elif stance == "AGAINST":
                        color = "🔴"
                        bg_color = "#f8d7da"
                    else:
                        color = "🟡"
                        bg_color = "#fff3cd"
                    
                    st.markdown(
                        f"""
                        <div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; text-align: center;">
                            <h2>{color} {stance}</h2>
                            <h3>Confidence: {confidence:.1%}</h3>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Detailed results
                    st.subheader("Detailed Results")
                    result_df = pd.DataFrame([{
                        "Target": result["target"],
                        "Text": result["text"][:100] + "..." if len(result["text"]) > 100 else result["text"],
                        "Predicted Stance": result["label"],
                        "Confidence": f"{result['confidence']:.3f}"
                    }])
                    st.dataframe(result_df, use_container_width=True)
            else:
                st.error("Please provide both target and text.")


def batch_prediction_page():
    """Batch prediction interface."""
    st.header("Batch Stance Prediction")
    
    st.subheader("Upload Data")
    st.markdown("Upload a CSV file with columns 'target' and 'text', or use the sample data below.")
    
    # Sample data
    sample_data = {
        "target": [
            "climate change",
            "artificial intelligence", 
            "vaccination",
            "climate change",
            "artificial intelligence"
        ],
        "text": [
            "I strongly believe we need immediate action on climate change.",
            "AI poses serious risks to human employment.",
            "Vaccines have saved millions of lives.",
            "Climate change is a hoax created by scientists.",
            "AI will revolutionize healthcare and improve lives."
        ]
    }
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="CSV file should have 'target' and 'text' columns"
    )
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.DataFrame(sample_data)
        st.info("Using sample data. Upload a CSV file to use your own data.")
    
    st.subheader("Data Preview")
    st.dataframe(df.head(), use_container_width=True)
    
    if st.button("Run Batch Prediction", type="primary"):
        if len(df) > 0:
            with st.spinner(f"Processing {len(df)} predictions..."):
                results = st.session_state.detector.batch_predict(
                    df["text"].tolist(),
                    df["target"].tolist()
                )
            
            # Create results dataframe
            results_df = pd.DataFrame(results)
            results_df["confidence"] = results_df["confidence"].round(3)
            
            st.subheader("Prediction Results")
            st.dataframe(results_df, use_container_width=True)
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="stance_predictions.csv",
                mime="text/csv"
            )
            
            # Visualization
            st.subheader("Results Visualization")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Stance distribution
                stance_counts = results_df["label"].value_counts()
                fig_pie = px.pie(
                    values=stance_counts.values,
                    names=stance_counts.index,
                    title="Stance Distribution"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Confidence distribution
                fig_hist = px.histogram(
                    results_df,
                    x="confidence",
                    title="Confidence Score Distribution",
                    nbins=20
                )
                st.plotly_chart(fig_hist, use_container_width=True)


def model_evaluation_page():
    """Model evaluation interface."""
    st.header("Model Evaluation")
    
    st.subheader("Generate Synthetic Dataset")
    dataset_size = st.slider(
        "Dataset Size",
        min_value=100,
        max_value=2000,
        value=500,
        step=100
    )
    
    if st.button("Generate Dataset and Evaluate", type="primary"):
        with st.spinner("Generating synthetic dataset and evaluating model..."):
            # Generate dataset
            dataset = st.session_state.detector.create_synthetic_dataset(size=dataset_size)
            
            # Split dataset
            config = load_config()
            train_split = config["data"]["train_split"]
            eval_split = config["data"]["eval_split"]
            
            train_size = int(train_split * len(dataset))
            eval_size = int(eval_split * len(dataset))
            
            train_dataset = dataset.select(range(train_size))
            eval_dataset = dataset.select(range(train_size, train_size + eval_size))
            test_dataset = dataset.select(range(train_size + eval_size, len(dataset)))
            
            # Evaluate model
            results = st.session_state.detector.evaluate_model(test_dataset)
            
            # Generate confusion matrix
            st.session_state.detector.plot_confusion_matrix(test_dataset)
        
        # Display results
        st.subheader("Evaluation Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{results['accuracy']:.3f}")
        with col2:
            st.metric("Precision (Weighted)", f"{results['precision_weighted']:.3f}")
        with col3:
            st.metric("Recall (Weighted)", f"{results['recall_weighted']:.3f}")
        with col4:
            st.metric("F1 Score (Weighted)", f"{results['f1_weighted']:.3f}")
        
        # Per-class metrics
        st.subheader("Per-Class Metrics")
        
        metrics_df = pd.DataFrame({
            "Class": ["FAVOR", "AGAINST", "NONE"],
            "Precision": [
                results["precision_per_class"]["FAVOR"],
                results["precision_per_class"]["AGAINST"],
                results["precision_per_class"]["NONE"]
            ],
            "Recall": [
                results["recall_per_class"]["FAVOR"],
                results["recall_per_class"]["AGAINST"],
                results["recall_per_class"]["NONE"]
            ],
            "F1 Score": [
                results["f1_per_class"]["FAVOR"],
                results["f1_per_class"]["AGAINST"],
                results["f1_per_class"]["NONE"]
            ]
        })
        
        st.dataframe(metrics_df, use_container_width=True)
        
        # Visualization
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=("Precision", "Recall", "F1 Score"),
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
        )
        
        classes = ["FAVOR", "AGAINST", "NONE"]
        
        fig.add_trace(
            go.Bar(x=classes, y=metrics_df["Precision"], name="Precision"),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=classes, y=metrics_df["Recall"], name="Recall"),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=classes, y=metrics_df["F1 Score"], name="F1 Score"),
            row=1, col=3
        )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Confusion matrix
        st.subheader("Confusion Matrix")
        try:
            confusion_matrix_path = "./results/confusion_matrix.png"
            if Path(confusion_matrix_path).exists():
                st.image(confusion_matrix_path, caption="Confusion Matrix")
            else:
                st.warning("Confusion matrix not found. Please run evaluation first.")
        except Exception as e:
            st.error(f"Error loading confusion matrix: {e}")


def dataset_explorer_page():
    """Dataset exploration interface."""
    st.header("Dataset Explorer")
    
    st.subheader("Synthetic Dataset Statistics")
    
    dataset_size = st.slider(
        "Dataset Size",
        min_value=100,
        max_value=1000,
        value=300,
        step=50
    )
    
    if st.button("Generate and Explore Dataset", type="primary"):
        with st.spinner("Generating synthetic dataset..."):
            dataset = st.session_state.detector.create_synthetic_dataset(size=dataset_size)
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(dataset)
        
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Unique Targets", df["target"].nunique())
        with col3:
            st.metric("Unique Stances", df["label"].nunique())
        
        # Dataset distribution
        st.subheader("Dataset Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Target distribution
            target_counts = df["target"].value_counts()
            fig_targets = px.bar(
                x=target_counts.index,
                y=target_counts.values,
                title="Target Distribution"
            )
            st.plotly_chart(fig_targets, use_container_width=True)
        
        with col2:
            # Stance distribution
            stance_counts = df["label"].value_counts()
            fig_stances = px.pie(
                values=stance_counts.values,
                names=stance_counts.index,
                title="Stance Distribution"
            )
            st.plotly_chart(fig_stances, use_container_width=True)
        
        # Cross-tabulation
        st.subheader("Target vs Stance Cross-tabulation")
        crosstab = pd.crosstab(df["target"], df["label"], margins=True)
        st.dataframe(crosstab, use_container_width=True)
        
        # Sample data
        st.subheader("Sample Data")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Download dataset
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Dataset as CSV",
            data=csv,
            file_name="synthetic_stance_dataset.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
