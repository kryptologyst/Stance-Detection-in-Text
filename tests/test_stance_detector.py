"""
Test suite for the stance detection system.

This module contains unit tests for the StanceDetector class and related functionality.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from stance_detector import StanceDetector


class TestStanceDetector:
    """Test cases for the StanceDetector class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = StanceDetector()
    
    def test_init(self):
        """Test StanceDetector initialization."""
        assert self.detector.model_name == "cardiffnlp/twitter-roberta-base-stance-climate"
        assert self.detector.device in ["cpu", "cuda"]
        assert self.detector.tokenizer is None
        assert self.detector.model is None
        assert self.detector.pipeline is None
    
    def test_init_custom_model(self):
        """Test initialization with custom model."""
        detector = StanceDetector(model_name="bert-base-uncased")
        assert detector.model_name == "bert-base-uncased"
    
    @patch('stance_detector.AutoTokenizer')
    @patch('stance_detector.AutoModelForSequenceClassification')
    @patch('stance_detector.pipeline')
    def test_load_model(self, mock_pipeline, mock_model_class, mock_tokenizer_class):
        """Test model loading."""
        # Mock the transformers components
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_pipeline_instance = Mock()
        
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model
        mock_pipeline.return_value = mock_pipeline_instance
        
        # Test loading
        self.detector.load_model()
        
        # Verify calls
        mock_tokenizer_class.from_pretrained.assert_called_once_with(self.detector.model_name)
        mock_model_class.from_pretrained.assert_called_once_with(
            self.detector.model_name,
            num_labels=3
        )
        mock_pipeline.assert_called_once()
        
        # Verify attributes
        assert self.detector.tokenizer == mock_tokenizer
        assert self.detector.model == mock_model
        assert self.detector.pipeline == mock_pipeline_instance
    
    def test_create_synthetic_dataset(self):
        """Test synthetic dataset creation."""
        dataset = self.detector.create_synthetic_dataset(size=100)
        
        assert len(dataset) == 100
        assert "text" in dataset.column_names
        assert "target" in dataset.column_names
        assert "label" in dataset.column_names
        assert "id" in dataset.column_names
        
        # Check that all labels are valid
        valid_labels = {"FAVOR", "AGAINST", "NONE"}
        labels = set(dataset["label"])
        assert labels.issubset(valid_labels)
        
        # Check that all targets are valid
        valid_targets = {"climate change", "artificial intelligence", "vaccination"}
        targets = set(dataset["target"])
        assert targets.issubset(valid_targets)
    
    def test_create_synthetic_dataset_size(self):
        """Test synthetic dataset with different sizes."""
        for size in [10, 50, 200]:
            dataset = self.detector.create_synthetic_dataset(size=size)
            assert len(dataset) == size
    
    @patch.object(StanceDetector, 'load_model')
    def test_predict_stance_without_model(self, mock_load_model):
        """Test prediction when model is not loaded."""
        mock_pipeline = Mock()
        mock_pipeline.return_value = [{"label": "FAVOR", "score": 0.9}]
        self.detector.pipeline = mock_pipeline
        
        result = self.detector.predict_stance("test text", "test target")
        
        mock_load_model.assert_called_once()
        assert result["label"] == "FAVOR"
        assert result["confidence"] == 0.9
    
    def test_predict_stance_with_model(self):
        """Test prediction when model is already loaded."""
        mock_pipeline = Mock()
        mock_pipeline.return_value = [{"label": "AGAINST", "score": 0.8}]
        self.detector.pipeline = mock_pipeline
        
        result = self.detector.predict_stance("test text", "test target")
        
        assert result["label"] == "AGAINST"
        assert result["confidence"] == 0.8
        assert result["text"] == "test text"
        assert result["target"] == "test target"
    
    def test_predict_stance_return_label_only(self):
        """Test prediction returning only label."""
        mock_pipeline = Mock()
        mock_pipeline.return_value = [{"label": "NONE", "score": 0.7}]
        self.detector.pipeline = mock_pipeline
        
        result = self.detector.predict_stance("test text", "test target", return_confidence=False)
        
        assert result == "NONE"
    
    def test_batch_predict(self):
        """Test batch prediction."""
        mock_pipeline = Mock()
        mock_pipeline.side_effect = [
            [{"label": "FAVOR", "score": 0.9}],
            [{"label": "AGAINST", "score": 0.8}]
        ]
        self.detector.pipeline = mock_pipeline
        
        texts = ["text1", "text2"]
        targets = ["target1", "target2"]
        
        results = self.detector.batch_predict(texts, targets)
        
        assert len(results) == 2
        assert results[0]["label"] == "FAVOR"
        assert results[1]["label"] == "AGAINST"
    
    def test_batch_predict_mismatched_lengths(self):
        """Test batch prediction with mismatched input lengths."""
        texts = ["text1", "text2"]
        targets = ["target1"]  # Mismatched length
        
        with pytest.raises(ValueError, match="Number of texts must match number of targets"):
            self.detector.batch_predict(texts, targets)
    
    def test_evaluate_model(self):
        """Test model evaluation."""
        # Create a small test dataset
        test_data = [
            {"text": "I support climate action", "target": "climate change", "label": "FAVOR"},
            {"text": "Climate change is fake", "target": "climate change", "label": "AGAINST"},
            {"text": "I'm not sure about climate", "target": "climate change", "label": "NONE"}
        ]
        
        from datasets import Dataset
        test_dataset = Dataset.from_list(test_data)
        
        # Mock the prediction pipeline
        mock_pipeline = Mock()
        mock_pipeline.side_effect = [
            [{"label": "FAVOR", "score": 0.9}],
            [{"label": "AGAINST", "score": 0.8}],
            [{"label": "NONE", "score": 0.7}]
        ]
        self.detector.pipeline = mock_pipeline
        
        # Test evaluation
        with tempfile.TemporaryDirectory() as temp_dir:
            results = self.detector.evaluate_model(
                test_dataset, 
                save_results=True,
                output_path=f"{temp_dir}/results.json"
            )
            
            # Check results structure
            assert "accuracy" in results
            assert "precision_weighted" in results
            assert "recall_weighted" in results
            assert "f1_weighted" in results
            assert "precision_per_class" in results
            assert "recall_per_class" in results
            assert "f1_per_class" in results
            
            # Check that results file was created
            assert Path(f"{temp_dir}/results.json").exists()
    
    def test_plot_confusion_matrix(self):
        """Test confusion matrix plotting."""
        # Create a small test dataset
        test_data = [
            {"text": "I support climate action", "target": "climate change", "label": "FAVOR"},
            {"text": "Climate change is fake", "target": "climate change", "label": "AGAINST"}
        ]
        
        from datasets import Dataset
        test_dataset = Dataset.from_list(test_data)
        
        # Mock the prediction pipeline
        mock_pipeline = Mock()
        mock_pipeline.side_effect = [
            [{"label": "FAVOR", "score": 0.9}],
            [{"label": "AGAINST", "score": 0.8}]
        ]
        self.detector.pipeline = mock_pipeline
        
        # Test plotting
        with tempfile.TemporaryDirectory() as temp_dir:
            self.detector.plot_confusion_matrix(
                test_dataset,
                save_path=f"{temp_dir}/confusion_matrix.png"
            )
            
            # Check that plot was created
            assert Path(f"{temp_dir}/confusion_matrix.png").exists()


class TestConfiguration:
    """Test configuration loading and validation."""
    
    def test_config_loading(self):
        """Test loading configuration from JSON file."""
        config_path = Path(__file__).parent.parent / "config" / "config.json"
        
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
            
            # Check required sections
            assert "model" in config
            assert "training" in config
            assert "data" in config
            assert "paths" in config
            
            # Check model configuration
            assert "default_model" in config["model"]
            assert "max_length" in config["model"]
            assert "batch_size" in config["model"]


if __name__ == "__main__":
    pytest.main([__file__])
