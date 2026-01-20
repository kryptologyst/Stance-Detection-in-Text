# Stance Detection in Text

A comprehensive stance detection system that determines whether text expresses a **favor**, **against**, or **neutral** stance toward a given target topic. This project uses state-of-the-art transformer models and provides multiple interfaces for easy usage.

## Features

- **Modern Architecture**: Built with Hugging Face Transformers and PyTorch
- **Multiple Interfaces**: CLI, Web UI (Streamlit), and Python API
- **Pre-trained Models**: Uses specialized stance detection models
- **Fine-tuning Support**: Train custom models on your data
- **Synthetic Data Generation**: Create datasets for testing and training
- **Comprehensive Evaluation**: Detailed metrics and visualizations
- **Type Hints & Documentation**: Fully documented with type annotations
- **Configuration Management**: JSON-based configuration system

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- See `requirements.txt` for complete dependencies

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kryptologyst/Stance-Detection-in-Text.git
   cd Stance-Detection-in-Text
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Web Interface (Recommended for beginners)

Launch the Streamlit web interface:

```bash
streamlit run web_app/app.py
```

Then open your browser to `http://localhost:8501` and explore the interactive interface.

### Command Line Interface

**Single prediction**:
```bash
python src/cli.py predict --target "climate change" --text "I strongly support climate action"
```

**Batch prediction**:
```bash
python src/cli.py batch-predict --input data.json --output results.json
```

**Model evaluation**:
```bash
python src/cli.py evaluate --dataset-size 500
```

### Python API

```python
from src.stance_detector import StanceDetector

# Initialize detector
detector = StanceDetector()
detector.load_model()

# Single prediction
result = detector.predict_stance(
    text="I believe climate change is real and urgent",
    target="climate change"
)
print(f"Stance: {result['label']} (confidence: {result['confidence']:.3f})")

# Batch prediction
texts = ["Climate action is needed", "Climate change is a hoax"]
targets = ["climate change", "climate change"]
results = detector.batch_predict(texts, targets)
```

## Usage Examples

### 1. Single Stance Detection

```python
from src.stance_detector import StanceDetector

detector = StanceDetector()
detector.load_model()

# Example predictions
examples = [
    ("climate change", "I strongly believe we need immediate action on climate change."),
    ("artificial intelligence", "AI poses serious risks to human employment."),
    ("vaccination", "Vaccines have saved millions of lives.")
]

for target, text in examples:
    result = detector.predict_stance(text, target)
    print(f"Target: {target}")
    print(f"Text: {text}")
    print(f"Stance: {result['label']} (confidence: {result['confidence']:.3f})")
    print("-" * 50)
```

### 2. Model Evaluation

```python
# Generate synthetic dataset and evaluate
dataset = detector.create_synthetic_dataset(size=1000)

# Split dataset
train_size = int(0.7 * len(dataset))
eval_size = int(0.15 * len(dataset))

train_dataset = dataset.select(range(train_size))
eval_dataset = dataset.select(range(train_size, train_size + eval_size))
test_dataset = dataset.select(range(train_size + eval_size, len(dataset)))

# Evaluate model
results = detector.evaluate_model(test_dataset)
print(f"Accuracy: {results['accuracy']:.3f}")
print(f"F1 Score: {results['f1_weighted']:.3f}")
```

### 3. Model Fine-tuning

```python
# Create training data
dataset = detector.create_synthetic_dataset(size=1000)

# Fine-tune model
detector.fine_tune_model(
    train_dataset=dataset,
    output_dir="./models/fine_tuned_stance",
    num_epochs=3
)
```

## Project Structure

```
stance-detection-text/
├── src/                    # Source code
│   ├── stance_detector.py  # Main stance detection class
│   └── cli.py             # Command line interface
├── web_app/               # Web interface
│   └── app.py            # Streamlit application
├── config/               # Configuration files
│   └── config.json       # Main configuration
├── data/                 # Data directory (created at runtime)
├── models/               # Model directory (created at runtime)
├── results/              # Results directory (created at runtime)
├── tests/                # Test files
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore file
└── README.md            # This file
```

## Configuration

The system uses `config/config.json` for configuration:

```json
{
  "model": {
    "default_model": "cardiffnlp/twitter-roberta-base-stance-climate",
    "max_length": 512,
    "batch_size": 16
  },
  "training": {
    "num_epochs": 3,
    "learning_rate": 2e-5,
    "warmup_steps": 500
  },
  "data": {
    "synthetic_dataset_size": 1000,
    "train_split": 0.7,
    "eval_split": 0.15,
    "test_split": 0.15
  }
}
```

## Available Models

The system supports several pre-trained stance detection models:

- `cardiffnlp/twitter-roberta-base-stance-climate` (default)
- `cardiffnlp/twitter-roberta-base-stance-vaccine`
- `cardiffnlp/twitter-roberta-base-stance-brexit`
- `bert-base-uncased` (general purpose)

## Performance Metrics

The system provides comprehensive evaluation metrics:

- **Overall**: Accuracy, Precision, Recall, F1-Score
- **Per-class**: Individual metrics for FAVOR, AGAINST, NONE
- **Visualizations**: Confusion matrix, performance charts
- **Confidence Scores**: Prediction confidence for each result

## 🔧 Advanced Usage

### Custom Dataset Format

For batch processing, use JSON format:

```json
[
  {
    "text": "I support climate action",
    "target": "climate change"
  },
  {
    "text": "AI is dangerous",
    "target": "artificial intelligence"
  }
]
```

### Training Custom Models

```python
# Load your custom dataset
from datasets import Dataset

custom_data = [
    {"text": "Your text", "target": "Your target", "label": "FAVOR"},
    # ... more examples
]

dataset = Dataset.from_list(custom_data)

# Fine-tune
detector.fine_tune_model(
    train_dataset=dataset,
    output_dir="./models/custom_stance",
    num_epochs=5,
    learning_rate=1e-5
)
```

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

## Logging

The system includes comprehensive logging:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

Logs are saved to the `logs/` directory and include:
- Model loading progress
- Training metrics
- Evaluation results
- Error messages

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face for the Transformers library
- Cardiff NLP for pre-trained stance detection models
- The open-source ML community

## Support

For questions, issues, or contributions, please:
1. Check the existing issues
2. Create a new issue with detailed description
3. Contact the maintainers
# Stance-Detection-in-Text
