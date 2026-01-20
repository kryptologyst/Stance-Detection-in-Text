"""
Stance Detection Module

This module provides a comprehensive stance detection system using state-of-the-art
transformer models. It supports both pre-trained models and fine-tuning capabilities
for custom stance detection tasks.

Author: AI Assistant
Date: 2024
"""

import logging
import json
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StanceDetector:
    """
    A comprehensive stance detection system using transformer models.
    
    This class provides functionality for:
    - Loading pre-trained stance detection models
    - Fine-tuning models on custom datasets
    - Predicting stances for new text
    - Evaluating model performance
    """
    
    def __init__(
        self, 
        model_name: str = "cardiffnlp/twitter-roberta-base-stance-climate",
        device: Optional[str] = None
    ):
        """
        Initialize the stance detector.
        
        Args:
            model_name: Name of the pre-trained model to use
            device: Device to run the model on ('cpu', 'cuda', or None for auto)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
        logger.info(f"Initializing StanceDetector with model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
    def load_model(self) -> None:
        """Load the pre-trained model and tokenizer."""
        try:
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            logger.info("Loading model...")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=3  # FAVOR, AGAINST, NONE
            )
            
            logger.info("Creating pipeline...")
            self.pipeline = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict_stance(
        self, 
        text: str, 
        target: str, 
        return_confidence: bool = True
    ) -> Union[Dict[str, Union[str, float]], str]:
        """
        Predict the stance of text toward a target.
        
        Args:
            text: The text to analyze
            target: The target topic
            return_confidence: Whether to return confidence scores
            
        Returns:
            Dictionary with label and score, or just the label string
        """
        if not self.pipeline:
            self.load_model()
        
        # Format the input for stance detection
        formatted_text = f"Target: {target}. Text: {text}"
        
        try:
            result = self.pipeline(formatted_text)
            
            if return_confidence:
                return {
                    "label": result[0]["label"],
                    "confidence": result[0]["score"],
                    "text": text,
                    "target": target
                }
            else:
                return result[0]["label"]
                
        except Exception as e:
            logger.error(f"Error predicting stance: {e}")
            raise
    
    def batch_predict(
        self, 
        texts: List[str], 
        targets: List[str]
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Predict stances for multiple text-target pairs.
        
        Args:
            texts: List of texts to analyze
            targets: List of corresponding targets
            
        Returns:
            List of prediction dictionaries
        """
        if len(texts) != len(targets):
            raise ValueError("Number of texts must match number of targets")
        
        results = []
        for text, target in zip(texts, targets):
            result = self.predict_stance(text, target)
            results.append(result)
        
        return results
    
    def create_synthetic_dataset(self, size: int = 1000) -> Dataset:
        """
        Create a synthetic dataset for stance detection.
        
        Args:
            size: Number of samples to generate
            
        Returns:
            Hugging Face Dataset object
        """
        logger.info(f"Creating synthetic dataset with {size} samples...")
        
        # Define targets and corresponding stance examples
        targets_and_examples = {
            "climate change": {
                "FAVOR": [
                    "I strongly believe we need immediate action on climate change.",
                    "The scientific evidence clearly shows climate change is real and urgent.",
                    "We must reduce carbon emissions to save our planet.",
                    "Climate change is the biggest threat to humanity."
                ],
                "AGAINST": [
                    "Climate change is a hoax created by scientists for funding.",
                    "There's no real evidence that humans cause climate change.",
                    "Climate change policies will destroy our economy.",
                    "The climate has always changed naturally."
                ],
                "NONE": [
                    "The weather seems different lately.",
                    "I heard something about climate on the news.",
                    "Climate is a complex topic with many factors.",
                    "I'm not sure what to think about climate change."
                ]
            },
            "artificial intelligence": {
                "FAVOR": [
                    "AI will revolutionize healthcare and improve lives.",
                    "Artificial intelligence can solve many world problems.",
                    "AI technology is advancing rapidly and beneficially.",
                    "I'm excited about the potential of AI."
                ],
                "AGAINST": [
                    "AI poses serious risks to human employment.",
                    "Artificial intelligence could become dangerous.",
                    "AI development should be slowed down.",
                    "Machines will never replace human intelligence."
                ],
                "NONE": [
                    "AI is becoming more common in daily life.",
                    "I use AI tools sometimes for work.",
                    "Artificial intelligence is a growing field.",
                    "AI seems to be everywhere these days."
                ]
            },
            "vaccination": {
                "FAVOR": [
                    "Vaccines have saved millions of lives throughout history.",
                    "I trust the scientific process behind vaccine development.",
                    "Vaccination is crucial for public health.",
                    "Vaccines are safe and effective."
                ],
                "AGAINST": [
                    "Vaccines contain harmful chemicals and toxins.",
                    "I don't trust pharmaceutical companies.",
                    "Natural immunity is better than vaccination.",
                    "Vaccines cause autism and other health problems."
                ],
                "NONE": [
                    "I'm not sure about vaccine safety.",
                    "Vaccines seem to work for some people.",
                    "There are different opinions about vaccination.",
                    "I need to learn more about vaccines."
                ]
            }
        }
        
        # Generate synthetic data
        data = []
        targets_list = list(targets_and_examples.keys())
        
        for i in range(size):
            target = np.random.choice(targets_list)
            stance = np.random.choice(["FAVOR", "AGAINST", "NONE"])
            examples = targets_and_examples[target][stance]
            text = np.random.choice(examples)
            
            data.append({
                "text": text,
                "target": target,
                "label": stance,
                "id": i
            })
        
        dataset = Dataset.from_list(data)
        logger.info(f"Created synthetic dataset with {len(dataset)} samples")
        
        return dataset
    
    def fine_tune_model(
        self, 
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        output_dir: str = "./models/fine_tuned_stance",
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5
    ) -> None:
        """
        Fine-tune the model on a custom dataset.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            output_dir: Directory to save the fine-tuned model
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for training
        """
        if not self.tokenizer or not self.model:
            self.load_model()
        
        logger.info("Starting fine-tuning...")
        
        # Tokenize the datasets
        def tokenize_function(examples):
            texts = [f"Target: {target}. Text: {text}" 
                    for text, target in zip(examples["text"], examples["target"])]
            return self.tokenizer(texts, truncation=True, padding=True)
        
        train_tokenized = train_dataset.map(tokenize_function, batched=True)
        if eval_dataset:
            eval_tokenized = eval_dataset.map(tokenize_function, batched=True)
        
        # Create label mapping
        label2id = {"FAVOR": 0, "AGAINST": 1, "NONE": 2}
        id2label = {0: "FAVOR", 1: "AGAINST", 2: "NONE"}
        
        # Update model config
        self.model.config.label2id = label2id
        self.model.config.id2label = id2label
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset else False,
            learning_rate=learning_rate,
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=eval_tokenized if eval_dataset else None,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Train the model
        trainer.train()
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Fine-tuning completed! Model saved to {output_dir}")
    
    def evaluate_model(
        self, 
        test_dataset: Dataset,
        save_results: bool = True,
        output_path: str = "./results/evaluation_results.json"
    ) -> Dict[str, float]:
        """
        Evaluate the model on a test dataset.
        
        Args:
            test_dataset: Test dataset for evaluation
            save_results: Whether to save results to file
            output_path: Path to save evaluation results
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.pipeline:
            self.load_model()
        
        logger.info("Evaluating model...")
        
        # Get predictions
        predictions = []
        true_labels = []
        
        for example in test_dataset:
            pred = self.predict_stance(example["text"], example["target"])
            predictions.append(pred["label"])
            true_labels.append(example["label"])
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average="weighted"
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            true_labels, predictions, average=None, labels=["FAVOR", "AGAINST", "NONE"]
        )
        
        results = {
            "accuracy": accuracy,
            "precision_weighted": precision,
            "recall_weighted": recall,
            "f1_weighted": f1,
            "precision_per_class": {
                "FAVOR": precision_per_class[0],
                "AGAINST": precision_per_class[1],
                "NONE": precision_per_class[2]
            },
            "recall_per_class": {
                "FAVOR": recall_per_class[0],
                "AGAINST": recall_per_class[1],
                "NONE": recall_per_class[2]
            },
            "f1_per_class": {
                "FAVOR": f1_per_class[0],
                "AGAINST": f1_per_class[1],
                "NONE": f1_per_class[2]
            }
        }
        
        if save_results:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Evaluation results saved to {output_path}")
        
        logger.info(f"Evaluation completed. Accuracy: {accuracy:.3f}")
        return results
    
    def plot_confusion_matrix(
        self, 
        test_dataset: Dataset,
        save_path: str = "./results/confusion_matrix.png"
    ) -> None:
        """
        Plot and save confusion matrix for the test dataset.
        
        Args:
            test_dataset: Test dataset
            save_path: Path to save the confusion matrix plot
        """
        if not self.pipeline:
            self.load_model()
        
        # Get predictions
        predictions = []
        true_labels = []
        
        for example in test_dataset:
            pred = self.predict_stance(example["text"], example["target"])
            predictions.append(pred["label"])
            true_labels.append(example["label"])
        
        # Create confusion matrix
        cm = confusion_matrix(true_labels, predictions, labels=["FAVOR", "AGAINST", "NONE"])
        
        # Plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                   xticklabels=["FAVOR", "AGAINST", "NONE"],
                   yticklabels=["FAVOR", "AGAINST", "NONE"])
        plt.title("Confusion Matrix - Stance Detection")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        
        # Save plot
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Confusion matrix saved to {save_path}")


def main():
    """Main function to demonstrate the stance detector."""
    # Initialize detector
    detector = StanceDetector()
    
    # Load model
    detector.load_model()
    
    # Example predictions
    examples = [
        ("climate change", "I believe that the evidence strongly supports the need for immediate action on climate change."),
        ("artificial intelligence", "AI poses serious risks to human employment and should be regulated."),
        ("vaccination", "Vaccines have saved millions of lives and are crucial for public health.")
    ]
    
    print("Stance Detection Results:")
    print("=" * 50)
    
    for target, text in examples:
        result = detector.predict_stance(text, target)
        print(f"Target: {target}")
        print(f"Text: {text}")
        print(f"Stance: {result['label']} (confidence: {result['confidence']:.3f})")
        print("-" * 50)
    
    # Create and evaluate on synthetic data
    print("\nCreating synthetic dataset...")
    dataset = detector.create_synthetic_dataset(size=100)
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    eval_size = int(0.15 * len(dataset))
    
    train_dataset = dataset.select(range(train_size))
    eval_dataset = dataset.select(range(train_size, train_size + eval_size))
    test_dataset = dataset.select(range(train_size + eval_size, len(dataset)))
    
    print(f"Dataset split - Train: {len(train_dataset)}, Eval: {len(eval_dataset)}, Test: {len(test_dataset)}")
    
    # Evaluate on test set
    print("\nEvaluating model...")
    results = detector.evaluate_model(test_dataset)
    
    print("\nEvaluation Results:")
    print(f"Accuracy: {results['accuracy']:.3f}")
    print(f"F1 Score (weighted): {results['f1_weighted']:.3f}")
    
    # Plot confusion matrix
    detector.plot_confusion_matrix(test_dataset)


if __name__ == "__main__":
    main()
