#!/usr/bin/env python3
"""
Command Line Interface for Stance Detection

This module provides a command-line interface for the stance detection system.
Users can run predictions, evaluations, and model training from the terminal.

Author: AI Assistant
Date: 2024
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from stance_detector import StanceDetector


def load_config() -> dict:
    """Load configuration from JSON file."""
    config_path = Path(__file__).parent.parent / "config" / "config.json"
    with open(config_path, "r") as f:
        return json.load(f)


def predict_command(args):
    """Handle single prediction command."""
    detector = StanceDetector(model_name=args.model)
    detector.load_model()
    
    result = detector.predict_stance(args.text, args.target)
    
    print(f"\nTarget: {args.target}")
    print(f"Text: {args.text}")
    print(f"Predicted Stance: {result['label']}")
    print(f"Confidence: {result['confidence']:.3f}")
    
    if args.output:
        output_data = {
            "target": args.target,
            "text": args.text,
            "predicted_stance": result["label"],
            "confidence": result["confidence"]
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


def batch_predict_command(args):
    """Handle batch prediction command."""
    detector = StanceDetector(model_name=args.model)
    detector.load_model()
    
    # Load input data
    with open(args.input, "r") as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        print("Error: Input file should contain a list of dictionaries with 'text' and 'target' keys")
        return
    
    texts = [item["text"] for item in data]
    targets = [item["target"] for item in data]
    
    print(f"Processing {len(texts)} predictions...")
    results = detector.batch_predict(texts, targets)
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Batch prediction completed. Results saved to {args.output}")
    
    # Print summary
    stance_counts = {}
    for result in results:
        stance = result["label"]
        stance_counts[stance] = stance_counts.get(stance, 0) + 1
    
    print("\nSummary:")
    for stance, count in stance_counts.items():
        print(f"  {stance}: {count}")


def evaluate_command(args):
    """Handle model evaluation command."""
    detector = StanceDetector(model_name=args.model)
    detector.load_model()
    
    print("Generating synthetic dataset...")
    dataset = detector.create_synthetic_dataset(size=args.dataset_size)
    
    # Split dataset
    config = load_config()
    train_split = config["data"]["train_split"]
    eval_split = config["data"]["eval_split"]
    
    train_size = int(train_split * len(dataset))
    eval_size = int(eval_split * len(dataset))
    
    train_dataset = dataset.select(range(train_size))
    eval_dataset = dataset.select(range(train_size, train_size + eval_size))
    test_dataset = dataset.select(range(train_size + eval_size, len(dataset)))
    
    print(f"Dataset split - Train: {len(train_dataset)}, Eval: {len(eval_dataset)}, Test: {len(test_dataset)}")
    
    print("Evaluating model...")
    results = detector.evaluate_model(test_dataset, output_path=args.output)
    
    print("\nEvaluation Results:")
    print(f"Accuracy: {results['accuracy']:.3f}")
    print(f"Precision (weighted): {results['precision_weighted']:.3f}")
    print(f"Recall (weighted): {results['recall_weighted']:.3f}")
    print(f"F1 Score (weighted): {results['f1_weighted']:.3f}")
    
    print("\nPer-class metrics:")
    for stance in ["FAVOR", "AGAINST", "NONE"]:
        print(f"  {stance}:")
        print(f"    Precision: {results['precision_per_class'][stance]:.3f}")
        print(f"    Recall: {results['recall_per_class'][stance]:.3f}")
        print(f"    F1 Score: {results['f1_per_class'][stance]:.3f}")
    
    # Generate confusion matrix
    detector.plot_confusion_matrix(test_dataset)
    print(f"\nConfusion matrix saved to ./results/confusion_matrix.png")


def train_command(args):
    """Handle model training command."""
    detector = StanceDetector(model_name=args.model)
    detector.load_model()
    
    print("Generating synthetic dataset...")
    dataset = detector.create_synthetic_dataset(size=args.dataset_size)
    
    # Split dataset
    config = load_config()
    train_split = config["data"]["train_split"]
    eval_split = config["data"]["eval_split"]
    
    train_size = int(train_split * len(dataset))
    eval_size = int(eval_split * len(dataset))
    
    train_dataset = dataset.select(range(train_size))
    eval_dataset = dataset.select(range(train_size, train_size + eval_size))
    test_dataset = dataset.select(range(train_size + eval_size, len(dataset)))
    
    print(f"Dataset split - Train: {len(train_dataset)}, Eval: {len(eval_dataset)}, Test: {len(test_dataset)}")
    
    print("Starting fine-tuning...")
    detector.fine_tune_model(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    print(f"Training completed! Model saved to {args.output_dir}")


def create_dataset_command(args):
    """Handle dataset creation command."""
    detector = StanceDetector()
    
    print(f"Creating synthetic dataset with {args.size} samples...")
    dataset = detector.create_synthetic_dataset(size=args.size)
    
    # Convert to list for JSON serialization
    data = [{"text": item["text"], "target": item["target"], "label": item["label"]} 
            for item in dataset]
    
    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Dataset saved to {args.output}")
    
    # Print statistics
    targets = [item["target"] for item in data]
    labels = [item["label"] for item in data]
    
    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(data)}")
    print(f"Unique targets: {len(set(targets))}")
    print(f"Unique labels: {len(set(labels))}")
    
    print(f"\nTarget distribution:")
    for target in set(targets):
        count = targets.count(target)
        print(f"  {target}: {count}")
    
    print(f"\nLabel distribution:")
    for label in set(labels):
        count = labels.count(label)
        print(f"  {label}: {count}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Stance Detection CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single prediction
  python cli.py predict --target "climate change" --text "I support climate action"
  
  # Batch prediction
  python cli.py batch-predict --input data.json --output results.json
  
  # Model evaluation
  python cli.py evaluate --dataset-size 500 --output results.json
  
  # Model training
  python cli.py train --dataset-size 1000 --epochs 3 --output-dir ./models/fine_tuned
  
  # Create dataset
  python cli.py create-dataset --size 1000 --output dataset.json
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Single prediction")
    predict_parser.add_argument("--target", required=True, help="Target topic")
    predict_parser.add_argument("--text", required=True, help="Text to analyze")
    predict_parser.add_argument("--model", default="cardiffnlp/twitter-roberta-base-stance-climate", 
                               help="Model to use")
    predict_parser.add_argument("--output", help="Output file for results")
    
    # Batch predict command
    batch_parser = subparsers.add_parser("batch-predict", help="Batch prediction")
    batch_parser.add_argument("--input", required=True, help="Input JSON file")
    batch_parser.add_argument("--output", required=True, help="Output JSON file")
    batch_parser.add_argument("--model", default="cardiffnlp/twitter-roberta-base-stance-climate", 
                             help="Model to use")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Model evaluation")
    eval_parser.add_argument("--dataset-size", type=int, default=500, 
                            help="Size of synthetic dataset")
    eval_parser.add_argument("--output", default="./results/evaluation_results.json", 
                            help="Output file for results")
    eval_parser.add_argument("--model", default="cardiffnlp/twitter-roberta-base-stance-climate", 
                            help="Model to use")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Model training")
    train_parser.add_argument("--dataset-size", type=int, default=1000, 
                             help="Size of synthetic dataset")
    train_parser.add_argument("--output-dir", default="./models/fine_tuned", 
                             help="Output directory for model")
    train_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    train_parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    train_parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    train_parser.add_argument("--model", default="cardiffnlp/twitter-roberta-base-stance-climate", 
                             help="Base model to fine-tune")
    
    # Create dataset command
    dataset_parser = subparsers.add_parser("create-dataset", help="Create synthetic dataset")
    dataset_parser.add_argument("--size", type=int, default=1000, help="Dataset size")
    dataset_parser.add_argument("--output", required=True, help="Output JSON file")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "predict":
            predict_command(args)
        elif args.command == "batch-predict":
            batch_predict_command(args)
        elif args.command == "evaluate":
            evaluate_command(args)
        elif args.command == "train":
            train_command(args)
        elif args.command == "create-dataset":
            create_dataset_command(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
