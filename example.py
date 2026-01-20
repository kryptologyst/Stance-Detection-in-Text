#!/usr/bin/env python3
"""
Example script demonstrating the stance detection system.

This script shows how to use the StanceDetector class for various tasks
including single predictions, batch processing, and model evaluation.

Author: AI Assistant
Date: 2024
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from stance_detector import StanceDetector


def main():
    """Main example function."""
    print("🎯 Stance Detection System Demo")
    print("=" * 50)
    
    # Initialize detector
    print("Initializing stance detector...")
    detector = StanceDetector()
    
    # Load model
    print("Loading pre-trained model...")
    detector.load_model()
    print("✅ Model loaded successfully!\n")
    
    # Example 1: Single predictions
    print("📝 Example 1: Single Stance Predictions")
    print("-" * 40)
    
    examples = [
        ("climate change", "I strongly believe we need immediate action on climate change."),
        ("artificial intelligence", "AI poses serious risks to human employment and should be regulated."),
        ("vaccination", "Vaccines have saved millions of lives and are crucial for public health."),
        ("climate change", "Climate change is a hoax created by scientists for funding."),
        ("artificial intelligence", "AI will revolutionize healthcare and improve lives.")
    ]
    
    for i, (target, text) in enumerate(examples, 1):
        print(f"\nExample {i}:")
        print(f"Target: {target}")
        print(f"Text: {text}")
        
        result = detector.predict_stance(text, target)
        
        # Color coding for stance
        stance = result["label"]
        confidence = result["confidence"]
        
        if stance == "FAVOR":
            emoji = "🟢"
        elif stance == "AGAINST":
            emoji = "🔴"
        else:
            emoji = "🟡"
        
        print(f"Predicted Stance: {emoji} {stance}")
        print(f"Confidence: {confidence:.3f}")
    
    # Example 2: Batch prediction
    print(f"\n\n📊 Example 2: Batch Prediction")
    print("-" * 40)
    
    batch_texts = [
        "I support renewable energy initiatives",
        "Nuclear power is dangerous and should be banned",
        "I'm not sure about energy policy"
    ]
    batch_targets = ["renewable energy", "nuclear power", "energy policy"]
    
    print("Processing batch predictions...")
    batch_results = detector.batch_predict(batch_texts, batch_targets)
    
    print("\nBatch Results:")
    for i, result in enumerate(batch_results, 1):
        print(f"{i}. {result['label']} (confidence: {result['confidence']:.3f})")
    
    # Example 3: Synthetic dataset creation and evaluation
    print(f"\n\n🧪 Example 3: Model Evaluation")
    print("-" * 40)
    
    print("Creating synthetic dataset...")
    dataset = detector.create_synthetic_dataset(size=200)
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    eval_size = int(0.15 * len(dataset))
    
    train_dataset = dataset.select(range(train_size))
    eval_dataset = dataset.select(range(train_size, train_size + eval_size))
    test_dataset = dataset.select(range(train_size + eval_size, len(dataset)))
    
    print(f"Dataset split:")
    print(f"  Training: {len(train_dataset)} samples")
    print(f"  Evaluation: {len(eval_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    print("\nEvaluating model on test set...")
    results = detector.evaluate_model(test_dataset)
    
    print("\n📈 Evaluation Results:")
    print(f"  Accuracy: {results['accuracy']:.3f}")
    print(f"  Precision (weighted): {results['precision_weighted']:.3f}")
    print(f"  Recall (weighted): {results['recall_weighted']:.3f}")
    print(f"  F1 Score (weighted): {results['f1_weighted']:.3f}")
    
    print("\nPer-class metrics:")
    for stance in ["FAVOR", "AGAINST", "NONE"]:
        print(f"  {stance}:")
        print(f"    Precision: {results['precision_per_class'][stance]:.3f}")
        print(f"    Recall: {results['recall_per_class'][stance]:.3f}")
        print(f"    F1 Score: {results['f1_per_class'][stance]:.3f}")
    
    # Generate confusion matrix
    print("\nGenerating confusion matrix...")
    detector.plot_confusion_matrix(test_dataset)
    print("✅ Confusion matrix saved to ./results/confusion_matrix.png")
    
    print(f"\n🎉 Demo completed successfully!")
    print("\nNext steps:")
    print("1. Run 'streamlit run web_app/app.py' for interactive web interface")
    print("2. Use 'python src/cli.py --help' for command-line options")
    print("3. Check the README.md for more detailed usage examples")


if __name__ == "__main__":
    main()
