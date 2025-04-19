#!/usr/bin/env python
"""
Command-line interface for the model registry.
"""

import os
import sys
import argparse
import json
import logging
from typing import List, Optional
import pandas as pd

from .model_registry import ModelRegistry
from .model_registrar import ModelRegistrar

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def register_command(args):
    """Register a model from a training run."""
    registrar = ModelRegistrar(args.registry_dir)
    model_id = registrar.register_from_run(
        output_dir=args.output_dir,
        model_id=args.model_id,
        description=args.description,
        evaluation_file=args.eval_file
    )
    
    if model_id:
        logger.info(f"Successfully registered model with ID: {model_id}")
    else:
        logger.error("Failed to register model")
        sys.exit(1)


def list_command(args):
    """List models in the registry."""
    registry = ModelRegistry(args.registry_dir)
    df = registry.list_models(task_type=args.task_type, model_type=args.model_type)
    
    if df.empty:
        logger.info("No models found matching the criteria")
        return
    
    if args.sort:
        if args.sort in df.columns:
            df = df.sort_values(by=args.sort, ascending=not args.desc)
        else:
            logger.warning(f"Sort column '{args.sort}' not found in metrics")
    
    if args.output:
        # Save to file
        if args.output.endswith(".csv"):
            df.to_csv(args.output, index=False)
        elif args.output.endswith(".json"):
            df.to_json(args.output, orient="records", indent=2)
        else:
            df.to_csv(args.output, index=False)
        logger.info(f"Saved model list to {args.output}")
    else:
        # Print to console
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(df)


def compare_command(args):
    """Compare models."""
    registry = ModelRegistry(args.registry_dir)
    
    # Get model IDs from arguments or from registry
    model_ids = args.model_ids
    if not model_ids and args.task_type:
        # If no model IDs provided, compare all models of the specified task type and model type
        df = registry.list_models(task_type=args.task_type, model_type=args.model_type)
        model_ids = df["model_id"].tolist()
    
    if not model_ids:
        logger.error("No models specified for comparison")
        sys.exit(1)
    
    # Compare models
    comparison = registry.compare_models(
        model_ids=model_ids,
        metrics=args.metrics.split(",") if args.metrics else None,
        sort_by=args.sort,
        ascending=not args.desc
    )
    
    if comparison.empty:
        logger.info("No models found for comparison")
        return
    
    if args.output:
        # Save to file
        if args.output.endswith(".csv"):
            comparison.to_csv(args.output, index=False)
        elif args.output.endswith(".json"):
            comparison.to_json(args.output, orient="records", indent=2)
        else:
            comparison.to_csv(args.output, index=False)
        logger.info(f"Saved comparison to {args.output}")
    else:
        # Print to console
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(comparison)
    
    # Generate plot if requested
    if args.plot:
        plot_path = args.plot_output if args.plot_output else "model_comparison.png"
        registry.plot_model_comparison(
            model_ids=model_ids,
            metric=args.plot,
            output_path=plot_path,
            title=f"Model Comparison by {args.plot}"
        )
        logger.info(f"Saved plot to {plot_path}")


def best_command(args):
    """Show or export the best model(s)."""
    registry = ModelRegistry(args.registry_dir)
    
    if args.export:
        # Export best models to inference directory
        registrar = ModelRegistrar(args.registry_dir)
        exported = registrar.register_best_models_for_inference(
            inference_dir=args.export,
            task_types=[args.task_type] if args.task_type else None,
            model_types=[args.model_type] if args.model_type else None
        )
        
        logger.info(f"Exported {len(exported)} best models to {args.export}")
        
        # Print summary of exported models
        for key, path in exported.items():
            logger.info(f"  - {key}: {path}")
    else:
        # Show best model information
        best_model = registry.get_best_model(
            task_type=args.task_type,
            model_type=args.model_type
        )
        
        if best_model:
            if args.output:
                # Save to file
                with open(args.output, "w") as f:
                    json.dump(best_model, f, indent=2)
                logger.info(f"Saved best model info to {args.output}")
            else:
                # Print to console
                print(json.dumps(best_model, indent=2))
        else:
            logger.info(f"No best model found for task: {args.task_type}, model type: {args.model_type}")


def delete_command(args):
    """Delete a model from the registry."""
    registry = ModelRegistry(args.registry_dir)
    
    if not args.force:
        # Confirm deletion
        model = registry.get_model(args.model_id)
        if not model:
            logger.error(f"Model {args.model_id} not found in registry")
            sys.exit(1)
            
        print(f"About to delete model: {args.model_id}")
        print(f"Type: {model['model_type']}")
        print(f"Description: {model['description']}")
        
        confirm = input("Are you sure you want to delete this model? [y/N] ")
        if confirm.lower() not in ["y", "yes"]:
            logger.info("Deletion cancelled")
            return
    
    # Delete the model
    success = registry.delete_model(args.model_id)
    
    if success:
        logger.info(f"Successfully deleted model {args.model_id}")
    else:
        logger.error(f"Failed to delete model {args.model_id}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Model Registry CLI")
    parser.add_argument("--registry-dir", default="./model_registry", help="Path to the registry directory")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Register command
    register_parser = subparsers.add_parser("register", help="Register a model from a training run")
    register_parser.add_argument("--output-dir", required=True, help="Directory containing the training outputs")
    register_parser.add_argument("--model-id", help="Optional ID for the model (will be auto-generated if not provided)")
    register_parser.add_argument("--description", help="Optional description of the model")
    register_parser.add_argument("--eval-file", help="Path to evaluation results (relative to output_dir)")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List models in the registry")
    list_parser.add_argument("--task-type", help="Filter by task type (classification or regression)")
    list_parser.add_argument("--model-type", help="Filter by model type")
    list_parser.add_argument("--sort", help="Column to sort by")
    list_parser.add_argument("--desc", action="store_true", help="Sort in descending order")
    list_parser.add_argument("--output", help="Path to save the output (CSV or JSON)")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare models")
    compare_parser.add_argument("--model-ids", nargs="*", help="IDs of models to compare")
    compare_parser.add_argument("--task-type", help="Filter by task type (if model IDs not provided)")
    compare_parser.add_argument("--model-type", help="Filter by model type (if model IDs not provided)")
    compare_parser.add_argument("--metrics", help="Comma-separated list of metrics to include")
    compare_parser.add_argument("--sort", help="Metric to sort by")
    compare_parser.add_argument("--desc", action="store_true", help="Sort in descending order")
    compare_parser.add_argument("--output", help="Path to save the output (CSV or JSON)")
    compare_parser.add_argument("--plot", help="Metric to plot")
    compare_parser.add_argument("--plot-output", help="Path to save the plot")
    
    # Best command
    best_parser = subparsers.add_parser("best", help="Show or export the best model(s)")
    best_parser.add_argument("--task-type", required=True, help="Task type (classification or regression)")
    best_parser.add_argument("--model-type", help="Model type (if not provided, returns the overall best model)")
    best_parser.add_argument("--export", help="Directory to export the best model(s) to")
    best_parser.add_argument("--output", help="Path to save the output (JSON)")
    
    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a model from the registry")
    delete_parser.add_argument("model_id", help="ID of the model to delete")
    delete_parser.add_argument("--force", action="store_true", help="Skip confirmation")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    if args.command == "register":
        register_command(args)
    elif args.command == "list":
        list_command(args)
    elif args.command == "compare":
        compare_command(args)
    elif args.command == "best":
        best_command(args)
    elif args.command == "delete":
        delete_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main() 