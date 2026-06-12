"""
CLI tool for World Model Friends.

Provides commands for processing raw data into datasets and training the world model.
"""

import sys

import click

from world_model_friends import config
from world_model_friends.data_wrangling.compile_datasets import compile_datasets
from world_model_friends.data_wrangling.io import load_parquet_files
from world_model_friends.predictor.evaluate import evaluate_world_model
from world_model_friends.predictor.train import train_world_model


@click.group()
def cli() -> None:
    """World Model Friends CLI tool."""
    pass


@cli.command(name="process")
@click.option(
    "--raw_data_file_path",
    "-f",
    type=click.Path(exists=True),
    default=config.get_config("process", "raw_data_file_path"),
    help="Path to the CSV file.",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="data",
    help="Directory to save the processed parquet files.",
)
@click.option(
    "--num-sequences",
    "-n",
    type=int,
    default=config.get_config("process", "num_sequences"),
    help="Number of sequences to generate over all splits.",
)
@click.option(
    "--max-context-length",
    "-k",
    type=int,
    default=config.get_config("process", "max_context_length"),
    help="Maximum context length.",
)
@click.option(
    "--test-ratio",
    type=float,
    default=0.1,
    help="Proportion of raw data for testing.",
)
@click.option(
    "--val-ratio",
    type=float,
    default=0.1,
    help="Proportion of raw data for validation.",
)
def run_compile_datasets(
    raw_data_file_path: str,
    output_dir: str,
    num_sequences: int,
    max_context_length: int,
    test_ratio: float,
    val_ratio: float,
) -> None:
    """
    Reads CSV, splits sequentially, generates, embeds and stores on disk.

    Args:
        raw_data_file_path (str): Path to the CSV file.
        output_dir (str): Directory to save the processed data.
        num_sequences (int): Number of sequences to generate over all splits.
        max_context_length (int): Maximum context length.
        test_ratio (float): Proportion of raw data for testing.
        val_ratio (float): Proportion of raw data for validation.

    Returns:
        None
    """
    compile_datasets(
        raw_data_file_path=raw_data_file_path,
        output_dir=output_dir,
        num_sequences=num_sequences,
        max_context_length=max_context_length,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
    )


@cli.command(name="train")
@click.option(
    "--train-file",
    "-t",
    type=click.Path(),
    default=config.get_config("train", "train_file"),
    help="Path to the training parquet file.",
)
@click.option(
    "--val-file",
    "-v",
    type=click.Path(),
    default=config.get_config("train", "val_file"),
    help="Path to the validation parquet file.",
)
def run_train_world_model(train_file: str, val_file: str) -> None:
    """
    Trains the world model using the provided parquet files.

    Args:
        train_file (str): Path to the training parquet file.
        val_file (str): Path to the validation parquet file.

    Returns:
        None
    """
    try:
        train_df = load_parquet_files(train_file)
        val_df = load_parquet_files(val_file)

        train_world_model(train_df=train_df, val_df=val_df)
        click.echo("Training completed successfully.")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command(name="evaluate")
@click.option(
    "--model-path",
    "-m",
    type=click.Path(exists=True),
    required=True,
    help="Path to the model artifact (.pt).",
)
@click.option(
    "--test-file",
    "-v",
    type=click.Path(),
    default=config.get_config("train", "test_file"),
    help="Path to the test parquet files.",
)
def run_evaluate_world_model(model_path: str, test_file: str) -> None:
    """
    Evaluates a trained world model artifact.

    Args:
        model_path (str): Path to the model artifact (.pt).
        test_file (str): Path to the test parquet files.

    Returns:
        None
    """
    try:
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        test_df = load_parquet_files(test_file)
        evaluate_world_model(model_path, test_df, device)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
