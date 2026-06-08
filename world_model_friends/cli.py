"""
CLI tool for World Model Friends.

Provides commands for processing raw data into datasets and training the world model.
"""

import sys

import click

import world_model_friends.data_wrangling.io as io
import world_model_friends.data_wrangling.script_sequencer as sequencer
import world_model_friends.world_model.evaluate as evaluate
import world_model_friends.world_model.train as train
from world_model_friends import config


@click.group()
def cli() -> None:
    """World Model Friends CLI tool."""
    pass


@cli.command(name="process")
@click.option(
    "--raw_data_file_path",
    "-f",
    type=click.Path(exists=True),
    default=config.get_config("process", "file_path"),
    help="Path to the CSV file.",
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
def compile_datasets(
    raw_data_file_path: str,
    num_sequences: int,
    max_context_length: int,
    test_ratio: float,
    val_ratio: float,
) -> None:
    """
    Reads CSV, splits sequentially, generates, embeds and stores on disk.

    Args:
        raw_data_file_path (str): Path to the CSV file.
        num_sequences (int): Number of sequences to generate over all splits.
        max_context_length (int): Maximum context length.
        test_ratio (float): Proportion of raw data for testing.
        val_ratio (float): Proportion of raw data for validation.
        output (str): Path to save the processed data.

    Returns:
        None
    """
    try:
        # 1. Load CSV
        df = io.load_csv_to_polars(raw_data_file_path=raw_data_file_path)
        click.echo(f"Successfully loaded {raw_data_file_path}")

        # 2. Split raw data sequentially
        test_df, val_df, train_df = sequencer.split_raw_data(
            df=df, test_ratio=test_ratio, val_ratio=val_ratio
        )

        # Distribute the total num_sequences across the splits proportionally
        n_test = int(num_sequences * test_ratio)
        n_val = int(num_sequences * val_ratio)
        n_train = num_sequences - n_test - n_val

        # 3. Generate, Embed and Store for each split
        # processing
        test_df = sequencer.process_split(
            split_df=test_df,
            split_name="test",
            n_sequences=n_test,
            max_context_length=max_context_length,
        )
        val_df = sequencer.process_split(
            split_df=val_df,
            split_name="val",
            n_sequences=n_val,
            max_context_length=max_context_length,
        )
        train_df = sequencer.process_split(
            split_df=train_df,
            split_name="train",
            n_sequences=n_train,
            max_context_length=max_context_length,
        )

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


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
@click.option(
    "--max-files",
    "-m",
    type=int,
    default=config.get_config("train", "max_files"),
    help="Limit the number of files to read.",
)
def train_world_model(train_file: str, val_file: str, max_files: int) -> None:
    """
    Trains the world model using the provided parquet files.

    Args:
        train_file (str): Path to the training parquet file.
        val_file (str): Path to the validation parquet file.
        max_files (int): Limit the number of files to read.

    Returns:
        None
    """
    try:
        train_df = io.load_parquet_files(train_file, max_files)
        val_df = io.load_parquet_files(val_file, max_files)

        train.main(train_df, val_df)
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
@click.option(
    "--max-files",
    "-f",
    type=int,
    default=config.get_config("train", "max_files"),
    help="Limit the number of files to read.",
)
def evaluate_model(model_path: str, test_file: str, max_files: int) -> None:
    """
    Evaluates a trained world model artifact.

    Args:
        model_path (str): Path to the model artifact (.pt).
        test_file (str): Path to the test parquet files.
        max_files (int): Limit the number of files to read.

    Returns:
        None
    """
    try:
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        test_df = io.load_parquet_files(test_file, max_files)
        evaluate.evaluate(model_path, test_df, device)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
