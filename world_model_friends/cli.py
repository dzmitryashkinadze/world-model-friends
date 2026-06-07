"""
CLI tool for World Model Friends.

Provides commands for processing raw data into datasets and training the world model.
"""

import sys

import click
import polars as pl

import world_model_friends.data_wrangling.io as io
import world_model_friends.data_wrangling.script_sequencer as sequencer
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
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=config.get_config("process", "output"),
    help="Path to save the processed data.",
)
def compile_datasets(
    raw_data_file_path: str,
    num_sequences: int,
    max_context_length: int,
    test_ratio: float,
    val_ratio: float,
    output: str,
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
    type=click.Path(exists=True),
    default=config.get_config("train", "train_file"),
    help="Path to the training parquet file.",
)
@click.option(
    "--val-file",
    "-v",
    type=click.Path(exists=True),
    default=config.get_config("train", "val_file"),
    help="Path to the validation parquet file.",
)
def train_world_model(train_file: str, val_file: str) -> None:
    """
    Trains the world model using the provided parquet files.

    Args:
        train_file (str): Path to the training parquet file.
        val_file (str): Path to the validation parquet file.

    Returns:
        None
    """
    try:
        train_df = pl.read_parquet(train_file)
        val_df = pl.read_parquet(val_file)
        train.main(train_df, val_df)
        click.echo("Training completed successfully.")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
