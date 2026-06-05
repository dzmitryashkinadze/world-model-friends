import sys

import click
import polars as pl

import world_model_friends.data_wrangling.io as io
import world_model_friends.data_wrangling.script_sequencer as sequencer
import world_model_friends.world_model.train as train
from world_model_friends import config


@click.group()
def cli():
    """World Model Friends CLI tool."""
    pass


@cli.command(name="process")
@click.option(
    "--file",
    "-f",
    type=click.Path(exists=True),
    default=config.get_config("process", "file"),
    help="Path to the CSV file.",
)
@click.option(
    "--num-sequences",
    "-n",
    type=int,
    default=config.get_config("process", "num_sequences"),
    help="Number of sequences to generate.",
)
@click.option(
    "--max-context-length",
    "-k",
    type=int,
    default=config.get_config("process", "max_context_length"),
    help="Maximum context length.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=config.get_config("process", "output"),
    help="Path to save the processed data.",
)
def compile_datasets(file, num_sequences, max_context_length, output):
    """Reads CSV, picks sequences, embeds them and stores on disk."""
    try:
        # 1. Load CSV
        df = io.load_csv_to_polars(file)
        click.echo(f"Successfully loaded {file}")

        # 2. Generate sequences
        sequences_df = sequencer.generate_sequences(
            df, num_sequences, max_context_length
        )
        click.echo(f"Generated {len(sequences_df)} sequences.")

        # 3. Embed sequences
        sequences_df = sequencer.embed_sequences(sequences_df)
        click.echo("Embedded sequences.")

        # 4. Split into train, test, and val folds
        train_df, val_df, test_df = sequencer.split_data(sequences_df)
        click.echo("Split training data into train, test, and val folds.")

        # 5. Store on disk
        train_output, test_output, val_output = io.save_folds(
            train_df, test_df, val_df, output
        )

        click.echo(
            f"Successfully saved folds to: {train_output}, {test_output}, {val_output}"
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
def train_world_model(train_file, val_file):
    """Trains the world model using the provided parquet files."""
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
