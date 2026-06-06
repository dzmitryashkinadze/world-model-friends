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
    file, num_sequences, max_context_length, test_ratio, val_ratio, output
):
    """Reads CSV, splits sequentially, generates, embeds and stores on disk."""
    try:
        # 1. Load CSV
        df = io.load_csv_to_polars(file)
        click.echo(f"Successfully loaded {file}")

        # 2. Split raw data sequentially
        test_df, val_df, train_df = sequencer.split_raw_data(df, test_ratio, val_ratio)

        # 3. Generate, Embed and Store for each split
        # Distribute the total num_sequences across the splits proportionally
        n_test = int(num_sequences * test_ratio)
        n_val = int(num_sequences * val_ratio)
        n_train = num_sequences - n_test - n_val

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

        # 4. Save folds
        t_df = test_df if test_df is not None else pl.DataFrame()
        v_df = val_df if val_df is not None else pl.DataFrame()
        tr_df = train_df if train_df is not None else pl.DataFrame()
        train_output, test_output, val_output = io.save_folds(tr_df, t_df, v_df, output)

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
