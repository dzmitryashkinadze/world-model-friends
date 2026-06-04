import sys

import click

import world_model_friends.data_wrangling.io as io
import world_model_friends.data_wrangling.script_sequencer as sequencer


@click.group()
def cli():
    """World Model Friends CLI tool."""
    pass


@cli.command(name="process")
@click.option(
    "--file",
    "-f",
    type=click.Path(exists=True),
    default="data/Friends_script.csv",
    help="Path to the CSV file.",
)
@click.option(
    "--num-sequences",
    "-n",
    type=int,
    default=10,
    help="Number of sequences to generate.",
)
@click.option(
    "--max-context-length",
    "-k",
    type=int,
    default=5,
    help="Maximum context length.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="data/processed_data.parquet",
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
        all_names = df["Name"].unique().to_list()
        training_df = sequencer.embed_sequences(sequences_df, all_names)
        click.echo("Embedded sequences.")

        # 4. Split into train, test, and val folds
        train_df, val_df, test_df = sequencer.split_data(training_df)
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


if __name__ == "__main__":
    cli()
