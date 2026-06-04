import sys

import click

import world_model_friends.data_wrangling.loader as loader
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
        df = loader.load_csv_to_polars(file)
        click.echo(f"Successfully loaded {file}")

        # 2. Generate sequences
        sequences_df = sequencer.generate_sequences(
            df, num_sequences, max_context_length
        )
        click.echo(f"Generated {len(sequences_df)} sequences.")

        # 3. Prepare training data (includes embedding)
        all_names = df["Name"].unique().to_list()
        training_df = sequencer.prepare_training_data(sequences_df, all_names)
        click.echo("Prepared training data with embeddings.")

        # 4. Store on disk
        training_df.write_parquet(output)
        click.echo(f"Successfully saved processed data to {output}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
