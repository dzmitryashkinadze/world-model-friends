import sys

import click

import data_wrangling.loader as loader


@click.group()
def cli():
    """World Model Friends CLI tool."""
    pass


@cli.command(name="load")
@click.option(
    "--file", "-f", type=click.Path(exists=True), help="Path to the CSV file."
)
def load_data(file):
    """Loads a CSV file using the data_wrangling loader."""
    try:
        df = loader.load_csv_to_polars(file)
        click.echo(f"Successfully loaded {file}")
        click.echo(df.head())
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
