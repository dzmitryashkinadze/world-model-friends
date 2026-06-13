"""
CLI tool for World Model Friends.

Provides commands for processing raw data into datasets and training the world model.
"""

import sys

import click

from world_model_friends.config import get_config
from world_model_friends.data_wrangling.compile_datasets import compile_datasets
from world_model_friends.data_wrangling.io import load_parquet_files
from world_model_friends.decoder.vector_search_decoder import (
    VectorSearchDecoder,
)
from world_model_friends.encoder.embeddings import embed_inference_request
from world_model_friends.predictor.evaluate import evaluate_world_model
from world_model_friends.predictor.inference import infer
from world_model_friends.predictor.train import train_world_model


@click.group()
def cli() -> None:
    """World Model Friends CLI tool."""
    pass


@cli.command(name="process")
@click.option(
    "--raw_data_file_path",
    type=click.Path(exists=True),
    default=get_config(section="process", key="raw_data_file_path"),
    help="Path to the CSV file.",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="data",
    help="Directory to save the processed parquet files.",
)
@click.option(
    "--n-sequences",
    type=int,
    default=get_config(section="process", key="n_sequences"),
    help="Number of sequences to generate over all splits.",
)
@click.option(
    "--max-context-length",
    type=int,
    default=get_config(section="process", key="max_context_length"),
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
    n_sequences: int,
    max_context_length: int,
    test_ratio: float,
    val_ratio: float,
) -> None:
    """
    Reads CSV, splits sequentially, generates, embeds and stores on disk.

    Args:
        raw_data_file_path (str): Path to the CSV file.
        output_dir (str): Directory to save the processed data.
        n_sequences (int): Number of sequences to generate over all splits.
        max_context_length (int): Maximum context length.
        test_ratio (float): Proportion of raw data for testing.
        val_ratio (float): Proportion of raw data for validation.

    Returns:
        None
    """
    compile_datasets(
        raw_data_file_path=raw_data_file_path,
        output_dir=output_dir,
        n_sequences=n_sequences,
        max_context_length=max_context_length,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
    )


@cli.command(name="train")
@click.option(
    "--train-file",
    type=click.Path(),
    default=get_config(section="train", key="train_file"),
    help="Path to the training parquet file.",
)
@click.option(
    "--val-file",
    type=click.Path(),
    default=get_config(section="train", key="val_file"),
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
        train_df = load_parquet_files(pattern=train_file)
        val_df = load_parquet_files(pattern=val_file)

        train_world_model(train_df=train_df, val_df=val_df)
        click.echo(message="Training completed successfully.")

    except Exception as e:
        click.echo(message=f"Error: {e}", err=True)
        sys.exit(1)


@cli.command(name="evaluate")
@click.option(
    "--model-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the model artifact (.pt).",
)
@click.option(
    "--test-file",
    type=click.Path(),
    default=get_config(section="train", key="test_file"),
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

        test_df = load_parquet_files(pattern=test_file)
        evaluate_world_model(model_path, test_df, device)

    except Exception as e:
        click.echo(message=f"Error: {e}", err=True)
        sys.exit(1)


@cli.command(name="inference")
@click.option(
    "--context",
    type=click.STRING,
    required=True,
    help="Context dialogue",
)
@click.option(
    "--context_names",
    type=click.STRING,
    multiple=True,
    default=["Chandler"],
    help="Names of people from the context dialogue [Joey, Rachel, etc]",
)
@click.option(
    "--target_name",
    type=click.STRING,
    default="Joey",
    help="Target name (Joey, Rachel, Monica, etc)",
)
@click.option(
    "--model-path",
    type=click.Path(exists=True),
    default=get_config(section="train", key="model_artifact_path"),
    help="Path to the model artifact (.pt).",
)
def run_world_model_inference(
    context: str,
    context_names: list[str],
    target_name: str,
    model_path: str,
) -> str:
    """
    Runs the world model inference

    Args:
        context: Context for world model prediction
        target_identity: Target identity for the prediction
        model_path: Model path

    Returns:
        Predicted scipt line (answer).
    """
    try:
        # 2. Embed inference request
        request = embed_inference_request(
            context_names=context_names,
            context_text=context,
            target_name=target_name,
        )

        # 3. Run inference
        target_embedding = infer(request, model_path=model_path)

        # 4. Decode
        decoder_cls = VectorSearchDecoder(
            script_with_line_embeddings_path=get_config(
                section="process", key="script_with_line_embeddings_path"
            )
        )
        results = decoder_cls.decode(
            target_embedding=target_embedding, speaker=target_name
        )

        # Return the top result
        if results:
            top_result = results[0]
            click.echo(message=top_result["Lines"])
            return top_result["Lines"]
        else:
            click.echo(message="No results found from decoder.", err=True)
            return ""

    except Exception as e:
        click.echo(message=f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
