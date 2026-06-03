# Friends World Model Project

An experimental project aimed at training and benchmarking a **latent-space world model**. The goal is to predict the *semantic embedding* of the next dialogue turn in a *Friends* episode script, learning the underlying narrative flow and character transitions in a compressed latent representation.

## Architecture

The system is implemented as a **Click-based CLI**, providing a modular workflow for data processing and modeling.

### Tooling & Development
- **Dependency Management**: [uv](https://github.com/astral-sh/uv)
- **Code Quality**: `pre-commit` for linting and running tests.
- **Development Guide**: See [DEVELOPMENT.md](DEVELOPMENT.md) for strict coding standards.

### Core Components
1. **Data Loader & Sequencer**: Processes raw CSV data into sliding window sequences.
2. **Embedding Engine**: Generates hybrid embeddings (semantic for the dialogue + one hot for the identity of those involved in the dialogue).
3. **SQLite Storage**: A persistent database used to store and index the generated embeddings, facilitating efficient training and retrieval.
4. **World Model (Deep Learning)**: A transition model trained to predict the next latent state (upcoming in development).
### CLI Workflow (Planned)
* `create_datasets`: Load CSV, generate sequences, and store embeddings in SQLite.
* `train`: Train the latent transition model.

## Project Roadmap

### 1. Data Loading
* **Goal:** Load the structured dialogue dataset.
* **Source:** [Kaggle Friends Script Dataset](https://www.kaggle.com/datasets/kimmik123/friends-scriptcsv) (`data/Friends_script.csv`, Columns: `Name`, `Lines`).

### 2. Training Dataset Compilation
* **Goal:** Construct a structured dataset for supervised learning.
* **Pipeline:**
    1. **Sequence Generation**: Apply a sliding window of variable size $N \in [1, 10]$ to the dialogue stream.
    2. **Hybrid Embedding**: For each turn in the sequence, generate a dual-component representation:
        * **Semantic Component**: Lightweight sentence transformer (`all-MiniLM-L6-v2`).
        * **Identity Component**: One-hot encoded speaker identity.
    3. **Triplet Structuring**: Organize sequences into triplets formatted for training:
        * **Inputs**: 
            1. **Speaker Sequence**: A sequence of one-hot vectors for the $N$ speakers.
            2. **Dialogue Sequence**: A sequence of semantic embeddings for the $N$ turns.
            3. **Target Speaker**: A one-hot vector representing the identity of the next speaker.
        * **Target**: The semantic embedding of the $(N+1)$-th dialogue turn.
    4. **Data Partitioning**: Split the resulting dataset into **Training**, **Validation**, and **Test** sets.

### 3. World Model Training
* **Goal:** Train a predictor model that maps a sequence of context to the next semantic state.
* **Training Inputs:**
    1. **Speaker Sequence:** A sequence of one-hot vectors representing the identity of the speaker for each turn in the sequence.
    2. **Dialogue Sequence:** A sequence of semantic embeddings for the dialogue turns.
    3. **Target Speaker:** A one-hot vector representing the identity of the speaker in the next turn.
* **Output:** The semantic embedding of the next dialogue turn (the "answer").
* **Architecture:** A transition model (e.g., an MLP or a small Transformer) that processes these inputs to predict the next semantic state.

### 4. Output Interpretation
* **Goal:** Convert abstract latent predictions into human-readable dialogue to make the model's output user-friendly.
* **Method:**
    1. **Targeted Search**: Take the predicted semantic embedding and perform a cosine similarity search in the SQLite database.
    2. **Character Filtering**: Restrict the search space to dialogue turns belonging to the predicted `Target Speaker`.
    3. **Nearest Neighbor Projection**: Return the text of the most similar dialogue turn (the one with the highest cosine similarity) to represent the model's predicted next state.
