# Friends World Model Project

An experimental project aimed at training and benchmarking a **latent-space world model**. The goal is to predict the *semantic embedding* of the next dialogue turn in a *Friends* episode script, learning the underlying narrative flow and character transitions in a compressed latent representation.

## Project Roadmap

### 1. Data Loading
* **Goal:** Load the structured dialogue dataset.
* **Source:** `data/Friends_script.csv` (Columns: `Name`, `Lines`).

### 2. Sequence Generation (Sliding Window)
* **Goal:** Transform the dialogue stream into a supervised learning dataset.
* **Method:** Use a sliding window of size $N=10$.
    * **Input:** A sequence of 10 consecutive dialogue turns $(T_1, T_2, \dots, T_{10})$.
    * **Target:** The $(N+1)$-th dialogue turn ($T_{11}$).
    * Each turn $T_i$ is a tuple of `(Speaker, Text)`.

### 3. Embedding (Hybrid Representation)
* **Goal:** Convert text turns into high-dimensional latent vectors, enriched with identity information.
* **Method:** Use a dual-component embedding approach for each turn $T_i$:
    1.  **Semantic Component:** Use a lightweight sentence transformer to embed the text content.
    2.  **Identity Component:** A one-hot encoded vector representing the speaker.
        *   **Categories:** `[Rachel, Monica, Chandler, Joey, Phoebe, Ross, Other]`.
* **Hybrid Vector:** $\text{emb}_i = [\text{emb}_{\text{semantic}} ; \text{emb}_{\text{identity}}]$.
* **Lightweight Embedding Model `sentence-transformers/all-MiniLM-L6-v2`: Extremely fast and lightweight.

### 4. World Model Training
* **Goal:** Train a predictor model that maps a sequence of embeddings to the next embedding.
* **Architecture:** A transition model (e.g., an MLP or a small Transformer) $f(\text{emb}_{n-9}, \dots, \text{emb}_{n}) \rightarrow \text{emb}_{n+1}$.

### 5. Benchmarking Strategy
* **Goal:** Compare the "Latent Predictor" against conventional Large Language Models (LLMs).
* **Metric:** **Cosine Similarity Score.**
    * Measure $\text{cos}(\text{Predicted\_Emb}, \text{Actual\_Next\_Emb})$.
* **Baselines:**
    * **LLM Baseline:** Use an LLM (e.g., GPT-2 or Llama) to generate the text of the next chunk, then embed that generated text, and compare it to the ground truth embedding.
    * **Naive Baseline:** Predict the mean embedding of the entire dataset (the "average" scene).
    * **Random Walk:** Predict the embedding of a randomly selected chunk.
* **Evaluation:** Test the model's ability to handle "narrative drift" and see if the latent predictor can maintain higher cosine similarity than the text-generation-based LLM approach.
