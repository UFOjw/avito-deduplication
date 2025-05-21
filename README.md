# Duplicate Listing Detection

This project focuses on developing an algorithm to detect duplicate listings using a combination of textual information, image embeddings, and structured data attributes. It integrates multiple modalities to improve the robustness of duplicate detection in real-world classified advertisement datasets.

## Project Structure

* `prepared text.ipynb` — Preprocessing and embedding of textual data.
* `triplet creation.ipynb` — Construction of training data using triplet sampling for metric learning.
* `train of a text model.ipynb` — Training a text-based model.
* `explore and visual emb.ipynb` — Exploration and building of visual embeddings.
* `merged models and extracted emb.ipynb` — Merging modalities and extracting joint embeddings.
* `GBDT.ipynb` — Using Gradient Boosted Decision Trees accompany merged modalities.
* `baseline.ipynb` — Establishing baseline metrics using simple heuristics or single-modality models.

## Features

* **Multimodal Embedding**: Combines text, image, and metadata embeddings.
* **Triplet Loss Training**: Trains embeddings to bring similar listings closer in vector space.
* **Visual and Textual Alignment**: Uses separate encoders and aligns their latent representations.
* **Gradient Boosting for Final Decision**: CatBoost classifier refines predictions using feature distances.

## Solution to the Problem

The key challenge in duplicate detection lies in variability — listings might contain different wording, image angles, or incomplete data. To address this:

* We preprocess text into dense semantic embeddings using pretrained transformers.
* Visual content is processed through pretrained image encoders (e.g., CLIP, SigLIP) to extract meaningful features.
* A triplet loss strategy ensures the model learns to distinguish true duplicates from hard negatives.
* Embeddings are merged or compared using various strategies (concatenation, distance metrics).
* A CatBoost classifier is trained on the combined feature space to produce a final duplicate probability score.
* This approach minimizes false positives while remaining scalable for production.
