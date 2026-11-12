# SPRINT: Structured Processing using Robust Integrated Neural Templates

[Read the full Thesis (PDF)](https://sebulah-b.github.io/SPRINT_Thesis.pdf) | [Processed Training Data (Hugging Face)](https://huggingface.co/datasets/Sebulah/SPRINT-DocBank-Subset) | [Original DocBank Dataset](https://github.com/doc-analysis/DocBank)

SPRINT is an applied machine learning pipeline and inference application designed to transform unstructured text into strictly formatted, layout-aware XML representations.

*Note on Data Provenance: The model was trained using a custom-engineered JSONL subset of the open-source [DocBank dataset](https://github.com/doc-analysis/DocBank). All credit for the raw data curation belongs to the original authors.*

## System Architecture

* **Base Model:** Sheared-LLaMA 1.3B (distilled from LLaMA2-7B via structured pruning).
* **Fine-Tuning:** Parameter-Efficient Fine-Tuning (PEFT) using LoRA (Rank 8, Alpha 16) applied to the query and value projection layers. Trained with 8-bit quantization to minimize VRAM overhead.
* **Dataset:** 50,000 to 100,000 instruction-tuned text-XML pairs curated from the DocBank corpus (arXiv LaTeX extractions).
* **Inference Pipeline:**
  1. **Interpretation:** Raw user input is passed to the base Sheared-LLaMA model to generate a coherent natural language response.
  2. **Refinement:** The base output is chained into a secondary prompt and processed by the LoRA-adapted model to strictly generate the bounding-box and semantic XML layout.

## Evaluation & Metrics

The model was evaluated strictly on structural fidelity and syntactic precision:

* **BLEU & ROUGE Scores:** The raw 50k paired fine-tune (without instruction overhead) achieved the highest lexical precision, scoring a 0.51 BLEU and outperforming instruction-heavy variants on ROUGE-L (Longest Common Subsequence).
* **XML Well-Formedness (Post-Mortem):** The pipeline achieved a 0% exact-match well-formedness score. While the ROUGE-L scores indicate the model successfully learned internal structural hierarchies, the autoregressive generation struggled with strict XML closure tags (`</annotation>`).
* **Future Optimizations:** This highlights a critical limitation in applying standard causal language models to rigid syntactic generation without forced decoding constraints or explicit end-of-sequence token emphasis.

## Repository Structure

* `/Dataset Preperation`: Scripts for parsing the raw DocBank `.txt` and `.json` annotations into tokenized `.jsonl` input-output triplets.
* `/Preprocessing and FineTuning`: PyTorch training loops, LoRA configurations, and dataset masking scripts for the Sheared-LLaMA adaptation.
* `/WebApp`: Flask application handling the two-stage local inference and Hugging Face API fallback routing.

## Local Setup

To run the inference application locally:

1. Clone the repository and install the dependencies:
   ```bash
   git clone [https://github.com/Sebulah-B/SPRINT.git](https://github.com/Sebulah-B/SPRINT.git)
   cd SPRINT/WebApp
   pip install -r requirements.txt
   ```
2. (Optional) Set your Hugging Face API key as an environment variable to enable the remote inference fallback.
3. Launch the Flask server:
   ```bash
   python app.py
   ```
4. Access the UI at `http://127.0.0.1:5000` to submit raw text and view the staged XML generation.
