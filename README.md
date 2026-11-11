# SPRINT: Structured Processing using Robust Integrated Neural Templates

SPRINT is a document generation pipeline that takes loosely structured text (such as academic document excerpts) and converts it into structured XML format. 
This XML can then be rendered into formatted DOCX or PDF documents using a lightweight web app. The system employs a two-stage model pipeline:
- A base LLM (Sheared-LLaMA-1.3B) extracts semantic meaning.
- A LoRA-fine-tuned model  converts the interpretation into XML layout annotations.

---

## Project Structure

├── Dataset Preparation/
│   └── Speedy_Extraction.py         # Builds subset of DocBank with matched text
├── Preprocessing and FineTuning/
│   └── finetune.ipynb              # Tokenizes, applies loss masking, and fine-tunes LoRA
├── WebApp/
│   ├── app.py                      # Flask app with base+LoRA/HF fallback inference
│   └── templates/index.html        # Web frontend
├── Structured Processing...pdf     # Full project write-up/report
└── README.md                       # This file

---
For size considerations, the files of Dataset preperation have been doubled zipped.

---
Created by Sebastian Blogg
University of South Wales