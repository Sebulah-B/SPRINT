import os
import tempfile
import uuid
from flask import Flask, request, render_template, send_file
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from peft import PeftModel
from huggingface_hub import InferenceApi, login
from pathlib import Path
from xml.etree import ElementTree as ET
from docx import Document
from docx.shared import Pt
from docx.oxml import parse_xml
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from fpdf import FPDF

# ─── CONFIG ────────────────────────────────────────────────────────────────

LOCAL_MODEL_DIR = "Model"
BASE_MODEL_ID = "princeton-nlp/Sheared-LLaMA-1.3B"
HF_MODEL_ID = "Sebulah/JOLT_Llama"

DEMO_BASE_ONLY = False   # Only shortcut base
DEMO_FULL = False        # Shortcut both base and LoRA

HF_TOKEN = os.environ.get("HF_TOKEN")
hf_inference = None
if HF_TOKEN:
    login()
    hf_inference = InferenceApi(repo_id=HF_MODEL_ID, token=HF_TOKEN, task="text-generation")

# ─── LOAD MODELS ───────────────────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)
tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

local_tokenizer_path = Path(LOCAL_MODEL_DIR)
local_tokenizer = LlamaTokenizer.from_pretrained(str(local_tokenizer_path), use_fast=True)
local_tokenizer.pad_token = local_tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID).to("cuda" if torch.cuda.is_available() else "cpu")
local_model = PeftModel.from_pretrained(base_model, LOCAL_MODEL_DIR).to("cuda" if torch.cuda.is_available() else "cpu")
base_model.eval()
local_model.eval()

# ─── UTILS ─────────────────────────────────────────────────────────────────

def is_well_formed_xml(xml_text):
    try:
        ET.fromstring(f"<root>{xml_text}</root>")
        return True
    except ET.ParseError:
        return False

def xml_to_docx_and_pdf(xml_string):
    PIXEL_TO_PT = 0.75
    doc = Document()
    section = doc.sections[0]
    doc.add_heading("XML Annotation", level=1)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.multi_cell(0, 10, "XML Annotation\n", align="L")

    try:
        root = ET.fromstring(f"<root>{xml_string}</root>")

        for obj in root.findall("object"):
            name = obj.find("name").text if obj.find("name") is not None else "unknown"
            text = obj.find("text").text if obj.find("text") is not None else ""

            bndbox = obj.find("bndbox")
            if bndbox is not None:
                xmin = int(bndbox.find("xmin").text or 0)
                ymin = int(bndbox.find("ymin").text or 0)
                xmax = int(bndbox.find("xmax").text or 100)
                ymax = int(bndbox.find("ymax").text or 50)

                paragraph = doc.add_paragraph()
                run = paragraph.add_run(text)
                run.font.size = Pt(10)
                paragraph.alignment = WD_PARAGRAPH_ALIGNMENT.LEFT

                p = paragraph._p
                pPr = p.get_or_add_pPr()
                pBdr = parse_xml(r"""
                    <w:pBdr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
                        <w:top w:val="single" w:sz="4" w:space="2" w:color="auto"/>
                        <w:left w:val="single" w:sz="4" w:space="2" w:color="auto"/>
                        <w:bottom w:val="single" w:sz="4" w:space="2" w:color="auto"/>
                        <w:right w:val="single" w:sz="4" w:space="2" w:color="auto"/>
                    </w:pBdr>
                """)
                pPr.append(pBdr)

                for _ in range(max(0, ymin // 20)):
                    doc.add_paragraph("")
                doc.add_paragraph(f"(Bounding box: ({xmin}, {ymin}) → ({xmax}, {ymax}))").italic = True

                # PDF
                pdf.multi_cell(0, 10, f"\nObject: {name}\nText: {text}\nBox: ({xmin},{ymin}) to ({xmax},{ymax})\n", align="L")

    except ET.ParseError:
        doc.add_paragraph("⚠️ Error parsing XML. Raw content below:")
        doc.add_paragraph(xml_string)
        pdf.multi_cell(0, 10, "⚠️ Error parsing XML\n" + xml_string, align="L")

    tmp_path = tempfile.gettempdir()
    base = f"{uuid.uuid4().hex}"
    docx_path = os.path.join(tmp_path, f"{base}.docx")
    pdf_path = os.path.join(tmp_path, f"{base}.pdf")
    doc.save(docx_path)
    pdf.output(pdf_path)
    return docx_path, pdf_path

# ─── FLASK APP ─────────────────────────────────────────────────────────────

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", xml="", raw="", error="", model_source="base")

@app.route("/generate", methods=["POST"])
def generate():
    excerpt = request.form.get("excerpt", "").strip()
    selected_source = request.form.get("model_select", "base")
    
    if not excerpt:
        return render_template("index.html", xml="", raw="", error="Please enter a non-empty excerpt.", model_source=selected_source)

    prompt = f"### Question:\n{excerpt}\n### Answer:"

    try:
        if selected_source == "local":
            if DEMO_FULL:
                base_output = "Quantum Computing is a method of performing calculations."
                xml = """<size><width>773</width><height>1000</height><depth>3</depth></size>
<object><name>paragraph</name><text>Quantum Computing is a method of performing calculations.</text>
<bndbox><xmin>706</xmin><ymin>30</ymin><xmax>710</xmax><ymax>41</ymax></bndbox></object>"""
                return render_template("index.html", xml=xml, raw=f"(demo mode: full shortcut)\n\nBase:\n{base_output}", error=None, model_source="local")

            if DEMO_BASE_ONLY:
                base_output = "Quantum Computing is a method of performing calculations."
            else:
                inputs_base = tokenizer(prompt, return_tensors="pt").to(base_model.device)
                out_ids_base = base_model.generate(**inputs_base, max_new_tokens=80, do_sample=True, top_p=0.9, temperature=0.8, pad_token_id=tokenizer.eos_token_id)
                full_output = tokenizer.decode(out_ids_base[0], skip_special_tokens=True).strip()
                base_output = full_output.split("### Answer:")[-1].strip() if "### Answer:" in full_output else full_output.strip()

            lora_prompt = f"Generate XML annotation for the following excerpt:\n{base_output}"
            inputs_lora = local_tokenizer(lora_prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(local_model.device)
            out_ids_lora = local_model.generate(**inputs_lora, max_new_tokens=150, min_length=20, num_beams=4, do_sample=False, pad_token_id=local_tokenizer.eos_token_id)
            xml = local_tokenizer.decode(out_ids_lora[0], skip_special_tokens=True).strip()

            return render_template("index.html", xml=xml, raw=f"(base output → LoRA refinement)\n\nBase:\n{base_output}", error=None, model_source="local")

        elif selected_source == "hf":
            if hf_inference is None:
                raise RuntimeError("HuggingFace model not available.")

            inputs_base = tokenizer(prompt, return_tensors="pt").to(base_model.device)
            out_ids_base = base_model.generate(**inputs_base, max_new_tokens=200, do_sample=True, top_p=0.9, temperature=0.8, pad_token_id=tokenizer.eos_token_id)
            full_output = tokenizer.decode(out_ids_base[0], skip_special_tokens=True).strip()
            base_output = full_output.split("### Answer:")[-1].strip() if "### Answer:" in full_output else full_output.strip()

            jolt_prompt = f"Refine or improve the following XML content:\n{base_output}"
            payload = {
                "inputs": jolt_prompt,
                "parameters": {
                    "max_new_tokens": 200,
                    "do_sample": True,
                    "top_p": 0.9,
                    "temperature": 0.0
                }
            }
            raw_resp = hf_inference(payload, raw_response=True)
            data = raw_resp.json()
            xml = data[0].get("generated_text", "") if isinstance(data, list) else data.get("generated_text", "")
            return render_template("index.html", xml=xml, raw=f"(local base → JOLT via HF)\n\nBase:\n{base_output}", error=None, model_source="hf")

        else:  # base
            inputs = tokenizer(prompt, return_tensors="pt").to(base_model.device)
            out_ids = base_model.generate(**inputs, max_new_tokens=500, do_sample=True, top_p=0.9, temperature=0.8, pad_token_id=tokenizer.eos_token_id)
            xml = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
            return render_template("index.html", xml=xml, raw="(generated from base model)", error=None, model_source="base")

    except Exception as err:
        return render_template("index.html", xml="", raw="", error=f"Generation failed: {err}", model_source=selected_source)

@app.route("/download", methods=["POST"])
def download():
    xml = request.form.get("xml_data", "")
    filetype = request.form.get("format", "docx")

    if not xml or not is_well_formed_xml(xml):
        return "Invalid or malformed XML", 400

    docx_path, pdf_path = xml_to_docx_and_pdf(xml)
    if filetype == "pdf":
        return send_file(pdf_path, as_attachment=True, download_name="annotation.pdf")
    else:
        return send_file(docx_path, as_attachment=True, download_name="annotation.docx")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
