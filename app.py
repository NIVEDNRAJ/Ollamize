from flask import Flask, request, render_template, jsonify
import requests
import io
from PyPDF2 import PdfReader
import math
import json


app = Flask(__name__)

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5"

# --- Helpers ------------------------------------------------
def extract_text_from_pdf(file_stream):
    reader = PdfReader(file_stream)
    text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    return "\n".join(text)

def chunk_text(text, max_chars=3000):
    """Split text into chunks of up to ~max_chars characters on sentence boundaries when possible."""
    text = text.replace("\r\n", " ").replace("\n", " ")
    if len(text) <= max_chars:
        return [text.strip()]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        # try find last period/semicolon before end to split nicely
        split_at = text.rfind('.', start, end)
        if split_at == -1:
            split_at = text.rfind(';', start, end)
        if split_at == -1:
            split_at = end
        chunk = text[start:split_at+1].strip()
        chunks.append(chunk)
        start = split_at + 1
    return chunks

def call_ollama(prompt, model=MODEL_NAME):
    payload = {
        "model": model,
        "prompt": prompt
    }
    with requests.post(OLLAMA_API_URL, json=payload, stream=True) as res:
        res.raise_for_status()
        full_output = ""
        for line in res.iter_lines():
            if not line:
                continue
            try:
                data = line.decode("utf-8")
                j = json.loads(data)
                # Each chunk has "response"
                if "response" in j:
                    full_output += j["response"]
            except Exception as e:
                print(f"Error decoding line: {e}")
        return full_output.strip()

def summarize_long_text_iterative(text):
    """
    1) Split into chunks
    2) Summarize each chunk
    3) Combine chunk summaries and summarize again (final concise summary)
    """
    chunks = chunk_text(text, max_chars=3000)
    chunk_summaries = []
    for i, ch in enumerate(chunks, start=1):
        prompt = (
            f"Summarize the following text concisely (one paragraph) keeping key points and facts. "
            f"Chunk {i} of {len(chunks)}:\n\n{ch}\n\nSummary:"
        )
        try:
            summary = call_ollama(prompt)
        except Exception as e:
            summary = f"[ERROR: {e}]"
        chunk_summaries.append(summary.strip())

    # if only one chunk, return its summary
    if len(chunk_summaries) == 1:
        return chunk_summaries[0]

    # combine chunk summaries & make final summary
    combined = "\n".join(f"Chunk {i+1}: {s}" for i, s in enumerate(chunk_summaries))
    final_prompt = (
        "You are an expert summarizer. Given the following chunk summaries, produce a single concise "
        "and coherent summary in 3-6 sentences containing the most important points.\n\n"
        f"{combined}\n\nFinal summary:"
    )
    final_summary = call_ollama(final_prompt)
    return final_summary.strip()

# --- Routes -----------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    # Accept text field OR file upload (pdf)
    raw_text = request.form.get('text', '').strip()
    uploaded_file = request.files.get('file')

    if uploaded_file and uploaded_file.filename.lower().endswith('.pdf'):
        # read pdf bytes
        file_stream = io.BytesIO(uploaded_file.read())
        text = extract_text_from_pdf(file_stream)
        if not text:
            return jsonify({'error': 'Could not extract text from PDF'}), 400
    elif raw_text:
        text = raw_text
    else:
        return jsonify({'error': 'No text or PDF provided'}), 400

    # Do iterative chunk summarization
    try:
        summary = summarize_long_text_iterative(text)
    except requests.exceptions.RequestException as e:
        return jsonify({'error': 'Failed to contact Ollama API', 'details': str(e)}), 500
    except Exception as e:
        return jsonify({'error': 'Error during summarization', 'details': str(e)}), 500

    return jsonify({'summary': summary})

if __name__ == '__main__':
    app.run(debug=True)
