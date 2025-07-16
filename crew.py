
# crew.py
# Eliminamos dependencia de crewai_tools
class Tool:
    def __init__(self):
        pass

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import fitz  # PyMuPDF
import os
import faiss

# ------------------------------
# Función para extraer artículos del PDF
# ------------------------------
def extraer_articulos(pdf_path):
    doc = fitz.open(pdf_path)
    texto = ""
    for page in doc:
        texto += page.get_text()
    partes = texto.split("ARTÍCULO")
    articulos = [("ARTÍCULO" + p).strip() for p in partes if p.strip()]
    return articulos

# Modelos de embeddings y generación

embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
generator = pipeline("text2text-generation", model=llm_model, tokenizer=tokenizer)

# Herramienta RAG personalizada

class CustomRagTool(Tool):
    name = "RAG Tool PDF Transito"
    description = "Busca contexto relevante en el Código de Tránsito"

    def __init__(self, pdf_texts):
        super().__init__()
        self.contexts = pdf_texts
        self.index = self._build_index()

    def _build_index(self):
        vectors = embedder.encode(self.contexts)
        idx = faiss.IndexFlatL2(vectors.shape[1])
        idx.add(vectors)
        return idx

    def run(self, input):
        vector = embedder.encode([input])
        _, indices = self.index.search(vector, k=3)
        result = "\n".join([self.contexts[i] for i in indices[0]])
        prompt = f"Resumen legal:\n{result}\n\nPregunta:\n{input}\nRespuesta:"
        return generator(prompt, max_new_tokens=200)[0]["generated_text"]
