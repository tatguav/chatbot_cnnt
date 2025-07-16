# crew.py
from crewai import Agent, CrewBase, agent
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer
from crewai_tools import Tool
import fitz  # PyMuPDF
import os

# RUTA RELATIVA al PDF
PDF_PATH = os.path.join("documents", "Codigo-nacional-de-transito.pdf")

# Función para extraer artículos del PDF
def extraer_articulos(pdf_path):
    doc = fitz.open(pdf_path)
    texto = ""
    for page in doc:
        texto += page.get_text()
    partes = texto.split("ARTÍCULO")
    articulos = [("ARTÍCULO" + p).strip() for p in partes if p.strip()]
    return articulos

# Cargar artículos reales desde el PDF
articulos = extraer_articulos(PDF_PATH)

# Cargar modelos
embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
generator = pipeline("text2text-generation", model=llm_model, tokenizer=tokenizer)

# Crear herramienta personalizada
class CustomRagTool(Tool):
    name = "RAG Tool PDF Transito"
    description = "Busca en el Código Nacional de Tránsito"

    def __init__(self, pdf_texts):
        super().__init__()
        self.contexts = pdf_texts
        self.index = self._build_index()

    def _build_index(self):
        vectors = embedder.encode(self.contexts)
        import faiss
        idx = faiss.IndexFlatL2(vectors.shape[1])
        idx.add(vectors)
        return idx

    def run(self, input):
        vector = embedder.encode([input])
        _, indices = self.index.search(vector, k=3)
        result = "\n".join([self.contexts[i] for i in indices[0]])
        prompt = f"Resumen legal:\n{result}\n\nPregunta:\n{input}\nRespuesta:"
        return generator(prompt, max_new_tokens=200)[0]["generated_text"]

# 🔹 Clase de CrewAI
class TransitoCrew(CrewBase):
    agents_config = "config/agents.yaml"

    @agent
    def responder(self):
        return Agent(
            config=self.agents_config["responder"],
            tools=[CustomRagTool(articulos)]
        )
