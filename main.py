from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from pana.loader import load_all, get_negocios_list
from pana.assistant import responder


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_all()
    yield


app = FastAPI(title="Pana Financiero API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:8080"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    pregunta: str
    id_negocio: str


class AskResponse(BaseModel):
    respuesta: str
    mensaje_carga: str = ""


@app.get("/")
def health():
    return {"status": "ok", "service": "Pana Financiero API"}


@app.get("/api/negocios")
def negocios():
    return get_negocios_list()


@app.post("/api/ask", response_model=AskResponse)
async def ask(body: AskRequest):
    if not body.pregunta.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía")
    if not body.id_negocio.strip():
        raise HTTPException(status_code=400, detail="id_negocio requerido")
    try:
        from pana.loading_messages import get_mensaje_carga
        respuesta = await responder(body.pregunta, body.id_negocio)
        return AskResponse(respuesta=respuesta, mensaje_carga=get_mensaje_carga())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ask/sql", response_model=AskResponse)
async def ask_sql(body: AskRequest):
    """
    Endpoint Text-to-SQL: precisión exacta para fechas y datos puntuales.
    El LLM genera SQL, SQLite ejecuta, el LLM solo redacta.
    """
    if not body.pregunta.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía")
    if not body.id_negocio.strip():
        raise HTTPException(status_code=400, detail="id_negocio requerido")
    try:
        from pana.assistant import sql_responder
        from pana.loading_messages import get_mensaje_carga
        respuesta = await sql_responder(body.pregunta, body.id_negocio)
        return AskResponse(respuesta=respuesta, mensaje_carga=get_mensaje_carga())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class GraficoRequest(BaseModel):
    respuesta: str
    mensaje_carga: str = ""

@app.post("/api/grafico")
async def grafico(body: GraficoRequest):
    if not body.respuesta.strip():
        return {}
    try:
        import json as json_lib
        from openai import AsyncOpenAI
        import os
        client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

        prompt = f"""Analiza este texto financiero y extrae SOLO los pares de valores numéricos comparativos que encuentres.
Devuelve SOLO un JSON válido con las etiquetas como claves y los números como valores float.
Sin explicaciones, sin markdown, solo el JSON.

Ejemplos de lo que debes devolver:
- Si el texto compara esta semana vs semana pasada: {{"esta semana": 153.90, "semana pasada": 291.74}}
- Si compara hoy vs ayer: {{"hoy": 58.54, "ayer": 72.30}}
- Si compara este mes vs mes anterior: {{"este mes": 1240.50, "mes anterior": 980.00}}
- Si no hay comparación clara (solo un valor): {{"total": 153.90}}
- Si no hay ningún número: {{}}

Texto a analizar:
{body.respuesta}"""

        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100,
        )

        raw = response.choices[0].message.content or "{}"
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        data = json_lib.loads(raw)
        if not isinstance(data, dict):
            return {}
        result = {k: float(v) for k, v in list(data.items())[:5]
                  if isinstance(v, (int, float))}
        return result
    except Exception:
        return {}
