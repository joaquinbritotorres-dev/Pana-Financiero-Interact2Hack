# Pana Financiero

Asistente conversacional de negocio para micro-comerciantes ecuatorianos.
Permite a un comerciante sin formación financiera preguntar en lenguaje
natural sobre su propio negocio ("¿cuánto vendí esta semana?", "¿qué
clientes no han vuelto?") y recibir respuestas claras y accionables en
segundos.

Desarrollado durante el Interact2Hack 2026 (USFQ) como solución al
Reto 2 de Deuna. El asistente responde sobre un dataset de transacciones
y nunca inventa datos: si la información no está disponible, lo indica.

**Stack:** Python · FastAPI · Pandas · OpenAI API

## Cómo funciona
- El backend expone una API que recibe preguntas en lenguaje natural.
- Pandas procesa los datos transaccionales del negocio.
- La OpenAI API traduce los datos en respuestas comprensibles.

# Pana Financiero — Backend

FastAPI + Pandas + OpenAI para el asistente conversacional financiero.

## Setup (macOS)

1. Navegar al backend:
   cd backend

2. Crear entorno virtual:
   python3 -m venv venv
   source venv/bin/activate

3. Instalar dependencias:
   pip install -r requirements.txt

4. Configurar API key:
   cp .env.example .env
   # Editar .env y pegar tu OPENAI_API_KEY

5. Correr el servidor:
   uvicorn main:app --reload

El servidor queda en http://localhost:8000
Documentación interactiva: http://localhost:8000/docs

## Endpoints
- GET  /api/negocios       Lista de 4 negocios
- POST /api/ask             Body: {pregunta, id_negocio} → {respuesta}

## Probar con curl
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"pregunta": "¿Cuánto vendí hoy?", "
