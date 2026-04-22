# Self Correcting Neural Systems — Backend API

A FastAPI backend implementing a modular three-component inference-time self correction pipeline based on the research abstract by Md Adnan Baqi (M.Tech CSE, 25CPMEA125).

## Architecture

```
Prompt → [Generator] → Initial Output
                              ↓
                        [Diagnoser]  ← calibrated uncertainty estimation
                         ↙       ↘
              is_correct           is_incorrect
                  ↓                     ↓
            Return output         [Explainer] → diagnostic rationale
                                        ↓
                                  [Corrector] → refined output
```

### Components

| Component | Role | Model Temperature |
|---|---|---|
| **Generator** | Single-pass baseline generation | 0.5 |
| **Diagnoser** | Calibrated correctness prediction (JSON output) | 0.1 |
| **Explainer** | Error rationale without revealing answer | 0.2 |
| **Corrector** | Conditioned refinement from rationale | 0.3 |

## Prerequisites

1. **Ollama** installed and running: https://ollama.com
2. A model pulled locally:
   ```bash
   ollama pull llama3
   # or
   ollama pull mistral
   # or
   ollama pull phi3   # lighter, faster on CPU
   ```

## Setup

```bash
# 1. Clone and enter the project
cd Self Correcting-neural-system

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env to set OLLAMA_DEFAULT_MODEL to your pulled model name
```

## Running the Server

```bash
# Development (with auto-reload)
uvicorn app.main:app --reload

# Production
python run.py
```

Server starts at: **http://localhost:8000**

Interactive docs at: **http://localhost:8000/docs**

## API Endpoints

### `POST /api/v1/infer`
Run the full Self Correcting pipeline.

**Request:**
```json
{
  "prompt": "A bat and a ball cost $1.10. The bat costs $1.00 more than the ball. How much does the ball cost?",
  "domain": "math",
  "model_name": "llama3"
}
```

**Domains:** `math`, `code`, `commonsense`, `qa`, `general`

**Response:**
```json
{
  "original_prompt": "...",
  "initial_output": "The ball costs $0.10",
  "diagnosis": {
    "confidence_score": 0.35,
    "is_correct": false,
    "reasoning": "The answer ignores the constraint..."
  },
  "was_refined": true,
  "rationale": "The error is that if ball = $0.10 then bat = $1.10...",
  "final_output": "The ball costs $0.05. The bat costs $1.05...",
  "model_used": "llama3",
  "pipeline_stages_executed": 4
}
```

### `GET /api/v1/health`
Returns Ollama connectivity status and available models.

### `GET /api/v1/models`
Returns all locally available Ollama model names.

## Running Tests

```bash
pytest tests/ -v
```

Tests mock Ollama calls and run without a live LLM.

## File Structure

```
Self Correcting-neural-system/
├── app/
│   ├── api/
│   │   └── routes.py          # FastAPI route handlers
│   ├── core/
│   │   ├── diagnoser.py       # Calibrated uncertainty estimation
│   │   ├── explainer.py       # Diagnostic rationale generation
│   │   ├── corrector.py       # Conditioned output refinement
│   │   └── pipeline.py        # Pipeline orchestrator
│   ├── models/
│   │   └── schemas.py         # Pydantic data models
│   ├── services/
│   │   └── llm_client.py      # Ollama API wrapper
│   ├── config.py              # Settings management
│   └── main.py                # FastAPI app factory
├── tests/
│   └── test_pipeline.py       # Integration tests
├── .env.example
├── requirements.txt
├── run.py                     # Production entrypoint
└── README.md
```

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_DEFAULT_MODEL` | `llama3` | Model to use when none specified |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `DIAGNOSER_CONFIDENCE_THRESHOLD` | `0.85` | Below this → trigger correction |
| `MAX_CORRECTION_RETRIES` | `2` | Reserved for future multi-pass correction |

## Swapping to Custom Trained Models

When your custom models are ready, only `app/services/llm_client.py` needs updating. Replace the `ollama.chat()` call with your model's inference endpoint — all pipeline logic stays identical.
