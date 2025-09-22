## Setup

- Create a `.env` with your keys:

```env
GEMINI_API_KEY=your_gemini_api_key
# optional tracing
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_PUBLIC_KEY=pk_...
LANGFUSE_SECRET_KEY=sk_...
```

- Install and run (using uv):

```bash
uv sync
uv run main.py
```

See the [DSPy API docs](https://dspy.ai) for details.
