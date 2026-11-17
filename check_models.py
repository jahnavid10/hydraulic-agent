import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

try:
    opts = types.HttpOptions(timeout=20000, retry_options=types.HttpRetryOptions(attempts=1))
except Exception:
    opts = types.HttpOptions(timeout=20000)
client = genai.Client(http_options=opts)

CANDIDATES = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3.1-flash-lite",
              "gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.8-flash",
              "gemini-flash-lite-latest", "gemini-flash-latest"]

for name in CANDIDATES:
    t = time.time()
    try:
        r = client.models.generate_content(model=name, contents="Reply with the single word: ok")
        print(f"{name}: OK ({time.time()-t:.1f}s) -> {r.text.strip()}", flush=True)
    except Exception as e:
        print(f"{name}: FAILED ({time.time()-t:.1f}s) -> {str(e)[:110]}", flush=True)