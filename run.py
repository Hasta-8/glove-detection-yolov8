"""Start the API: uv run python run.py  (http://127.0.0.1:8004, docs at /docs)."""

import argparse

import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Glove detection API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8004)
    args = parser.parse_args()
    uvicorn.run("app.main:create_app", factory=True, host=args.host, port=args.port)
