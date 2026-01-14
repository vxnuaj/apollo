"""
Convenience script to run the Apollo inference server.

Usage:
    python run_server.py
    
Or with custom host/port:
    python run_server.py --host 0.0.0.0 --port 8080
"""

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Run Apollo Inference Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    
    args = parser.parse_args()
    
    print(f"""
Apollo Inference Server
Server:  http://{args.host}:{args.port}
Docs:    http://{args.host}:{args.port}/docs
Health:  http://{args.host}:{args.port}/v1/health
    """)
    
    uvicorn.run(
        "server.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level="info",
    )


if __name__ == "__main__":
    main()
