FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    mcp \
    httpx \
    anthropic \
    chromadb \
    sentence-transformers \
    python-dotenv \
    cryptography \
    eth-account \
    web3 \
    fastapi \
    uvicorn \
    x402

COPY . .

ENV PHOENIXD_URL=http://host.docker.internal:9740
ENV PHOENIXD_PASSWORD=""
ENV ANTHROPIC_API_KEY=""

EXPOSE 8005

CMD ["python3", "server.py"]
