# Container

## Image

```text
ghcr.io/r0b0tlab/r0b0bench:v1.0.0-rc1
```

Client-only. Point at any OpenAI-compatible server.

## Build (arm64 GB10)

```bash
docker build -t ghcr.io/r0b0tlab/r0b0bench:v1.0.0-rc1 .
docker push ghcr.io/r0b0tlab/r0b0bench:v1.0.0-rc1
```

## Run

```bash
docker run --rm --network host \
  -v /tmp/r0b0bench-out:/out \
  -v /path/to/tokenizer:/tokenizer:ro \
  ghcr.io/r0b0tlab/r0b0bench:v1.0.0-rc1 \
  run --profile core-subset \
    --base-url http://127.0.0.1:18082/v1 \
    --model xyz-aquila-nvfp4 \
    --tokenizer /tokenizer \
    --output /out
```

## Compose

See `docker-compose.yml` for `bench` + optional `aquila-vllm` demo service.
