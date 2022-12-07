FROM ghcr.io/bluesky/tiled:v0.1.0a74 as base

FROM base as builder

WORKDIR /build
COPY . .
RUN pip install .

FROM base as app
COPY --from=builder $VIRTUAL_ENV $VIRTUAL_ENV
COPY deploy/spin/docker/tiled/gunicorn_config.py .
ENV GUNICORN_CONF="/deploy/gunicorn_config.py"
