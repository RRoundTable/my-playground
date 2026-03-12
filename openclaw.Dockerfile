FROM alpine/openclaw:2026.3.2

USER root
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
      libnspr4 libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 \
      libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 \
      libxrandr2 libgbm1 libasound2 libpango-1.0-0 libcairo2 && \
    rm -rf /var/lib/apt/lists/*
