FROM postgres:14

# Install build dependencies
RUN apt-get update && apt-get install -y \
    postgresql-server-dev-14 \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone and install pgvector
RUN git clone --branch v0.8.1 https://github.com/pgvector/pgvector.git /tmp/pgvector \
    && cd /tmp/pgvector \
    && make \
    && make install \
    && rm -rf /tmp/pgvector
