FROM python:3.11-slim

WORKDIR /app

# install system dependencies torch/PyG need
RUN apt-get update && apt-get install -y \
    git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# install Python dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.1.0 && \
    pip install --no-cache-dir torch-geometric numpy

# copy the application
COPY . .

# make pipeline executable
RUN chmod +x pipeline.sh

# default to interactive shell when run with -it
CMD ["/bin/bash"]