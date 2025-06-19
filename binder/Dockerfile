FROM ollama/ollama:latest

USER root

# Pre-pull models during image build
RUN ollama serve & \
    sleep 10 && \
    ollama pull qwen3:0.6b && \
    pkill ollama


# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

# # Set working directory
# WORKDIR /app

# Update package list and install dependencies for Python build
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tzdata \
        wget \
        build-essential \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
        libxml2-dev \
        libxmlsec1-dev \
        libffi-dev \
        liblzma-dev \
    && ln -fs /usr/share/zoneinfo/UTC /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata \
    && rm -rf /var/lib/apt/lists/*

# Download and install Python 3.11.11
RUN wget https://www.python.org/ftp/python/3.11.11/Python-3.11.11.tgz && \
    tar xzf Python-3.11.11.tgz && \
    cd Python-3.11.11 && \
    ./configure --enable-optimizations --prefix=/usr/local && \
    make -j$(nproc) && \
    make altinstall && \
    cd .. && \
    rm -rf Python-3.11.11 Python-3.11.11.tgz

# Create symlinks for python3 and pip3
RUN ln -s /usr/local/bin/python3.11 /usr/local/bin/python3 && \
    ln -s /usr/local/bin/pip3.11 /usr/local/bin/pip3

# Upgrade pip
RUN /usr/local/bin/python3.11 -m pip install --upgrade pip

# Set working directory
WORKDIR /home/${NB_USER}/app

USER ${NB_USER}

COPY --chown=${NB_USER}:users . ./LLM_Reasoner
RUN pip install --no-cache-dir -r LLM_Reasoner/requirements.txt

COPY --chown=${NB_USER}:users start_services.sh ./LLM_Reasoner
RUN chmod +x ./LLM_Reasoner/start_services.sh

# Set working directory
WORKDIR /home/${NB_USER}/app/LLM_Reasoner

EXPOSE 5000

ENTRYPOINT []

CMD ["bash", "./start_services.sh"]
