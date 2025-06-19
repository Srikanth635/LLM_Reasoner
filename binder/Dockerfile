FROM intel4coro/base-notebook:20.04-noetic-vnc

USER root

#FROM ollama/ollama:latest
RUN curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz && \
    tar -C /usr -xzf ollama-linux-amd64.tgz && \
    rm ollama-linux-amd64.tgz

# Create ollama user and set permissions
RUN useradd -r -s /bin/false -m -d /usr/share/ollama ollama

# Create necessary directories
RUN mkdir -p /usr/share/ollama && \
    chown ollama:ollama /usr/share/ollama

# Switch to ollama user for model download (optional - see note below)

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
    TZ=UTC \
    OLLAMA_HOST=0.0.0.0:11434

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

USER root

RUN conda create -n agentic_ai311 python=3.11 -y && \
    echo "source /opt/conda/etc/profile.d/conda.sh" >> /home/${NB_USER}/.bashrc && \
    echo "conda activate agentic_ai311" >> /home/${NB_USER}/.bashrc && \
    chown ${NB_USER}:users /home/${NB_USER}/.bashrc

# Set working directory
WORKDIR /home/${NB_USER}

USER ${NB_USER}

COPY --chown=${NB_USER}:users . ./LLM_Reasoner

SHELL ["conda", "run", "-n", "agentic_ai311", "/bin/bash", "-c"]
RUN pip install --no-cache-dir -r LLM_Reasoner/requirements.txt

COPY --chown=${NB_USER}:users start_services.sh ./LLM_Reasoner
RUN chmod +x ./LLM_Reasoner/start_services.sh

# Set working directory
WORKDIR /home/${NB_USER}/LLM_Reasoner

EXPOSE 5000

COPY --chown=${NB_USER}:users binder/webapps.json ${ROS_WS}/src/rvizweb/webapps/app.json

ENTRYPOINT []

CMD ["bash", "./start_services.sh"]
