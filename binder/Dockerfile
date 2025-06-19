FROM ollama/ollama:latest

ENV NB_USER=jovyan

RUN apt update && \
    apt install -y \
        gdb g++ clang cmake make doxygen \
        libeigen3-dev \
        libspdlog-dev \
        libraptor2-dev \
        librdf0 \
        librdf0-dev \
        libmongoc-1.0-0 \
        libmongoc-dev \
        libgtest-dev \
        libfmt-dev \
        software-properties-common \
        git \
        wget \
        curl \
        gnupg \
        sudo \
        python3 \
        python3-dev \
        python3-pip \
        python3-setuptools

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

USER root

RUN conda create -n agentic_ai311 python=3.11 -y && \
    echo "source /opt/conda/etc/profile.d/conda.sh" >> /home/${NB_USER}/.bashrc && \
    echo "conda activate agentic_ai311" >> /home/${NB_USER}/.bashrc && \
    chown ${NB_USER}:users /home/${NB_USER}/.bashrc

# Set working directory
WORKDIR /home/${NB_USER}

USER ${NB_USER}

COPY --chown=${NB_USER}:users . ./LLM_Reasoner
RUN pip install --no-cache-dir -r LLM_Reasoner/requirements.txt

COPY --chown=${NB_USER}:users start_services.sh ./LLM_Reasoner
RUN chmod +x ./LLM_Reasoner/start_services.sh

# Set working directory
WORKDIR /home/${NB_USER}/LLM_Reasoner

EXPOSE 5000

ENTRYPOINT []

CMD ["bash", "./start_services.sh"]
