ARG CUDA_VERSION=12.4.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt update -y \
    && apt install software-properties-common -y \
    && add-apt-repository ppa:deadsnakes/ppa -y && apt update \
    && apt install python3.10 python3.10-dev -y \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 2 \
    && update-alternatives --set python3 /usr/bin/python3.10 && apt install python3.10-distutils -y \
    && apt install curl git sudo -y \
    && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3 get-pip.py \
    && python3 --version \
    && python3 -m pip --version \
    && rm -rf /var/lib/apt/lists/* \
    && apt clean

RUN ldconfig /usr/local/cuda-12.1/compat/

WORKDIR /sgl-workspace

RUN python3 -m pip install --upgrade pip setuptools wheel html5lib six \
    && git clone --depth=1 https://github.com/sgl-project/sglang.git \
    && cd sglang \
    && if [ "$BUILD_TYPE" = "srt" ]; then \
         python3 -m pip --no-cache-dir install -e "python[srt]"; \
       else \
         python3 -m pip --no-cache-dir install -e "python[all]"; \
       fi

ARG CUDA_VERSION
RUN if [ "$CUDA_VERSION" = "12.1.1" ]; then \
        export CUDA_IDENTIFIER=cu121 && \
        python3 -m pip --no-cache-dir install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/; \
    elif [ "$CUDA_VERSION" = "12.4.1" ]; then \
        export CUDA_IDENTIFIER=cu124 && \
        python3 -m pip --no-cache-dir install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.4/; \
    elif [ "$CUDA_VERSION" = "11.8.0" ]; then \
        export CUDA_IDENTIFIER=cu118 && \
        python3 -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu118 && \
        python3 -m pip --no-cache-dir install flashinfer -i https://flashinfer.ai/whl/cu118/torch2.4/; \
    else \
        echo "Unsupported CUDA version: $CUDA_VERSION" && exit 1; \
    fi

# Install Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN python3 -m pip install --upgrade -r /requirements.txt

RUN python3 -m pip cache purge

# Setup for Option 2: Building the Image with the Model included
ARG MODEL_NAME=""
ARG TOKENIZER_NAME=""
ARG BASE_PATH="/runpod-volume"
ARG QUANTIZATION=""
ARG MODEL_REVISION=""
ARG TOKENIZER_REVISION=""

ENV MODEL_NAME=$MODEL_NAME \
    MODEL_REVISION=$MODEL_REVISION \
    TOKENIZER_NAME=$TOKENIZER_NAME \
    TOKENIZER_REVISION=$TOKENIZER_REVISION \
    BASE_PATH=$BASE_PATH \
    QUANTIZATION=$QUANTIZATION \
    HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=1 

ENV PYTHONPATH="/:/vllm-workspace"


COPY src /src
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    if [ -f /run/secrets/HF_TOKEN ]; then \
        export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
    fi && \
    if [ -n "$MODEL_NAME" ]; then \
        python3 /src/download_model.py; \
    fi

# Start the handler
CMD ["python3", "/src/handler.py"]