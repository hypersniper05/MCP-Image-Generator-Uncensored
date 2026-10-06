# syntax=docker/dockerfile:1.7
# -----------------------------------------------------------------------------
# Image Gen MCP
#   target "gpu": NVIDIA CUDA build (RTX 20xx and newer, incl. RTX 50xx Blackwell)
#   target "cpu": CPU-only build (x86-64 and arm64)
# The inference engine is stable-diffusion.cpp (MIT), built from a pinned tag.
# -----------------------------------------------------------------------------
ARG CUDA_VERSION=12.8.2
ARG UBUNTU_VERSION=24.04
ARG SDCPP_REF=master-929-3f8527a
ARG SDCPP_REPO=https://github.com/leejet/stable-diffusion.cpp

# ---------------------------------------------------------------- sources
FROM ubuntu:${UBUNTU_VERSION} AS sdcpp-src
ARG SDCPP_REF
ARG SDCPP_REPO
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch "${SDCPP_REF}" --recursive --shallow-submodules "${SDCPP_REPO}" /src
# Local engine patches (docker/patches/*.patch). Their names go to /opt/sdcpp/MCP_PATCHES so the service knows
# which request keys this build reads (sdcpp-circular-json: per-request circular_x / circular_y).
COPY docker/patches/ /patches/
RUN for p in /patches/*.patch; do git -C /src apply --verbose "$p" || exit 1; done \
 && (cd /patches && ls -1 *.patch | sed 's/[.]patch$//') > /src/MCP_PATCHES

# ---------------------------------------------------------------- build (CUDA)
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS build-gpu
# 75=RTX 20xx/T4, 80=A100, 86=RTX 30xx, 89=RTX 40xx, 90=H100, 120=RTX 50xx
ARG CUDA_ARCHS="75-virtual;80-virtual;86-real;89-real;90-virtual;120a-real"
ARG BUILD_JOBS=8
RUN apt-get update && apt-get install -y --no-install-recommends cmake build-essential git \
 && rm -rf /var/lib/apt/lists/*
COPY --from=sdcpp-src /src /src
WORKDIR /src
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DSD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" -DGGML_CUDA_NCCL=OFF \
      -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
      -DSD_BUILD_SHARED_LIBS=ON -DSD_BUILD_SHARED_GGML_LIB=ON \
      -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON "-DCMAKE_INSTALL_RPATH=\$ORIGIN" \
      -DSD_SERVER_BUILD_FRONTEND=OFF \
 && cmake --build build --config Release -j "${BUILD_JOBS}" \
 && mkdir -p /opt/sdcpp && cp -a build/bin/. /opt/sdcpp/ && cp /src/MCP_PATCHES /opt/sdcpp/ \
 && rm -f /opt/sdcpp/*.a

# ---------------------------------------------------------------- build (CPU)
FROM ubuntu:${UBUNTU_VERSION} AS build-cpu
ARG BUILD_JOBS=8
RUN apt-get update && apt-get install -y --no-install-recommends cmake build-essential git ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY --from=sdcpp-src /src /src
WORKDIR /src
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
      -DSD_BUILD_SHARED_LIBS=ON -DSD_BUILD_SHARED_GGML_LIB=ON \
      -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON "-DCMAKE_INSTALL_RPATH=\$ORIGIN" \
      -DSD_SERVER_BUILD_FRONTEND=OFF \
 && cmake --build build --config Release -j "${BUILD_JOBS}" \
 && mkdir -p /opt/sdcpp && cp -a build/bin/. /opt/sdcpp/ && cp /src/MCP_PATCHES /opt/sdcpp/ \
 && rm -f /opt/sdcpp/*.a

# ---------------------------------------------------------------- python app
FROM ubuntu:${UBUNTU_VERSION} AS app-base
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates  && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --no-compile .  && /opt/venv/bin/python -m compileall -q /opt/venv/lib

# ---------------------------------------------------------------- runtime (GPU)
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS gpu
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    IMAGEGEN_CONFIG=/app/config.yaml \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_CACHE_PATH=/models/.cache/cuda \
    CUDA_CACHE_MAXSIZE=2147483648
RUN apt-get update && apt-get install -y --no-install-recommends python3 ca-certificates libgomp1 \
 && rm -rf /var/lib/apt/lists/*
COPY --from=app-base /opt/venv /opt/venv
COPY --from=build-gpu /opt/sdcpp /opt/sdcpp
WORKDIR /app
EXPOSE 5005
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python3 -c "import os,sys,urllib.request; u='http://127.0.0.1:%s/health' % os.environ.get('IMAGEGEN_PORT', '5005'); sys.exit(0 if urllib.request.urlopen(u, timeout=4).status == 200 else 1)"
ENTRYPOINT ["python3", "-m", "imagegen_mcp"]

# ---------------------------------------------------------------- runtime (CPU)
FROM ubuntu:${UBUNTU_VERSION} AS cpu
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    IMAGEGEN_CONFIG=/app/config.yaml \
    IMAGEGEN_DEVICE=cpu \
    NVIDIA_VISIBLE_DEVICES=void
RUN apt-get update && apt-get install -y --no-install-recommends python3 ca-certificates libgomp1 \
 && rm -rf /var/lib/apt/lists/*
COPY --from=app-base /opt/venv /opt/venv
COPY --from=build-cpu /opt/sdcpp /opt/sdcpp
WORKDIR /app
EXPOSE 5005
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python3 -c "import os,sys,urllib.request; u='http://127.0.0.1:%s/health' % os.environ.get('IMAGEGEN_PORT', '5005'); sys.exit(0 if urllib.request.urlopen(u, timeout=4).status == 200 else 1)"
ENTRYPOINT ["python3", "-m", "imagegen_mcp"]
