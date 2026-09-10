# PyTorch 2.12 CPU 容器环境

## ABI 问题与边界

原始实验宿主为 CentOS 7、x86_64、glibc 2.17。官方
`torch-2.12.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl` 要求的 Linux
用户态 ABI 高于该宿主，因此不能由宿主 Python 直接加载。

本环境使用 Debian 12 Bookworm 用户态和 Python 3.12。基础镜像通过 AWS
Public ECR 提供的 Docker Official Images 镜像获取，因为验证设备所在网络无法
连接 Docker Hub；镜像已按内容 digest 固定。PyTorch 仅从官方 CPU wheel 索引
安装，镜像中不安装 CUDA。

明确未执行：

- 不升级或覆盖宿主 glibc；
- 不通过 `LD_LIBRARY_PATH` 注入自编译 libc；
- 不复制仓库源码、`.git`、SSH key、`.env` 或其他凭据到镜像；
- 不下载项目运行时数据或模型；
- 不重跑 F1、F2A、F2B 或 F2C 的正式 development/locked audit。

## 固定输入

```text
base_image = public.ecr.aws/docker/library/python:3.12-slim-bookworm
base_image_digest = sha256:782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254
python = 3.12
torch = 2.12.0+cpu
cpu_threads_default = 1
```

`requirements.lock` 包含 Python 直接依赖、传递依赖和 wheel/sdist SHA-256。
Docker build context 仅为本目录，`.dockerignore` 又将其收窄到 lock 文件，因而
不会把仓库或凭据发送给构建器。

## 构建

在仓库根目录执行：

```bash
docker build \
  --file environments/torch212_cpu/Dockerfile \
  --tag lunwen-torch212-cpu:dev \
  environments/torch212_cpu
```

## 开发验证

```bash
docker run --rm -it \
  --cpus=4 \
  --mount type=bind,src="$PWD",dst=/workspace/lunwen,readonly \
  --workdir /workspace/lunwen \
  lunwen-torch212-cpu:dev \
  bash
```

容器内运行：

```bash
./environments/torch212_cpu/run_tests.sh
```

测试脚本显式禁用 pytest 和 Ruff 缓存，因此仓库可保持只读挂载。

## 无网络运行

```bash
mkdir -p .runs
docker run --rm \
  --network none \
  --cpus=4 \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=1g \
  --mount type=bind,src="$PWD",dst=/workspace/lunwen,readonly \
  --mount type=bind,src="$PWD/.runs",dst=/workspace/runtime \
  --workdir /workspace/lunwen \
  lunwen-torch212-cpu:<frozen-tag> \
  python environments/torch212_cpu/verify_environment.py
```

如需正式运行 F2C，应将预注册命令显式传给 `run_f2c.sh`；脚本自身不会猜测或
自动运行任何 one-shot audit。正式输出只写入单独挂载的 `/workspace/runtime`。

## Apptainer fallback

```bash
apptainer build lunwen-torch212-cpu.sif \
  environments/torch212_cpu/apptainer.def

apptainer exec \
  --bind "$PWD:/workspace/lunwen" \
  lunwen-torch212-cpu.sif \
  python environments/torch212_cpu/verify_environment.py
```

Apptainer 默认以调用者身份运行，不需要把宿主账号写入镜像。
