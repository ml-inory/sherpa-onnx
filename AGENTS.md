# AGENTS.md

## 项目目标

- 将sherpa-onnx的模型都通过$magnetar转换到axmodel，并实现axera后端，直到上板测试sherpa-onnx的axera provider的demo通过，效果需要与sherpa-onnx原生一致.

- 上板验证可使用 root@192.168.31.201 密码123456  如果空间不足可以通过mount好的目录/root/rzyang以访问本机home目录

- 如果$magnetar转换失败了，及时止损，跳到下一个模型即可


## 强制约束

- 风格要求与sherpa-onnx对齐
- 经过$magnetar转换好的模型需要分类放到~/HF下，并写个README说明对应的sherpa模型、作用、与原模型对分的效果、速度,README需要符合HuggingFace的格式


## 爱芯开发知识

- pulsar2镜像: https://hf-mirror.com/AXERA-TECH/Pulsar2
- pulsar2文档: https://pulsar2-docs.readthedocs.io/zh-cn/latest/
- 爱芯HF模型: https://hf-mirror.com/AXERA-TECH
- C++ BSP / 交叉编译器:
  - AX650 BSP SDK (AX runtime头文件和库):
    下载: https://hf-mirror.com/AXERA-TECH/AX650-Community-Hub/resolve/main/sdk/edge-computing-AX650_SDK_V3.10.2/02.%20SDK/AX650_SDK_V3.10.2/AX650_SDK_V3.10.2_20260513151335.tgz
    HF页面: https://hf-mirror.com/AXERA-TECH/AX650-Community-Hub/tree/main/sdk/edge-computing-AX650_SDK_V3.10.2/02.%20SDK/AX650_SDK_V3.10.2
  - AX650 交叉编译器(同AX620E)
  - AX620E (待更新，暂用 Arm GNU 裸工具链):
    https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz
- 爱芯开源Github repos: https://github.com/AXERA-TECH
- 本机的docker镜像可能已安装pulsar2，应优先使用最新版本
- 如果需要上板运行，可用remote-infer SKILL完成
- LLM编译: https://github.com/AXERA-TECH/ax-llm
