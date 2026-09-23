# NEAD

论文 *Abductive Artificial Intelligence Reveals the Laws Underlying Transportation Resilience* 的官方实现。

NEAD（Network Emergence Abductive Discovery）首先用图神经网络代理模型学习交通韧性动力学，再通过基于 GNNExplainer 的结构解耦器识别关键网络结构，最后将低维表示提炼为显式符号公式。

[English](README.md)

## 环境配置

推荐使用 Python 3.12。训练和解释模型时强烈建议使用支持 CUDA 的 GPU。

```bash
conda create -p ./venv python=3.12 -y
conda activate ./venv
pip install -r requirements.txt
```

PySR 会安装并调用 Julia，首次运行可能需要额外时间准备 Julia 环境。

## 数据

大体积数据将通过 GitHub 之外的渠道发布。请将美国、全球南方和 GitHub 原始路网分别放入 `data/raw/`、`data/raw_globalsouth/` 和 `data/raw_github/`，将美国和全球南方的生成场景分别放入 `data/augmentation/` 和 `data/augmentation_globalsouth/`。仓库已包含体积较小的元信息和城市列表。

## 运行流程

请在仓库根目录依次运行：

```bash
bash scripts/1.prepare_data.sh
bash scripts/2.train_surrogate.sh
bash scripts/3.run_decoupler.sh
bash scripts/4.get_lowdim_functions.sh
bash scripts/5.extract_formula.sh
```

100 个城市的数据生成与完整训练耗时较长。`scripts/` 还包含容量与 OD 反事实实验及速度测试；实验结果默认写入 `logs/`。

批处理脚本默认以单机模式运行。如需多机协同，请启动 `src/utils/share/lock_server.py`，并在运行脚本前设置 `LOCK_SERVER=http://主机:端口`。

## 许可证

本项目采用 [MIT License](LICENSE)。
