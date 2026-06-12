# PCINet
PCINet: A Prior-Guided Correlation Interaction Network for High-Resolution Remote Sensing Image Change Detection.
## 目录结构

```text
PCINet_code/
  train.py                         # 训练入口，支持 LEVIR / SYSU / GZCDD
  predict.py                       # 推理与论文风格可视化输出
  assess.py                        # 指标计算工具
  flops_counter.py                 # FLOPs / 参数量统计
  poly.py                          # poly learning-rate helper
  requirements.txt                 # 主要 Python 依赖
  EXPERIMENT_STEPS.md              # 实验步骤说明
  PCINet_RemoteSensing_Outline_CN.md
  models/
    MobileNetV2.py                 # MobileNetV2 backbone
    pgc_cdnet.py                   # PCINet 主体网络与模块
  dataload/
    LEVIRdataset.py                # LEVIR-CD 数据加载
    SYSUCDdataset.py               # SYSU-CD 数据加载
    GZCDDdataset.py                # GZ-CDD 数据加载
```


## 数据目录

代码不会自带数据集。运行前请把数据按以下结构放到代码包同级目录或通过对应环境变量 / 参数指定：

```text
LEVIR-CD/
  train/A
  train/B
  train/label
  test/A
  test/B
  test/label
```

SYSU-CD 与 GZ-CDD 默认识别 `train/A`、`train/B`、`train/label` 以及 `test/A`、`test/B`、`test/label`，也支持部分数据集使用 `val` 或 `OUT` 作为评估/标签目录。可用 `DATA_ROOT`、`SYSUCD_ROOT`、`GZCDD_ROOT` 等环境变量，或在推理时传入 `--root`。

## 代码摘要

PCINet 使用 MobileNetV2 作为骨干网络，`models/pgc_cdnet.py` 中包含主网络及核心模块。训练脚本采用交叉熵、Dice loss、多尺度辅助输出和 prior auxiliary loss 组合优化；推理脚本可输出预测 mask、概率图、误差图、overlay、compare panel 和 prior 可视化图。

