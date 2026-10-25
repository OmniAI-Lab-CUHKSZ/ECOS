# 环境准备

ALFWorld 和 WebShop 使用仓库内保留的文本环境适配器。模型权重、原始基准资产、Java 和索引不放进源码仓库。

## ALFWorld

安装 `requirements/environments.txt` 后，使用官方 `alfworld-download` 下载数据：

```bash
export ALFWORLD_DATA="$PWD/assets/alfworld"
alfworld-download
```

目录应包含 `json_2.1.1/{train,valid_seen,valid_unseen}` 和 `logic/{alfred.pddl,alfred.twl2}`。
若已有数据，直接设置到现有目录。SFT 材料中的 gamefile 是相对 `json_2.1.1` 的路径。
训练使用 train；固定128使用 valid_seen 的固定循环。完整274评测包含 valid_seen140 + valid_unseen134。

## WebShop

安装 Java 11 并设置 `JAVA_HOME`，安装 spaCy 的 `en_core_web_sm`。默认是原实验使用的1k商品环境、合成目标、catalog seed2026。
准备目录：

```text
assets/webshop/
  data/items_shuffle_1000.json
  data/items_ins_v2_1000.json
  data/items_human_ins.json
  search_engine/indexes_1k/
```

```bash
python scripts/prepare_webshop.py --asset-root assets/webshop --download
export WEBSHOP_ASSET_ROOT="$PWD/assets/webshop"
```

脚本从官方公开地址下载三份 JSON，沿用现有商品预处理生成 Lucene 索引；已有完整文件会复用，已有非空索引不会被覆盖。
也可以传入已经准备好的资产。默认不需要浏览器、ChromeDriver、ResNet 特征或外部网页服务。
WebShop 的成功仍是最终环境 `won`，不会把部分购物分数当作成功。

## 多机

核心框架保留 `trainer.nnodes`、`trainer.gpus_per_node`、`ray_init.address`、FSDP 和 elastic logical-world 支持。
`scripts/train_rl.py` 是单机的完整私有 Ray / 原生归档启动器。多机时需要为自己的环境提供 Ray 集群、逐节点网络变量和受限归档服务计划，使用渲染后的同一方法配置；不附带特定服务器的账号或自动 SSH 控制器。
配置数组的物理卡数之和应在2–8，logical world保持8。恢复到不同物理卡数时保留原生 Model、Adam、extra、数据与环境状态，使用框架的原生转换实现。
