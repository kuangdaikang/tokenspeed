# 作业 3 · Ascend HBM 与 Host 分层缓存

Qwen3.5 的 BF16 文本推理通过 TokenSpeed 正式调度器运行。Ascend 使用分页 Attention、GDN 与卷积，实现沿用 LCM 块池、Host L2、请求快照和跨 rank ACK。

## 状态与生命周期

| 状态 | 保存粒度与恢复位置 |
| --- | --- |
| Attention K/V | 每层每 rank 的分页 K/V，前缀粒度为 128 token。尾页的缓存键包括有效 token 位置。恢复时通过调度器的块表定位新设备块。 |
| GDN 递归状态 | 每线性 Attention 层的 FP32 `H×V×K` 本地分片，表示消费前 `T` 个 token 之后的状态。输入 checkpoint 和可写输出分别由两个 slot 表示。 |
| 卷积状态 | 与 GDN 使用同一个 `T`，保存最近 `W−1` 个原始输入。Ascend 适配在调用算子前补齐它的窗口接口，并将结果的最后 `W−1` 个位置写回。 |

Host L2 用内容哈希、缓存组和页位置定位已发布的只读前缀。推理请求的写入页与私有快照由请求及其调度器资源持有。普通 D2H 持有设备源块直到 ACK。retraction 在 forward 已完成的位置取镜像，设备块可以在同一计划内授予其他请求。forward stream 等待 D2H 完成后，才执行新使用者的清零、写入和推理。

可共享的镜像块进入 L2，剩余块和 slot 状态进入请求快照池。快照期间持有的 Host 块受 pin 保护。恢复为请求分配新的 HBM 页，两个 TP rank 分别完成本地分片的 H2D，并在调度器处理共同 ACK 后从相同的 `T` 续算。在 prefix admission 中，复用区间取 K/V、GDN 与卷积都可恢复的最长共同前缀。普通 prefill chunk 的结束位置可以只有 K/V 记录，因此高复用实验先发布较短请求的完整状态，再提交延长版请求。

Host L2 先从可用容量分配，紧张时沿用缓存协调器的淘汰顺序。只有未被引用的 L2 条目可以淘汰。后续请求从仍可用的共同 checkpoint 重算后缀，完全 miss 时重新 prefill。请求快照由活跃请求持有，恢复或结束时返还资源。

## 复现

以下命令从仓库根目录运行。使用本地的 `.venv-hw3`、CANN 9.0.0、PyTorch/torch_npu 2.9.0 和 Triton-Ascend 3.2.1。调度器从当前 checkout 构建，`OPENSSL_ROOT_DIR` 指向本机已有开发文件。

```bash
source .venv-hw3/bin/activate
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export PYTHONPATH="$PWD/python:$PWD/tokenspeed-kernel/python:$PWD/tokenspeed-kernel-npu/python${PYTHONPATH:+:$PYTHONPATH}"
export TOKENSPEED_HOST_CACHE_TRACE=1
OPENSSL_ROOT_DIR=/opt/miniconda3/envs/aics CMAKE_BUILD_PARALLEL_LEVEL=6 \
  python -m pip install --no-deps -e ./tokenspeed-scheduler
```

使用作业目录里的两份本地权重。0.8B 固定 Hugging Face revision `2fc06364715b967f1860aea9cf38778875588b17`，4B 固定 `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`。权重经 ModelScope 下载后按对应 Hugging Face revision 的 SHA256 验证。

```bash
export ASCEND_RT_VISIBLE_DEVICES=6,7
mkdir -p hw3_results/logs
python test/hw3/offloading.py "$HOME/models/Qwen3.5-0.8B" \
  --case reference --tp 2 --device-tokens 12288 \
  --host-ratio 0 --snapshot-ratio 0 --steps 16 --port 31943 \
  --output hw3_results/0.8B-tp2-reference.json \
  > hw3_results/logs/0.8B-tp2-reference.log 2>&1
python test/hw3/offloading.py "$HOME/models/Qwen3.5-0.8B" \
  --case resume --tp 2 --device-tokens 12288 \
  --host-ratio 2 --snapshot-ratio 1 --force-retraction-interval 3 \
  --steps 16 --port 31943 --output hw3_results/0.8B-tp2-resume.json \
  > hw3_results/logs/0.8B-tp2-resume.log 2>&1
```

4B 的配置采用 `--device-tokens 16384 --steps 12`。`reference` 顺序执行 A/B，`resume` 在 A 生成第一个 token 后加入 B。两个请求共享一段只读前缀并使用不同尾部。正确性判据是各请求贪心生成的 token 和相同模型、相同 TP 的参考执行完全一致。逐 rank `RestoreDone` 使用同一组操作 ID，`[Scheduler] restore` 记录各次恢复的 `T`。

性能实验使用 0.8B、TP=1、`--device-tokens 12288`、260 个输入 token、4 个输出 token、12 个请求。`--case high` 先发布 4 组 256 token checkpoint 再轮流续写，`--case low` 使用不同前缀。关闭卸载使用两个 ratio 均为 0。开启时固定 `--snapshot-ratio 1`，L2 的 `--host-ratio` 使用 `0.5`、`2`、`6`。性能实验保持 `--force-retraction-interval 0`，另有 8 个设备 LCM 块的并发压力用例。

```bash
python test/hw3/summarize.py --results hw3_results --logs hw3_results/logs \
  --output hw3_results/summary.json
```

`[cache_tiers]` 输出设备已用与可分配块、Host 使用量、前缀命中/未命中、淘汰、快照空闲容量、实际新分配页和续推位置。`[host_cache_copy]` 在本 rank 拷贝完成后记录页数据字节数、设备事件耗时和物理块映射。Host L2 命中指一次 admission 实际载入至少一个 Host 块；关闭两级 Host 资源时这组计数为零。快照 slot arena 的内存另计。性能数据中的 TTFT 含实际 prefill、传输和排队，按需恢复的 H2D 耗时用设备事件单列。

## 实测结果

0.8B 的所有性能配置都分配了 12 个可用设备 LCM 块，HBM 缓存 arena 为 156 MiB，含空块。每档完成 12 个请求。下表中的 Host 峰值是被占用的 L2 父块，Host 预分配容量还包含剩余 L2 和单独的快照池。

| 复用 | Host L2 比例 | 命中 / 未命中 / 淘汰 | Host L2 峰值 MiB | D2H / H2D MiB | 实际 prefill token | TTFT 中位 ms | 输出 token/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 高 | 关闭 | 0 / 0 / 0 | 0 | 0 / 0 | 3120 | 310.68 | 7.63 |
| 高 | 0.5 | 0 / 12 / 120 | 72 | 519.2 / 0 | 3120 | 316.20 | 7.48 |
| 高 | 2 | 0 / 12 / 70 | 288 | 371.8 / 0 | 3120 | 300.67 | 7.74 |
| 高 | 6 | 12 / 0 / 0 | 684 | 161.1 / 259.6 | 48 | 106.00 | 12.22 |
| 低 | 关闭 | 0 / 0 / 0 | 0 | 0 / 0 | 3120 | 337.30 | 6.86 |
| 低 | 0.5 | 0 / 12 / 114 | 72 | 553.2 / 0 | 3120 | 294.48 | 7.94 |
| 低 | 2 | 0 / 12 / 90 | 288 | 539.3 / 0 | 3120 | 325.28 | 7.10 |
| 低 | 6 | 0 / 12 / 14 | 864 | 559.5 / 0 | 3120 | 321.24 | 7.16 |

高复用且 Host 容量充足时，每个请求从 256 token 的共同 checkpoint 恢复，只计算 `[256, 260)`。H2D 中位为 3.61 ms。Host 容量较小时，请求回到位置 0，重算 `[0, 260)`，其时间进入 TTFT。低复用产生多次 D2H，所有请求仍计算完整 prefill，表中的小样本时延差不用于估算算法收益。8 块压力配置下，A/B 两个请求在两档均完成，合计各做了 520 个 prefill token。

0.8B 的 TP=1 和 TP=2 交替执行均通过逐 token 比对。两档都观察到 14 个真实恢复操作。TP=2 的 14 个操作全部在两个 rank 完成，取每个操作两 rank 中较慢的一侧，H2D 中位是 3.63 ms。结束时各 rank 的 Host pin 为零，快照池回到 12 个空闲块。另跑的 TP=1 采样分数用例中，每个请求生成 12 个 token，所选 token 的 logprob 最大绝对差为 0。

4B 的 TP=2 交替执行中，两会话各生成 12 个 token，均与参考完全一致。11 个恢复操作都收到两个 rank 的完成信号，H2D 中位是 4.10 ms。两个 rank 的 HBM 缓存 arena 合计 544 MiB，结束时各 rank 的 Host pin 为零，快照池各回到 16 个空闲块。

单芯片日志里的 A 第一次在 `T=263` 暂停。`WriteBackDone` 操作 5 写回状态组的设备父块 2 和 4，下一轮 B 为自己的线性 Attention 状态取得这两个父块。操作 7 将 A 的状态搬回设备，A 从 `T=263` 续推。块级证据见 `hw3_results/logs/0.8B-tp1-resume.log` 的这三组事件。

完整计数、各 rank 的容量和参考比对保存在 [summary.json](../../hw3_results/summary.json)。调度器回归 66 项通过，运行时回归 62 项通过、6 项跳过，内核回归 60 项通过、3 项跳过。原始日志在 `hw3_results/logs/`。
