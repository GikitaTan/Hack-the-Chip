# Hack the Chip —— NYU × Annapurna Labs

**Annapurna Labs** 是亚马逊负责设计 **AWS Trainium** 的团队。Trainium 是 AWS 上大量 AI 训练和推理任务所依赖的定制芯片。今天，我们将向你提供这套硬件、一个已能在上面运行的语言模型，以及一个我们关心的问题。

> **由 Agent 驱动、运行在你可控制芯片上的小模型，可以解决这个模型单独无法解决的问题。** 构建这样的循环，并证明它有效。

设计这些芯片的工程师将全天在活动现场。

**一句话概括核心思路：** 模型尝试解决问题，**检查器（checker）**给出评分，再将*分数和原因*一起反馈给模型，指导下一次尝试。决定这个循环是否有效的关键，不是模型，而是检查器。*“错了，偏差达 341%”* 虽然正确，却毫无帮助；*“sin(pi\*x) 这一项根本不应该存在”* 才指出了应该如何修改。这里的所有成果，都是通过八次独立实践中的教训摸索出来的。

如果你要恢复之前的工作，或者第一次接手这个项目，请查看 [`STATE.md`](STATE.md)，其中记录了当前的确切进展和下一步操作。

---

# 第一部分 —— 连接到你的芯片

每位参与者签到时都会领到一个**座位编号**。每个座位对应 Trainium 集群中的一个 Pod，名称为 `seat-<你的编号>`。它拥有自己的 Trainium 芯片，并且已在 `/workspace` 放置本仓库。你需要在第 6 步中自行启动模型，并通过自己的笔记本电脑连接到 Pod。**只能使用自己的座位，不要进入别人的 Pod。**

总共九步。第 1–5 步在**你的笔记本电脑上**执行；第 6–9 步在**你的 Pod 内**执行。

## 在你的笔记本电脑上

### 1. 安装 AWS CLI 和 kubectl

**macOS**（已安装 [Homebrew](https://brew.sh)）：

```bash
brew install awscli kubectl
```

没有 Homebrew？先使用 [AWS CLI 安装程序](https://awscli.amazonaws.com/AWSCLIV2.pkg)，然后执行：

```bash
curl -LO "https://dl.k8s.io/release/$(curl -Ls https://dl.k8s.io/release/stable.txt)/bin/darwin/arm64/kubectl" && chmod +x kubectl && sudo mv kubectl /usr/local/bin/
```

如果是 Intel 芯片的 Mac，将 `arm64` 替换为 `amd64`。

**Linux**（x86_64）：

```bash
curl -s "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip
unzip -q awscliv2.zip && sudo ./aws/install
curl -LO "https://dl.k8s.io/release/$(curl -Ls https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl && sudo mv kubectl /usr/local/bin/
```

**Windows**（PowerShell）：

```powershell
winget install -e --id Amazon.AWSCLI
winget install -e --id Kubernetes.kubectl
```

**之后关闭终端，重新打开一个新终端**，否则可能找不到刚安装的命令。检查：

```bash
aws --version
kubectl version --client
```

### 2. 粘贴活动凭证

组织者会在活动频道中发布三种格式的凭证：**macOS / Linux**、**Windows 命令提示符（CMD）**和 **Windows PowerShell**。只复制与你使用的终端对应的版本，并粘贴到你准备操作的终端中。格式如下：

```bash
# macOS / Linux
export AWS_ACCESS_KEY_ID="ASIA..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_SESSION_TOKEN="..."
```

```bat
:: Windows 命令提示符（不要引号）
SET AWS_ACCESS_KEY_ID=ASIA...
SET AWS_SECRET_ACCESS_KEY=...
SET AWS_SESSION_TOKEN=...
```

```powershell
# Windows PowerShell
$env:AWS_ACCESS_KEY_ID="ASIA..."
$env:AWS_SECRET_ACCESS_KEY="..."
$env:AWS_SESSION_TOKEN="..."
```

> **这些凭证只在当前终端中生效。** 每打开一个新终端或新标签页，都要重新粘贴。凭证也会**过期**：如果 `kubectl` 突然报 `ExpiredToken` 或要求提供凭证，请粘贴频道里最新的一组。你的 Pod 和工作不会因此受到影响。不要将凭证提交到 Git，也不要在其他地方公开。

### 3. 将 kubectl 配置为连接集群

所有操作系统都使用同一条命令：

```bash
aws eks update-kubeconfig --name hack-hyd --region ap-south-2
```

成功时会输出类似 `Added new context ... to ...kube/config` 的信息。如果出现访问权限或凭证错误，说明当前终端中没有第 2 步的凭证，需要重新粘贴。

### 4. 找到你的 Pod

使用**自己的**座位编号。下面的 42 只是示例：

```bash
kubectl get pod seat-42
```

理想状态是 `READY 1/1`、`STATUS Running`。如果显示 `Init` 或 `0/1`，说明仍在启动，等一分钟再查。如果显示 `Pending`、`CrashLoopBackOff`、`Error` 等其他状态，请联系组织者，他们会给你安排备用座位。

### 5. 进入 Pod 的 Shell

```bash
kubectl exec -it seat-42 -- bash
```

> **在出现新的命令提示符（`root@seat-42:/workspace#`）之前，不要继续输入。** 否则命令会在你的笔记本电脑上执行。如果需要第二个终端，可以重新打开一个终端、再次粘贴凭证（第 2 步），然后运行相同的 `kubectl exec` 命令。两个终端会进入同一个 Pod。

## 在你的 Pod 内

### 6. 启动模型

```bash
neuron-ls
cd /workspace
./serve.sh
```

`neuron-ls` 应该列出 `NEURON DEVICE`。随后 `serve.sh` 将在你的芯片上启动模型；当模型能够通过 `http://localhost:8000` 响应请求时，会显示 `READY`。第一次大约需要 4 分钟。如果脚本报错退出，请联系组织者。

**保持这个终端开启。** 再打开**第二个终端**，重新粘贴凭证（第 2 步），通过 `kubectl exec -it seat-42 -- bash` 进入同一个 Pod。第 7–9 步都在第二个终端运行。在 `/workspace` 中执行 `./serve.sh --logs` 可以查看模型日志。

### 7. 执行一行初始化命令

```bash
git config --global --add safe.directory /workspace
```

在这里执行**任何 Git 命令前**都需要这一步。只要 Pod 还在，你的工作文件就会保存在 `/workspace`；**一旦 Pod 被替换，文件就会丢失**。因此，想长期保留的内容应推送到自己的 Git 仓库。

### 8. 运行项目 1 —— 热传导杆 Agent

```bash
cd /workspace/projects/01-heat-rod-pde

python level0_heatrod.py --selftest     # 在信任分数之前，先证明检查器正确
python agent.py --level 0 --all         # 热身：第 0 轮即解决

nohup python agent.py --level 1 --all > run.log 2>&1 < /dev/null &    # 正式运行
tail -f run.log                         # Ctrl+C 只停止查看日志，任务继续运行
```

> **如果想让长时间运行的任务在连接中断后仍然继续，必须按这种方式启动。** `kubectl exec` 会话可能因为 Wi-Fi、笔记本休眠或 `connection reset by peer` 而中断；在该 Shell 前台运行的程序也会随之终止。因此要将任务放到后台运行，将输出写入日志，再查看日志：
>
> ```bash
> nohup python agent.py ARGS > run.log 2>&1 < /dev/null &
> tail -f run.log        # Ctrl+C 只停止查看日志，任务继续运行
> ```
>
> 断线后，可以重新执行 `kubectl exec -it seat-42 -- bash` 连接，再用 `tail -f /workspace/projects/01-heat-rod-pde/run.log` 查看进展。`pgrep -af agent.py` 可检查程序是否仍在运行。

Pod 内已经安装了 `sympy` 和 `numpy`。**这个项目已经解决全部 6/6 个问题。** 详细说明见 [`projects/01-heat-rod-pde/`](projects/01-heat-rod-pde/)。

### 9. 运行项目 2 —— Kernel Agent

```bash
cd /workspace/projects/02-kernel-agent

python nkibench.py --selftest

nohup python agent.py --all --rounds 8 --samples 4 --context 8192 --repeat 5 > run.log 2>&1 < /dev/null &
tail -f run.log                         # 重连后：tail -f /workspace/projects/02-kernel-agent/run.log
```

**这个项目尚未解决——这正是它的意义。** 它是一个真正的开放问题，而不是预设了标准答案的练习。详细说明见 [`projects/02-kernel-agent/`](projects/02-kernel-agent/)。

> Pod 会在每个 Shell 中自动设置 `HEATROD_BASE_URL`、`KERNEL_AGENT_BASE_URL` 和模型名称。运行 `env | grep -E 'HEATROD|KERNEL_AGENT'` 即可查看，无需自行 `export`。

**组织者说明：** 集群部署、凭证分享和座位分配见 [`workshop/FACILITATOR.md`](workshop/FACILITATOR.md)。`serve.sh` 现在直接在座位 Pod 内运行，不再启动 Docker 容器，因此 [`STATE.md`](STATE.md) 中有关独立实例的说明只是历史记录。

---

# 第二部分 —— 已完成的实验与测量，避免重复踩坑

下面的所有数字都来自本仓库在这套硬件上的实际运行。

## 运行一次不代表得到可靠结果

项目 2 使用完全相同的设置连续运行五次，中间没有做任何修改：

```
level 1: solved 0/5   all = [0.30, 0.30, 0.30, 0.30, 0.30]
level 2: solved 4/5   all = [1.00, 1.00, 1.00, 0.50, 1.00]
level 3: solved 0/5   all = [0.30, 0.30, 0.30, 0.30, 0.30]
level 4: solved 0/5   all = [0.62, 0.62, 0.62, 0.62, 0.62]
```

仅仅由于随机性，Level 2 的分数就在 0.50 和 1.00 之间波动。**应该报告成功率，而不是最好的一次结果**——可以使用 `--repeat N`。另外，Level 1、3、4 的结果*完全没有波动*：这表明它们遇到的是能力瓶颈，而非随机波动。因此，修改这些关卡后产生的变化更容易归因；Level 2 的变化则未必如此。

**这项做法的实际价值：** 团队曾尝试一项看起来很合理的 Prompt 改进——加入一个完整的分块计算（tiling）示例，针对三个关卡共同卡住的问题。修改前后各运行五次，结果 Level 2 的成功率从 4/5 降至 **0/5**，Level 4 的分数从 0.62 降至 **0.30**。这是明确的退步，因此撤销了修改。**如果没有 `--repeat`，这项改动很可能就会被保留**，因为单次运行的结果可能会被误认为普通的随机波动。

## 并不是所有 Token 都同样有用

使用共享的 `gpt-oss-20b` 跑同样的关卡。该模型在服务端使用**贪心解码（greedy）**：

| 关卡 | Qwen3-8B（8B，采样） | gpt-oss-20b（20B，贪心解码） |
|---|---|---|
| 1 平均池化 | 0/5，始终为 0.30 | 0/3，始终为 0.30 |
| 2 转置 | **4/5**，0.5–1.0 | **3/3**，始终为 1.00 |
| 3 单块矩阵乘法 | 0/5，始终为 0.30 | 0/3，始终为 0.30 |
| 4 分块矩阵乘法 | **0.62** | **0.30** |

- **贪心解码的多次运行在字节层面完全一致**：错误及其顺序相同，每轮耗时精确到十分之一秒也相同，生成的 Kernel 代码逐字符相同。一个模型适合用逐步对比的方法定位问题；另一个则需要多次运行求统计结果。
- **贪心解码在能够成功的场景里更稳定：** Level 2 成功 3/3，而采样模型是 4/5。能力上限相同，但波动更小。
- **在 Level 4，8B 模型比 20B 模型表现更好**：0.62 对 0.30。gpt-oss 在六轮尝试中有四轮*完全没有返回代码*；每轮生成了约 10,000 个字符的隐藏推理内容，却花了 21 秒得到空结果。**用于推理的容量就无法用于给出答案**，而 Agent 循环需要实际答案。

因此，*模型能力更强*并不等于*Agent 表现更好*。对于贪心解码模型，Prompt 是主要的调节手段：原样重发同一段文字毫无作用。这也是 Agent 要维护失败尝试记录（ledger）的原因。

## 推理模式付出的成本大于收益

在 Qwen3 上测量，唯一改变的是 `--think`：

| 模式 | 单轮耗时 | 结果 |
|---|---|---|
| 关闭思考 | **约 8 秒** | 返回代码，分数 0.30–0.62 |
| 开启思考 | **446 秒** | 约 9,900 字符时被截断，得分 **0.00** |

所有样本最终的 `finish_reason=length`，且没有可用代码。**对这两个模型而言，简单增加 Token 预算并不能解决问题**——应该缩短 Prompt。如果你的 Agent 没有返回任何内容，优先检查这里。

## 这些设置不是随意的个人偏好

| 设置 | 原因 |
|---|---|
| `TP=2`，而不是 4 | `TP=4` 虽然可以正常提供服务，但每轮耗时相同——循环受生成长度限制，而不是计算能力限制。TP=2 还可以为后续真实设备计时保留 NC 0–1。 |
| 本地使用 `--samples 4`，gpt-oss 使用 `--samples 1` | Qwen3 的多次采样会产生不同结果，且额外开销很小（服务器同时运行四个样本）。gpt-oss 使用贪心解码，采样四次只会产生四份相同答案。 |
| `--context 8192` | 修复用 Prompt 需要包含上一个 Kernel、检查器的指导信息和失败记录。4096 的上下文窗口会挤占回答空间。 |
| gpt-oss 使用 `--terse 1` | Prompt 过长会使模型只顾推理而不回答：1,866 字符的 Prompt 导致 13,245 字符的推理且没有代码；缩短到 581 字符后得到了可运行代码。 |

---

# 第三部分 —— 项目应该连接哪个模型

**一个 NeuronCore 不能被两个进程共享。** 不过你的芯片有**四个逻辑核心**，模型服务器只使用其中两个，所以还有两个空闲。模型运行时执行 `neuron-top`：NC 2–3 承载模型，NC 0–1 的占用为 0 B。

| 项目需求 | 使用的模型 | 核心分配 |
|---|---|---|
| 不需要访问芯片，检查器在 CPU 上运行 | **本地 Qwen3-8B** | 无需分配核心 |
| 需要在芯片上编译并运行 Kernel | **本地 Qwen3-8B** | 固定到空闲核心：`NEURON_RT_VISIBLE_CORES=0,1` |
| 需要独占整块芯片进行模型服务实验 | **共享 `gpt-oss-20b`**（[`gptoss/`](gptoss/)），使用独立硬件 | 全部四个核心 |

**目前两个示例项目都属于第一行**，因此不需要远程模型。项目 1 的检查器是 SymPy；项目 2 则有意使用 `nki.simulate` 在 **CPU** 上模拟 Kernel——每次迭代只需数秒，而不必真正编译，这才是 Agent 应该大量尝试的地方。只有进行真实芯片上的性能计时时才需要核心，但这部分目前尚未实现。`NEURON_RT_VISIBLE_CORES` 是文档提供的机制，团队尚未实际验证，因此可能需要调试。

## 实例已经有 vLLM，为什么还需要容器？

AMI 在 `/opt` 中预装了 Neuron 虚拟环境，最新版本是 vLLM-Neuron **0.21**。这个版本**不能提供 Qwen3-8B 的推理服务**：其模型注册表仅包含 `LlamaForCausalLM`、`GptOssForCausalLM`、`Eagle3LlamaForCausalLM` 和 `Qwen3VLForConditionalGeneration`，最后一个是**视觉**模型。如果使用它加载 Qwen3-8B，会得到：

```
AttributeError: type object 'Qwen3ForCausalLM' has no attribute 'from_configs'
```

这个错误看起来像安装损坏，实际上是模型支持问题。`serve.sh` 使用支持该模型的 **0.24** 容器。如果想使用预装的虚拟环境，可以运行 `openai/gpt-oss-20b`，并添加 `--hf-overrides '{"quantization_config": {}}'`，以 bf16 格式加载。

`serve.sh` 中有三个**必不可少**的参数，都是通过排查实际报错确定的：

- `NEURON_SKIP_EFA_AFFINITY=1`：不设置时，Worker 会因 `No EFA device found` 而终止。它会跳过一种 CPU 位置优化，该优化假设存在单芯片环境中并不具备的网络硬件。
- `--no-enable-prefix-caching`：前缀缓存要求分段预填充（segmented prefill）的大小至少为 512。
- `--num-gpu-blocks-override`：不设置会出现越界错误。`serve.sh` 根据上下文长度计算这个参数，因此修改上下文长度后仍能保持正确。

`--tensor-parallel-size 2` 与硬件匹配：一块设置为 LNC=2 的芯片对应两个逻辑 NeuronCore。`k8s/` 目录中保存了相同配置的 Kubernetes manifest 文件，这也是开发时使用的方式；在当前实例上不需要使用这些文件。

---

# 第四部分 —— 比赛挑战

围绕一个模型构建 **Agent**：模型尝试解决问题，某个组件**检查**其结果，再将检查结论和原因反馈给模型，指导下一次尝试。循环持续进行，直到问题被解决，或者 Agent 能诚实地说明自己无法解决。

**无论选择哪个项目，都需要提交：**

1. **你的检查器（checker）**，以及为什么接受或拒绝某个结果的判断依据。这是组织方会保留的核心成果。
2. **尝试日志（attempt log）**：记录每一次尝试及其分数，让其他人能看出循环如何运作。
3. **一页报告**：说明运行了什么、使用什么硬件、结果如何，包括运行次数及结果的波动范围。

**你不需要是程序员，也能做出重要贡献。** 最困难的部分是决定什么才算正确，以及如何将检查结果转化为下一步指令；这两项本质上都不是编程任务。每队 3–5 人；全部由五名程序员组成的队伍往往反而表现不好。

## 示例项目

| 编号 | 项目 | 状态 |
|---|---|---|
| **1** | [**热传导杆 Agent**](projects/01-heat-rod-pde/)：小模型使用能够检验物理正确性的检查器，配合自己调用的计算器，求解热方程问题。 | **已解决 6/6** |
| **2** | [**Kernel Agent**](projects/02-kernel-agent/)：Agent 编写能在芯片上运行的 NKI Kernel，并持续验证自己的输出。 | 可以运行；**尚未解决** |

两个项目的 README 开头都附有真实的循环输出记录，便于你在开始前判断项目情况，分辨它是在取得进展还是在无效尝试。

你也可以**提出自己的项目**：它必须能在提供的硬件上运行，并产出上述三项交付物。请在 11:30 前找一位 Annapurna 工程师讨论，他们会坦诚告诉你该项目是否能在一天内完成。

增加一个项目，只需要在 `projects/` 下新建一个文件夹，其中包含 README 和检查器。为项目 2 增加一种*运算（operation）*，需要一个 NumPy 参考实现、一个输入构造函数，以及一次注册调用——其中 Level 8 提供了完整示例。

---

# 第五部分 —— 共享的 `gpt-oss-20b` 服务端点

这是一个运行于 Trainium2 上、具有 200 亿参数的开放权重模型。它通过兼容 OpenAI 的 API 提供服务，运行在与你的实例分离的硬件上。你可以用它与本地模型对比，也可以在需要使用自己整块芯片运行代码时调用它。

```bash
export GPTOSS_BASE_URL="https://..."     # 组织者会提供这个地址
pip install httpx
python gptoss/chat.py
```

它有几个看起来像 Bug、实际上却不是 Bug 的特性：回答前会先进行推理；采用贪心解码，因此原样重试没有意义；而 `tools=` 参数不起作用。**在基于它进行开发前，请阅读 [`gptoss/README.md`](gptoss/README.md)。** 其中所有数据都有实测依据。

---

## 安装依赖

```bash
pip install -r requirements.txt
```

## 许可证 / 使用方式

示例代码按活动时的原样提供，可以自由复用。
[`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) 和 [`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b) 分别遵循各自的许可证。
