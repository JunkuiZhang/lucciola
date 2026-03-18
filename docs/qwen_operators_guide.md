# Qwen3.0 推理核心算子实现路线图 (Learning Path)

为了实现 Qwen3.0 的纯文本推理引擎（无严重优化，注重逻辑易懂性），我们主要分为三个阶段：**前期准备算子**、**注意力机制相关算子**、**前馈网络及收尾算子**。这也是大模型前向传播计算图的顺序。完全从零写 CUDA 时，可以参照这个顺序逐个击破。

## 第一阶段：Embedding 与 归一化 (基础)
1. **Embedding 算子 (词表查找)**
   - **功能**: 根据输入的 Token ID，从 `model.embed_tokens.weight` 中拉取对应的特征向量。
   - **实现思路**: 其实就是一个简单的内存搬运算子。开启若干个线程，每个线程负责根据 token index 拷贝对应的一行 embedding 数据到输入 hidden_states 显存中。
2. **RMSNorm 算子 (Root Mean Square Normalization)**
   - **功能**: 计算张量的均方根并进行归一化，乘以对应的层权重 (weight)。
   - **出现位置**: Attention 之前 (`input_layernorm`)，MLP 之前 (`post_attention_layernorm`)，以及模型最后的 `norm`。
   - **实现思路**: C++ 或 CUDA 简单实现。先求每一个 token 向量的平方和，除以隐藏层维度得到均方根并加上极小的 epsilon，然后再用原向量除以这个均方根，最后与层权重逐元素相乘。可以通过 Block Reduce 来做基础的并行求和。

## 第二阶段：核心算子 - Attention 模块
本阶段包含大语言模型最核心的注意力交互逻辑，我们采用最朴素直观的 CUDA 核函数手写实现。

### 3. Linear Layer 算子 (线性映射 / 矩阵乘法 GEMM)
- **功能**: 计算大规模全连接层 $Y = X \cdot W^T$。在 LLM 中，几乎 80% 的时间都在做这个操作。
- **数据流 (Data Flow)**: 
  - `Input (X)`: `[num_tokens, in_features]`
  - `Weight (W)`: `[out_features, in_features]`
  - `Output (Y)`: `[num_tokens, out_features]`
- **实现细节与注释规范**:
  - 我们将手写一个 `linear_forward`。为了“懂逻辑”，初始版本每个线程计算输出矩阵 $Y$ 的一个元素或一行。
  - 会大量利用之前封装的 `Pack128` (float4) 来加速内积运算 (Dot Product) 中的内存读取。
  - **函数签名**: `void linear_forward(void* out, const void* input, const void* weight, int m, int n, int k, cudaStream_t stream)`。其中 `m=num_tokens`, `n=out_features`, `k=in_features`。

### 4. RoPE 算子 (旋转位置编码 - Rotary Position Embedding)
- **功能**: 将绝对位置编号转化为由正余弦构成的旋转矩阵，作用于 Q 和 K。它让模型懂得“词与词之间的距离”。
- **数学公式**: 把特征维度两两分组 $(x, y)$，应用复数旋转：
  $x' = x \cos(m\theta) - y \sin(m\theta)$
  $y' = x \sin(m\theta) + y \cos(m\theta)$
- **数据流 (Data Flow)**:
  - `Input Q / K`: 分别是 `[num_tokens, num_heads, head_dim]` 的张量。
  - `pos_ids`: `[num_tokens]` 记录每个 token 当前的绝对位置 $m$。
- **实现细节与注释规范**:
  - Qwen 使用的 RoPE 是按元素对相交的方式。`head_dim` 为 128，所以有 64 个旋转角度 $\theta$。
  - CUDA 的每一个线程处理一个 Token 的一个 Head 的一对 $(x, y)$。
  - **函数签名**: `void rope_forward(void* q, void* k, const int* pos_ids, int num_tokens, int num_q_heads, int num_k_heads, int head_dim, float rope_theta, cudaStream_t stream)`

### 5. 朴素注意力算子 (Naive Scaled Dot-Product Attention)
- **功能**: 计算 Q 和 K 的相似度，过 Softmax 得到权重，再乘以 V。
- **实现细节与注释规范**:
  - 对于"懂逻辑"的阶段，我们完全不需要写极度复杂的 FlashAttention。分三小步走：
  - **QK^T / Mask 判断**: 求 Q K 内积除以 $\sqrt{128}$，若遇到越界位则替换为极小值。
  - **Softmax / Attention * V**: 取指数并求和归一化，再乘对应的 V。

### 6. KV Cache 管理与拷贝算子
- **功能**: 逐字生成（Autoregressive Decode）的核心。把每一层新算出的单 Token `K/V`，按步长追加进显存池中。
- **实现细节与注释规范**:
  - 定义内存布局 `[num_layers, num_kv_heads, max_seq_len, head_dim]`，然后做精确的指偏移拷贝。

## 第三阶段：FFN 模块与输出生成
7. **SwiGLU / SiLU 激活函数算子**
   - **功能**: 处理 MLP 里的门控逻辑。公式表示为 $\text{SiLU}(Gate) \times Up$ 以及 $\text{SiLU}(x) = x \cdot \mathrm{sigmoid}(x)$
   - **出现位置**: 每个 Transformer Block 里的 MLP 模块中。
   - **实现思路**: 前面走完 `gate_proj` 和 `up_proj` 两个 Linear 矩阵乘之后，写一个并行的逐元素 Kernel：对 `gate_proj` 的张量做 $\mathrm{sigmoid}$，然后与原本自身相乘（这就是 SiLU），再与同样位置的 `up_proj` 张量做标量相乘即可完成 SwiGLU。
8. **Argmax / Greedy Sampling (采样算子)**
   - **功能**: 对最终过完 `lm_head` 后的 Logits (可能长达十几万长度)，寻找出最大值的索引，它也就是我们要预测的下一个字。
   - **实现思路**: CUDA 的 Block Reduce 实现最大值寻找。先只写一个找最大的逻辑（Greedy Search），等彻底跑通了模型打印出了可以理解的一句话，后面再考虑去写复杂的带有 top-p / temperature 的随机采样逻辑。

---

### 开发总结提示：
对于你自己写的 `lucciola` 推理引擎，你已经做好了权重分离和显存申请（GPU Arena）。接下来的工作就是：分配好这十几个张量内存 -> 写一个小算子 -> 用小算子把这块内存的值算进下一块内存 -> 如此循环直到最后。

*最容易卡住的地方往往不是写某个特定公式，而是 CUDA 多维张量数组下标算错/Stride步长没对齐。加油！*
