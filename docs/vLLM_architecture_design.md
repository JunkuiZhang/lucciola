# Lucciola 演进设计：向 vLLM 风格的连续批处理与 PagedAttention 架构演进

当前 Lucciola 为静态推理引擎。为了实现更高的吞吐量和更灵活的内存管理，本设计旨在将系统重构为支持**连续批处理 (Continuous Batching)** 和 **PagedAttention** 的动态推理架构，参考 vLLM 的核心理念。

## 1. 核心设计理念

传统的静态批处理存在严重的显存碎片化和请求排队等待问题。通过引入 **PagedAttention** 技术（将系统的 KV Cache 划分为固定小块的 Block）以及 **Continuous Batching**（在每次迭代步结束时动态加入或移除请求），可以极大提升系统的并发吞吐与 GPU 资源利用率。

## 2. 核心组件架构

为了实现上述目标，Lucciola 需要引入并实现以下几个核心抽象组件：

### 2.1 Sequence (序列与请求抽象)
在动态推理中，不再以单纯的 Tensor 作为最外层处理单元，而是以 `Sequence` 为核心。
*   **Request/SequenceGroup**：用户的单个请求（可能包含束搜索等多个生成序列）。
*   **Sequence**：单个正在生成的文本序列。维护当前的状态（Waiting, Running, Swapped, Finished）、已生成的 Token ID 列表、逻辑 Block 索引（Logical Token Blocks）。
*   **BlockTable**：记录当前 `Sequence` 的逻辑 Block ID 映射到物理 GPU Block ID 的映射表。

### 2.2 CacheEngine (KV Cache 内存引擎)
接管原来的静态 `gpu_arena`，实现分页内存的分配与管理。
*   **KVCache Memory Pool**：在系统初始化时，预先分配一大块连续的 GPU 显存用于存放 KV Cache。
*   **Block Allocator**：将显存划分为大小恒定的 Physical Block（例如每个 Block 存放 16 个 Token 的 K 和 V）。
*   **Swap Management**：实现 GPU 到 CPU 显存的被动卸载和主动加载（Swap-out / Swap-in），供调度器在显存不足时临时换出低优先级的序列。

### 2.3 Scheduler (调度器)
引擎的调度大脑，负责 Continuous Batching 的调度策略。
*   **Wait Queue, Running Queue, Swap Queue**：维护不同的请求状态队列。
*   **Step-level Scheduling**：在每个前向传播（Forward Pass）阶段开始前被调用：
    1. 判断当前显存是否有足够可用的 Physical Block。
    2. 将新请求从 `Wait Queue` 提升至 `Running Queue`（进入 Prefill 阶段）。
    3. 为 `Running Queue` 中存活的序列分配新的 Block（进入 Decode 阶段）。
    4. 如果物理块耗尽，将部分 `Running` 序列逐出到等待队列（Swap-out）或抢占 (Preemption)。
*   **BlockManager**：和 Scheduler 配合，维护空闲的 Physical Block 数量，管理分配与回收逻辑。

### 2.4 Worker / ModelRunner (模型执行与计算算子)
负责与 GPU Kernel 直接交互并执行前向传播。
*   **ModelRunner**：接收调度器传递过来的分配信息，组装 `InputMetadata`（包含本次迭代要执行的 token 张量、各个序列的相对长度、和 Block Tables 等映射信息）。
*   **PagedAttention Kernel**：改造现有的 `src/kernels/attention.cu`。修改原先基于连续指针的计算，使其能够基于 `block_tables` 在不连续的按 Block 切分的显存池中进行点积计算。

---

## 3. 分阶段实现路线图 (Phased Implementation Roadmap)

### 阶段一：内存池与 PagedAttention 算子重构 (Phase 1)
*   **目标**：实现在分页显存上的 Attention 计算。
*   **任务**：
    1. 将 `gpu_arena.cc` 升级为支持 Block 管理的 `CacheEngine` 和 `BlockManager`。
    2. 新增或修改 `src/kernels/attention.cu` 以支持 PagedAttention。
    3. 在测试文件中手动分配 `BlockTable` 并验证 PagedAttention 结果的数值正确性。

### 阶段二：序列抽象与元数据结构封装 (Phase 2)
*   **目标**：将数据流的颗粒度从 Tensor 提取为面向请求的 Request。
*   **任务**：
    1. 新增 `sequence.h/cc`，包含 `Sequence` 状态机的定义。
    2. 设计 `InputMetadata` 结构体，用于在 C++ 上层调度和底层 CUDA 核之间传输动态批处理所需信息（如各自的 seq_len, block_table）。

### 阶段三：连续批处理与调度器集成 (Phase 3)
*   **目标**：实现真正的 Iteration-level Batching。
*   **任务**：
    1. 开发 `Scheduler` 组件及其 FCFS（先来先服务）与内存感知调度策略。
    2. 修改核心引擎循环的代码，使执行由 `while(not_finished)` 变为由 `Scheduler::step()` 驱动的动态批处理。
    3. 支持一个批次中同时跑部分序列的 Prefill 阶段和另外一些序列的 Decode 阶段。

### 阶段四：前端异步服务化能力引入 (Phase 4)
*   **目标**：使得该引擎能对接真实高并发异步的请求并提升吞吐极值。
*   **任务**：
    1. 加入异步事件线程（Async Engine 面向对象网络通信）。
    2. 完善 Swap in/out 控制支持。
    3. 基于 Continuous Batching 和新的图模式进一步调优底层的 Linear 和 RMSNorm 操作。