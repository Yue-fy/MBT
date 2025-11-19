针对 **Physical Review Letters (PRL)** 的详细研究提案。

此提案的核心卖点是：**将“安德森局域化（Anderson Localization）”这一经典凝聚态难题，与“非线性量子几何（Nonlinear Quantum Geometry）”这一现代热点结合，预测在拓扑相变临界点会出现非线性霍尔效应的“灾变”（发散性增强）。**

---

# 📝 研究提案：安德森临界点的非线性霍尔灾变 (Nonlinear Hall Catastrophe at the Anderson Criticality)

## 1. 科学目标 (Scientific Objectives)
1.  **核心假设：** 在由无序驱动的拓扑相变点（Topological Anderson Insulator transition point），电子波函数呈现分形（Fractal）特征。这会导致描述波函数“距离”的 **量子度规 (Quantum Metric, $g$)** 发生发散。
2.  **推论：** 由于二阶非线性霍尔电导 $\sigma^{(2)}$ 正比于量子度规偶极（QMD），我们预测 $\sigma^{(2)}$ 在相变点会表现出比线性电导 $\sigma^{(1)}$ 更剧烈、更敏锐的**临界发散 (Critical Divergence)**。
3.  **区分机制：** 证明对于 $\mathcal{PT}$ 对称的 MnBi$_2$Te$_4$，磁性无序会迅速破坏 BCD，但 QMD 驱动的“灾变”在强标量无序下依然稳健。

---

## 2. 理论模型构建 (Model Construction)

为了研究安德森相变，我们需要一个能够描述 MnBi$_2$Te$_4$ 且计算量适中的模型。推荐使用 **Li et al. (PRL 2021) [cite_start][cite: 2979-2986]** 中的 **2D 有效晶格模型**。这个模型已经成功描述了线性输运中的 QAH-QH 共存相，非常适合扩展到非线性研究。

### 2.1 哈密顿量 (Tight-Binding Form)
将 Li et al. [cite_start]的 $k \cdot p$ 模型 [cite: 2983] 离散化为正方晶格上的紧束缚模型：

$$H = \sum_{\mathbf{r}} \psi_{\mathbf{r}}^\dagger \epsilon_{\mathbf{r}} \psi_{\mathbf{r}} + \sum_{\mathbf{r}, \hat{\mu}} (\psi_{\mathbf{r}}^\dagger T_{\hat{\mu}} \psi_{\mathbf{r}+\hat{\mu}} + h.c.)$$

* **基底：** 4分量 (轨道 $\tau \otimes$ 自旋 $\sigma$)。
* **参数映射 (Mapping from $k \cdot p$ to TB):**
    * **On-site term ($\epsilon_{\mathbf{r}}$):** $(m_0 + m_1 - 4t_0 - 4t_1) \tau_z \sigma_0 + V_{dis}(\mathbf{r})$
    * **Hopping terms ($T_x, T_y$):**
        * $T_x = \frac{t_0+t_1}{2} \tau_z \sigma_0 - \frac{i(v_2+v_4)}{2} \tau_x \sigma_x$
        * $T_y = \frac{t_0+t_1}{2} \tau_z \sigma_0 - \frac{i(v_2+v_4)}{2} \tau_y \sigma_y$
    * **具体参数值：** 直接采用 Li et al. [cite_start]的拟合参数：$v_2=1.41, v_4=1.09, m_0=1.025, m_1=0.975, t_0=0.5, t_1=1.0$ [cite: 2987]。
* [cite_start]**磁场项：** 通过 Peierls 替换 $T_{\hat{\mu}} \rightarrow T_{\hat{\mu}} e^{i \phi}$ 引入垂直磁场 $B_z$ [cite: 2985]。

### 2.2 无序项 (Disorder) - 关键创新点
你需要两种无序来做对比：
1.  [cite_start]**标量无序 (Scalar Disorder):** $H_{dis} = \sum V_i c_i^\dagger c_i$，其中 $V_i \in [-W/2, W/2]$ [cite: 2986]。这是模拟样品不纯、掺杂的主要模型。
2.  **磁性无序 (Magnetic Disorder):** 在 $\tau_z$ (质量项) 上引入随机扰动，模拟 Mn 磁矩的随机翻转。

---

## 3. 数值计算方法 (Numerical Methodology)

使用 **Python + Pyqula + NEGF**。

### 3.1 为什么用 NEGF 而不是 Kubo？
* Kubo 公式通常需要本征态 $|n\rangle$，在强无序体系（特别是 $400 \times 400$ 这种大尺寸）中对角化极其昂贵。
* NEGF (递归格林函数, RGF) 可以处理开边界、大尺寸 disordered supercell 的输运，且自然包含**安德森局域化**效应。

### 3.2 二阶电流算符 (Method of Calculation)
我们不直接推导复杂的二阶格林函数公式，而是使用**有限差分法 (Finite Difference Method)** 计算二阶响应，这对学生来说最不容易出错且物理图像清晰。

**算法步骤：**
1.  **构建器件：** 定义一个长条形散射区 ($L \times W$)，左右连接半无限长电极。
2.  **施加横向电场 ($E_y$):** 这是一个 Trick。我们在 Hamiltonian 中并不真的加 $E_y$，而是利用 NEGF 计算 **横向电流响应**。
    * 更简单的方法：施加纵向偏压 $V_x$，计算横向电流 $I_y$。
    * 由于 $C_{3v}$ 对称性或无序破缺，$I_y$ 可能包含 $I_y \propto V_x^2$ 的项。
3.  **栅压调控 ($E_z$):** 为了分离 BCD 和 QMD，我们需要通过调节 On-site 能势 $\Delta U(z)$ 来模拟栅压 $E_z$（在 2D 模型中体现为层间电势差或对称性破缺项）。
4.  **核心公式：**
    $$G_{xy}^{(2)} \approx \frac{I_y(+V_x) + I_y(-V_x) - 2I_y(0)}{V_x^2}$$
    或者利用对称性分离：
    $$\sigma_{odd}^{(2)} (BCD) = \frac{\sigma^{(2)}(+E_z) - \sigma^{(2)}(-E_z)}{2}$$
    $$\sigma_{even}^{(2)} (QMD) = \frac{\sigma^{(2)}(+E_z) + \sigma^{(2)}(-E_z)}{2}$$

---

## 4. 执行指令步骤 (Instructions for PhD Student)

### Step 1: 代码实现与线性基准 (Week 1-2)
* **任务：** 使用 `pyqula` 复现 Li et al. (PRL 2021) 的 Fig. 1(b)。
* **检查点：**
    * 构建 $100 \times 100$ 的格点。
    * 加入无序 $W=4$。
    * 计算线性霍尔电导 $\sigma_{xy} = \text{Tr}[\Gamma G^r \Gamma G^a]$。
    * [cite_start]**验证：** 确认你能看到随磁场增加，$\sigma_{xy}$ 从 $1 e^2/h$ (QAH) 变为 $2 e^2/h$ (QAH+QH) 的平台特征 [cite: 2992-2995]。如果不复现，检查参数。

### Step 2: 开发二阶计算模块 (Week 3-4)
* **任务：** 编写计算二阶电流的函数。
* **方法：**
    * 在散射区加上微小偏压 $V_{bias}$（修改 Self-energy $\Sigma_L, \Sigma_R$ 的费米分布函数 $f_L, f_R$）。
    * 计算流向上下边缘的局域电流分布 $J_y(\mathbf{r})$，然后积分得到总横向电流。
    * 做 $+V$ 和 $-V$ 的差分，提取二阶分量。

### Step 3: 寻找“灾变”峰 (Week 5-8)
* **任务：** 扫描无序强度 $W$。
* [cite_start]**参数设置：** 固定在 $C=1$ 到 $C=2$ 的相变边缘（例如 $\Phi/\phi_0 \approx 0.04$ [cite: 3123]）。
* **操作：**
    1.  细致扫描 $W$ (例如 3.0 到 5.0)。
    2.  对每个 $W$ 点，做 100-500 次无序构型平均（必须做平均，否则全是噪声！）。
    3.  记录 $\sigma_{xy}^{(1)}$ (线性) 和 $\sigma_{xy}^{(2)}$ (非线性)。
* **预期结果：** 当 $\sigma_{xy}^{(1)}$ 发生跳变（相变）时，$\sigma_{xy}^{(2)}$ 应该会出现一个**巨大的尖峰**。

### Step 4: 标度律分析 (Scaling Analysis) (Week 9-10)
* **任务：** 验证这个峰是否发散。
* **操作：**
    * 改变系统尺寸 $L$ ($50, 100, 200$)。
    * 如果峰值高度随 $L$ 增加而增加，或者随 $|W-W_c|^{-\nu}$ 发散，这就是 **PRL 的核心证据**。

---

## 5. 论文图表规划 (Figures for PRL)

**Fig. 1: 相图与非线性灾变 (The Hook)**
* **(a)** 线性电导 $\sigma_{xy}^{(1)}$ 随无序 $W$ 的变化（显示 $1 \to 0$ 或 $1 \to 2$ 的平台）。
* **(b)** **核心图：** 二阶非线性电导 $\sigma_{xy}^{(2)}$ 随 $W$ 的变化。展示在相变点 $W_c$ 处，信号出现尖锐的 **Giant Peak**。
* **(c)** 示意图：解释安德森局域化临界点波函数的分形维数如何导致量子度规发散。

**Fig. 2: 标度律分析 (The Proof)**
* 双对数坐标图：$\log(\sigma^{(2)})$ vs $\log|W - W_c|$。
* 展示临界指数 $\gamma$。与安德森局域化的临界指数 $\nu$ 建立联系。这将物理深度拉满。

**Fig. 3: BCD vs QMD 的命运 (The Mechanism)**
* 对比 **标量无序 (Scalar)** 和 **磁性无序 (Magnetic)**。
* **Scalar:** QMD 峰值保留，BCD 可能也有峰值。
* **Magnetic:** BCD 信号消失（Flat line），QMD 峰值依然存在但位置移动。
* **结论：** 证明 QMD 是驱动临界非线性输运的普适机制，而 BCD 极其脆弱。

**Fig. 4: 实验预测 (Connection to Experiment)**
* 计算 $R_{xx}$ 和 $V_{2\omega}$ 随磁场 $B$ 的扫描图（模拟 Li et al. [cite_start]Fig. 1b [cite: 2939]）。
* 预测在 $R_{xx}$ 的峰值处，$V_{2\omega}$ 会有一个更加显著的信号，指导实验去哪里找这个效应。

---

## 6. 为什么这个提案能发 PRL？

1.  **物理深度：** 它不是简单的计算材料性质，而是建立了一个 **安德森局域化 (Anderson Localization)** $\leftrightarrow$ **非线性量子几何 (Nonlinear Quantum Geometry)** 的新联系。
2.  **反直觉：** 通常认为无序会抑制霍尔效应，你证明了在临界点无序会**放大**非线性效应。
3.  **及时性：** 完美解释了为什么实验上（Wang et al., Nature-like paper）在脏样品中能看到量子化平台和复杂的输运行为，且预言了新的非线性现象。