###  完整的数据-模型-训练路线

1. **数据预处理层**：BPE Tokenizer训练与内存映射（mmap）数据加载
    
2. **模型架构层**：从RMSNorm到RoPE，从SwiGLU到Causal Self-Attention的完整实现
    
3. **优化层**：AdamW优化器、余弦退火学习率调度、梯度裁剪
    
4. **生成层**：Temperature scaling与Nucleus (top-p) sampling的解码策略
### 方法具体实现及数学原理
#### 一 . BPE Tokenizer
找到词级与字符级之间的平衡点：
**高频词保持完整**（如 "the", "and"），**低频词拆分成有意义的子词片段**（如 "playing" → "play" + "ing"）。
BPE训练是一个自底向上（Bottom-up）的聚类过程：

1. **初始化**：将词汇表设为所有字符（如26个字母+标点）
2. **计数**：统计所有相邻字符对（bigram）的出现频率 
3. **合并**：将频率最高的字符对合并为新符号，加入词汇表
4. **迭代**：重复步骤2-3，直到词汇表达到预定大小（如32k）
#### 方法的具体实现：

设训练语料为文本集合 D ，目标词汇表大小为 N 。
### $V_0 = \text{所有Unicode字符} \cup \{\langle \text{UNK} \rangle, \langle \text{SPC} \rangle\}$
#### 1.**构建频率表**： $\text{count}(x, y) = \sum_{w \in \mathcal{D}} \text{freq}(w) \cdot \mathbf{1}[(x, y) \in w]$
    
    其中 (x,y) 是相邻符号对，freq(w) 是词 w 在语料中的频率。后面是示性函数
#### 2.**选择最佳合并对**：$(x^*, y^*) = \mathop{\arg\max}_{(x, y)} \text{count}(x, y)$
#### 3.**更新词汇表和语料**： $V_{k+1} = V_k \cup \{x^* y^*\}$
    
    将所有 x∗y∗ 替换为新符号。
#### 4.**终止条件**：当 $∣V_k​∣=N$ 时停止。
#### 关键机制解析：
词尾标记：允许模型学习特定后缀含义
贪婪匹配策略：按照训练的合并顺序应用规则，虽不是最优规则但非常高效
            eg：merge=[('e','r'),('er','/w'),('l','o')........ ]
             用这个编码lower['l','o', 'w', 'er', '</w>']
             分别应用（e,r)['l' 'o' 'w' 'er']
                      (er,/w)[l o w 'er</w>']  
                      ........
            PS:如果存在er,o这一规则，那么这个规则将被跳过
   字符回退： 对于OOV字符（如训练集中没有的中文、表情符号），BPE会回退到字节级表示（Byte-level BPE）。GPT-2使用的BBPE（Byte-level BPE）

### 二：transformer现代版架构
#### 1.在原版的基础上使用简化的层归一化（因为发现layernorm中减去均值并非必要）
## $\text{RMSNorm}(x_i) = \frac{x_i}{\text{RMS}(x)} \cdot g_i \text{其中：} \text{RMS}(x) = \sqrt{\frac{1}{d_{\text{model}}} \sum_{j=1}^{d_{\text{model}}} x_j^2 + \epsilon}$
d 代表模型维度
gi 是可学习参数
#### 2.RoPE位置编码
通过旋转矩阵将位置信息编码到Query/Key向量中，使得Attention计算中自然体现相对位置信息。
### $R_k^m = \begin{bmatrix} \cos(m\theta_k) & -\sin(m\theta_k) \\ \sin(m\theta_k) & \cos(m\theta_k) \end{bmatrix}$
对于位置为m的向量x
RoPE(x,m)=x*R^m

### 3.SwiGLU:门控激活函数（替代RelU）
### $\text{SwiGLU}(x) = (\text{Swish}(xW_1) \odot (xW_3))W_2$

**⊙ （逐元素相乘）**


## 三：训练设置
1.使用Adam作为优化器
2.
#### $\alpha_t = \begin{cases} \frac{t}{T_w} \alpha_{\max} & t < T_w \text{ (warmup)} \\ \alpha_{\min} + \frac{1}{2}(\alpha_{\max} - \alpha_{\min}) \left(1 + \cos\left(\frac{t - T_w}{T_c - T_w} \pi\right)\right) & T_w \le t \le T_c \\ \alpha_{\min} & t > T_c \end{cases}$
