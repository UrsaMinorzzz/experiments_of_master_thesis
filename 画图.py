import matplotlib.pyplot as plt
import numpy as np

# 1. 全局配置学术级字体和字号
plt.rcParams.update({
    'font.size': 14,          # 全局基准字号
    'axes.titlesize': 18,     # 子图标题字号
    'axes.labelsize': 16,     # 坐标轴标签字号
    'xtick.labelsize': 15,    # X轴刻度字号
    'ytick.labelsize': 14,    # Y轴刻度字号
    'font.family': 'serif',   # 论文标配衬线字体
})

labels =['Naive Top-K', 'Static MMR', 'Ours (Dynamic)']
# 加入趋势箭头指示，降低审稿人的认知负荷
metrics_data = {
    'Redundancy (Lower is better) ↓':[0.0423, 0.0423, 0.0486],
    'Avg Item Count ↓':[4.82, 4.82, 4.72],
    'Avg Raw Cosine Score ↑': [0.2407, 0.2407, 0.2468],
    'End-to-End (EM) (%) ↑':[4.99, 4.99, 5.28]
}

colors =['#aec7e8', '#ffbb78', '#2ca02c'] 

# 2. 调整图幅比例，适应论文排版 (14, 11 更紧凑)
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.flatten()

for i, title in enumerate(metrics_data.keys()):
    ax = axes[i]
    values = metrics_data[title]
    
    # 3. width=0.45 让柱子变细，留出呼吸空间
    bars = ax.bar(labels, values, color=colors, edgecolor='black', alpha=0.8, width=0.45)
    
    ax.set_title(title, fontweight='bold', pad=15)
    
    # 动态预留顶部空间，防止文字被切掉
    y_max = max(values)
    ax.set_ylim(0, y_max * 1.25) 
    
    # 数值标注 (加大字号)
    for bar in bars:
        height = bar.get_height()
        if 'Count' in title: fmt = '{:.2f}'
        elif 'EM' in title: fmt = '{:.2f}%'
        else: fmt = '{:.4f}'
        
        ax.text(bar.get_x() + bar.get_width()/2., height + (y_max * 0.02),
                fmt.format(height),
                ha='center', va='bottom', fontsize=14, fontweight='bold')
                
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    # 4. 去除右侧和顶部的边框线 (学术图表去杂乱化)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
# 5. 必须存为 PDF 并在 LaTeX 中以 \includegraphics 引入
plt.savefig('experimental_results_2x2_academic.pdf', dpi=300, bbox_inches='tight')