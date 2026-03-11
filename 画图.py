import matplotlib.pyplot as plt
import numpy as np

# 数据整理
labels = ['Naive Top-K', 'Static MMR', 'Ours (Dynamic)']
# metrics_data = {
#     'Redundancy (Lower is better)': [0.0346, 0.0346, 0.0315],
#     'Avg Item Count': [4.84, 4.84, 4.73],
#     'Avg Raw Cosine Score': [0.2211, 0.2211, 0.2245],
#     'End-to-End (EM) (%)': [3.25, 3.25, 3.00]
# }
metrics_data = {
    'Redundancy (Lower is better)': [0.2908, 0.2908, 0.1556],
    'Avg Item Count': [19.90, 19.90, 15.56],
    'Avg Raw Cosine Score': [0.1525, 0.1525, 0.1827],
    'End-to-End (EM) (%)': [5.00, 5.25, 5.25]
}

# 绘图配置 (遵循 3x3 布局)
titles = list(metrics_data.keys())
colors = ['#aec7e8', '#ffbb78', '#2ca02c'] # 对应 [Top-K, Static MMR, Ours]

fig, axes = plt.subplots(2, 2, figsize=(18, 15))
axes = axes.flatten()

for i in range(len(axes)):
    ax = axes[i]
    if i < len(titles):
        title = titles[i]
        values = metrics_data[title]
        bars = ax.bar(labels, values, color=colors, edgecolor='black', alpha=0.8)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.set_ylim(0, max(values) * 1.3) # 预留数值显示空间
        
        # 数值标注
        for bar in bars:
            height = bar.get_height()
            if 'Count' in title: fmt = '{:.2f}'
            elif 'EM' in title: fmt = '{:.2f}%'
            else: fmt = '{:.4f}'
            
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    fmt.format(height),
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.6)
    else:
        # 隐藏未使用的子图
        ax.axis('off')

plt.tight_layout()
plt.savefig('experimental_results_3x3.png')