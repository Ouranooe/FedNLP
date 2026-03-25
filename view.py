import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import traceback

DATA_FILE = os.path.expanduser("~/fednlp_data/data_files/20news_data.h5")
PART_FILE = os.path.expanduser("~/fednlp_data/partition_files/20news_partition.h5")
PART_METHOD = "niid_label_clients=20_alpha=1"

def draw_heatmap():
    print("📂 正在全速解析 20News 数据，绝对安全模式启动...")
    try:
        with h5py.File(DATA_FILE, 'r') as data_h5, h5py.File(PART_FILE, 'r') as part_h5:
            
            # 1. 确定 Client 目录结构
            part_group = part_h5[PART_METHOD]
            if 'partition_data' in part_group:
                client_group = part_group['partition_data']
            else:
                client_group = part_group
                
            client_ids = sorted([int(k) for k in client_group.keys() if k.isdigit()])
            num_clients = len(client_ids)
            
            # 判断 Y 到底是不是文件夹
            is_y_group = isinstance(data_h5['Y'], h5py.Group)
            
            client_label_counters = {}
            all_labels_set = set()
            
            print("⏳ 正在逐个提取真实英文标签，请稍候...")
            for cid in client_ids:
                indices = client_group[str(cid)]['train'][()]
                client_labels = []
                
                for idx in indices:
                    # 安全读取单个标签
                    if is_y_group:
                        label = data_h5['Y'][str(idx)][()]
                    else:
                        label = data_h5['Y'][idx]
                        
                    # 兼容 Bytes 和 String
                    if isinstance(label, bytes):
                        label = label.decode('utf-8')
                    else:
                        label = str(label)
                        
                    client_labels.append(label)
                    all_labels_set.add(label)
                    
                client_label_counters[cid] = Counter(client_labels)
                
            classes = sorted(list(all_labels_set))
            num_classes = len(classes)
            print(f"\n🏷️ 成功提取 {num_classes} 个类别！")
            
            dist_matrix = np.zeros((num_clients, num_classes))

            print("\n📊 === 20News 客户端极端偏科展示 (仅显示 Top-3 主导类别) ===")
            for cid in client_ids:
                counter = client_label_counters[cid]
                
                # 获取数量前三的类别
                top_3 = counter.most_common(3)
                top_str = " | ".join([f"[{k}]: {v:4d}条" for k, v in top_3])
                
                # 统计一下有几个类是 0 条
                zero_classes = sum(1 for c in classes if counter.get(c, 0) == 0)
                total_items = sum(counter.values())
                
                print(f"🤖 Client {cid:2d} (共 {total_items:4d} 条) => 👑主导: {top_str} ... ⚠️完全缺失: {zero_classes} 个类别!")
                
                for c_idx, c in enumerate(classes):
                    dist_matrix[cid, c_idx] = counter.get(c, 0)
            
            # 2. 绘制并保存热力图
            plt.figure(figsize=(18, 10))  # 加宽以容纳20个类
            sns.heatmap(dist_matrix, annot=True, fmt="g", cmap="Purples", 
                        xticklabels=classes,
                        yticklabels=[f"Client {c}" for c in client_ids])
            
            plt.title(f"20News Distribution (Alpha=1.0) across {num_clients} Clients")
            plt.xlabel("News Newsgroups (20 Topics)")
            plt.ylabel("Client ID")
            plt.xticks(rotation=45, ha='right') # X轴倾斜，防止英文重叠
            plt.tight_layout()
            
            save_path = "20news_heatmap_alpha1.png"
            plt.savefig(save_path, dpi=300)
            print(f"\n✅ 成功！热力图已保存为: {save_path} (请下载查看)")
            
    except Exception as e:
        print(f"\n❌ 运行出错:")
        traceback.print_exc()

if __name__ == "__main__":
    draw_heatmap()