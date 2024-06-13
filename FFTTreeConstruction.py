import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.fftpack import fft, ifft

def fft_alignment(seq1, seq2):
    # 将序列转换为数值序列（ASCII码）
    seq1_num = np.array([ord(c) for c in seq1])
    seq2_num = np.array([ord(c) for c in seq2])
    
    # 填充序列使其长度相同
    max_len = max(len(seq1_num), len(seq2_num))
    seq1_num = np.pad(seq1_num, (0, max_len - len(seq1_num)), 'constant')
    seq2_num = np.pad(seq2_num, (0, max_len - len(seq2_num)), 'constant')
    
    # 计算FFT
    fft_seq1 = fft(seq1_num)
    fft_seq2 = fft(seq2_num)
    
    # 傅里叶逆变换&卷积计算相关性
    correlation = ifft(fft_seq1 * np.conj(fft_seq2))
    return np.abs(correlation).mean() / max_len  # 使用均值并进行标准化


# UPGMA聚类算法
class UPGMA:
    def __init__(self, sequences):
        self.sequences = sequences  # 序列列表
        self.n = len(sequences)  # 序列数量
        self.dist_matrix = self.calculate_distance_matrix()  # 计算距离矩阵
        self.history = []  # 聚类历史
        self.cluster_tree = self.upgma()  # 构建聚类树

    def calculate_distance_matrix(self):
        n = self.n
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist_matrix[i][j] = self.sequence_distance(self.sequences[i], self.sequences[j])
                dist_matrix[j][i] = dist_matrix[i][j]
        print(dist_matrix)
        return dist_matrix

    def sequence_distance(self, seq1, seq2):
        score = fft_alignment(seq1, seq2)
        return -score  # 使用比对得分的负值作为距离

    def upgma(self):
        clusters = [[i] for i in range(self.n)]  # 初始化每个序列为单独的簇
        distances = self.dist_matrix.copy()
        while len(clusters) > 1:
            min_dist = np.inf
            to_merge = (-1, -1)
            print(f"Clusters Length:{len(clusters)}")
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    dist = self.cluster_distance(clusters[i], clusters[j], self.dist_matrix)
                    if dist < min_dist:
                        min_dist = dist
                        to_merge = (i, j)

            if to_merge == (-1, -1):
                raise ValueError("No clusters to merge, something went wrong!")

            self.history.append((clusters[to_merge[0]], clusters[to_merge[1]], min_dist))  # 记录聚合历史

            new_cluster = clusters[to_merge[0]] + clusters[to_merge[1]]  # 合并簇
            clusters = [clusters[k] for k in range(len(clusters)) if k not in to_merge] + [new_cluster]  # 更新簇列表

            print(f"Clusters to merge: {to_merge}")
            print(f"New cluster: {new_cluster}")
            print(f"Updated clusters: {clusters}")

            new_distances = np.zeros((len(clusters), len(clusters)))
            for i in range(len(clusters) - 1):
                for j in range(i + 1, len(clusters)):
                    new_distances[i][j] = new_distances[j][i] = self.cluster_distance(clusters[i], clusters[j], self.dist_matrix)

            distances = new_distances
            print(distances)

        return clusters[0]

    def cluster_distance(self, cluster1, cluster2, distances):
        return np.mean([distances[i][j] for i in cluster1 for j in cluster2])  # 计算两个簇之间的平均距离

    def plot_tree(self): # 可视化
        G = nx.DiGraph()
        pos = {}
        labels = {}

        def add_edges(cluster, parent=None, depth=0, pos_x=0):
            node_id = '-'.join(map(str, cluster))  # 将列表转换为字符串
            G.add_node(node_id)
            if parent:
                parent_id = '-'.join(map(str, parent))  # 将列表转换为字符串
                G.add_edge(parent_id, node_id)
            pos[node_id] = (pos_x, -depth)
            labels[node_id] = ', '.join(map(str, cluster))
            return node_id

        def build_tree(cluster, parent=None, depth=0, pos_x=0, spacing=1.5):
            if len(cluster) == 1:
                return add_edges(cluster, parent, depth, pos_x)
            left_cluster = cluster[:len(cluster) // 2]
            right_cluster = cluster[len(cluster) // 2:]
            left_node = build_tree(left_cluster, cluster, depth + 1, pos_x - spacing / (depth + 1))
            right_node = build_tree(right_cluster, cluster, depth + 1, pos_x + spacing / (depth + 1))
            return add_edges(cluster, parent, depth, pos_x)

        root = build_tree(self.cluster_tree)
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, labels=labels, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold')
        plt.title('UPGMA Evolutionary Tree')
        plt.show()
    
with open('./BRCA1Gene/CanisBRCA1.txt', 'r') as file:
    Canis_sequence = file.read().replace('\n', '')
with open('./BRCA1Gene/GallusBRCA1.txt', 'r') as file:
    Gallus_sequence = file.read().replace('\n', '')
with open('./BRCA1Gene/HumanBRCA1.txt', 'r') as file:
    Human_sequence = file.read().replace('\n', '')
with open('./BRCA1Gene/MouseBRCA1.txt', 'r') as file:
    Mouse_sequence = file.read().replace('\n', '')
with open('./BRCA1Gene/RattusBRCA1.txt', 'r') as file:
    Rattus_sequence = file.read().replace('\n', '')

sequences = [Canis_sequence, Gallus_sequence, Human_sequence, Mouse_sequence, Rattus_sequence]
upgma = UPGMA(sequences)
print("UPGMA Cluster Tree:", upgma.cluster_tree)
upgma.plot_tree()