import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# 动态规划（DP）序列比对算法
def dp_alignment(seq1, seq2, match=2, mismatch=-1, gap_open=-5, gap_extend=-1):
    m, n = len(seq1), len(seq2)
    dp = np.zeros((m+1, n+1))  # DP表，用于记录比对得分
    gap_matrix = np.zeros((m+1, n+1))  # Gap矩阵，用于记录gap位置

    # 初始化边界条件
    for i in range(1, m+1):
        dp[i][0] = gap_open + (i-1) * gap_extend
    for j in range(1, n+1):
        dp[0][j] = gap_open + (j-1) * gap_extend

    # 填充DP表
    for i in range(1, m+1):
        for j in range(1, n+1):
            if seq1[i-1] == seq2[j-1]:
                score = match  # 匹配得分
            else:
                score = mismatch  # 不匹配得分
            dp[i][j] = max(
                dp[i-1][j-1] + score,  # 对角线方向
                dp[i-1][j] + (gap_open if gap_matrix[i-1][j] == 0 else gap_extend),  # 上方向
                dp[i][j-1] + (gap_open if gap_matrix[i][j-1] == 0 else gap_extend)  # 左方向
            )
            # 更新gap矩阵
            if dp[i][j] == dp[i-1][j] + (gap_open if gap_matrix[i-1][j] == 0 else gap_extend):
                gap_matrix[i][j] = 1
            elif dp[i][j] == dp[i][j-1] + (gap_open if gap_matrix[i][j-1] == 0 else gap_extend):
                gap_matrix[i][j] = 1
            else:
                gap_matrix[i][j] = 0

    # 回溯找到比对路径
    align1, align2 = [], []
    i, j = m, n
    while i > 0 and j > 0:
        if dp[i][j] == dp[i-1][j-1] + (match if seq1[i-1] == seq2[j-1] else mismatch):
            align1.append(seq1[i-1])
            align2.append(seq2[j-1])
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i-1][j] + (gap_open if gap_matrix[i-1][j] == 0 else gap_extend):
            align1.append(seq1[i-1])
            align2.append('-')
            i -= 1
        else:
            align1.append('-')
            align2.append(seq2[j-1])
            j -= 1

    while i > 0:
        align1.append(seq1[i-1])
        align2.append('-')
        i -= 1
    while j > 0:
        align1.append('-')
        align2.append(seq2[j-1])
        j -= 1

    return ''.join(align1[::-1]), ''.join(align2[::-1]), dp[m][n]

# DP比对示例
seq1 = "ATCG"
seq2 = "ACG"
align1, align2, score = dp_alignment(seq1, seq2)
print("DP Alignment:")
print(align1)
print(align2)
print("Score:", score)


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
        _, _, score = dp_alignment(seq1, seq2)
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
    
with open('./HumanHBA1.txt', 'r') as file:
    Human_sequence = file.read().replace('\n', '')
with open('./CSHBA1.txt', 'r') as file:
    CS_sequence = file.read().replace('\n', '')
with open('./MouseHBA1.txt', 'r') as file:
    Mouse_sequence = file.read().replace('\n', '')
with open('./NorwayRatHBA1.txt', 'r') as file:
    NorwayRat_sequence = file.read().replace('\n', '')

sequences = [Human_sequence, CS_sequence, Mouse_sequence, NorwayRat_sequence]
upgma = UPGMA(sequences)
print("UPGMA Cluster Tree:", upgma.cluster_tree)
upgma.plot_tree()
