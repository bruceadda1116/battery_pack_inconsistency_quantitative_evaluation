# 全局单图综合不一致性评价 - (内存常驻 + 群智回捞)
import os
import time
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import euclidean_distances
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==================== 0. 全局配置区 ====================
GLOBAL_SIGMA = 0.4617
WEIGHTS = {'Z_R0': 0.358, 'Z_Drop': 0.305, 'Z_DTW': 0.337}


# ==================== 1. 核心算法模块 ====================
def otsu_1d_threshold(scores):
    sorted_scores = np.sort(scores)
    N = len(scores)
    best_threshold, best_variance = 0, -1
    for i in range(1, N):
        t = (sorted_scores[i - 1] + sorted_scores[i]) / 2.0
        w0, w1 = i / N, (N - i) / N
        mu0, mu1 = np.mean(sorted_scores[:i]), np.mean(sorted_scores[i:])
        variance = w0 * w1 * (mu0 - mu1) ** 2
        if variance > best_variance:
            best_variance, best_threshold = variance, t
    return best_threshold


def process_global_graph(df_coords):
    cell_ids = df_coords.index.values
    N = len(df_coords)
    Z_vals = df_coords[['Z_R0', 'Z_Drop', 'Z_DTW']].values
    features_weighted = Z_vals * np.sqrt([WEIGHTS['Z_R0'], WEIGHTS['Z_Drop'], WEIGHTS['Z_DTW']])

    # 1. 距离矩阵与全连通高斯权重
    D = euclidean_distances(features_weighted)
    W_full = np.exp(-(D ** 2) / (2 * GLOBAL_SIGMA ** 2))
    W_upper = W_full[np.triu_indices(N, k=1)]
    cci_1_global = np.mean(W_upper) if len(W_upper) > 0 else 1.0

    # 2. 混合拓扑构建 (MST + M-KNN)
    K = min(max(5, int(0.05 * N)), N - 1)
    knn_mat = np.zeros((N, N), dtype=bool)
    for i in range(N):
        knn_mat[i, np.argsort(D[i])[1:K + 1]] = True
    mknn_mat = knn_mat & knn_mat.T

    G_dist = nx.from_numpy_array(D)
    mst_edges = nx.minimum_spanning_tree(G_dist, weight='weight').edges()
    mst_mat = np.zeros((N, N), dtype=bool)
    for u, v in mst_edges: mst_mat[u, v] = mst_mat[v, u] = True

    topology_mat = mknn_mat | mst_mat
    W_topology = np.zeros((N, N))
    W_topology[topology_mat] = W_full[topology_mat]

    # 3. PageRank 中心度计算
    G = nx.from_numpy_array(W_topology)
    try:
        pr_dict = nx.pagerank(G, alpha=0.95, weight='weight', max_iter=1000)
        PR_scores = np.array([pr_dict[i] for i in range(N)])
    except Exception:  # 修复：避免使用裸 except 吞噬系统级中断
        PR_scores = np.ones(N) / N

    # 4. 极严 CV 门卫切割与 Otsu 分类
    CV = np.std(PR_scores) / np.mean(PR_scores) if np.mean(PR_scores) > 0 else 0
    if CV < 0.1:
        core_indices, outlier_indices = np.arange(N), np.array([], dtype=int)
    else:
        t_star = otsu_1d_threshold(PR_scores)
        core_indices = np.where(PR_scores > t_star)[0]
        outlier_indices = np.where(PR_scores <= t_star)[0]

    # 5. LCC 最大连通子图提纯
    if len(core_indices) > 0:
        core_sub = nx.Graph(topology_mat[np.ix_(core_indices, core_indices)])
        largest_comp = list(max(nx.connected_components(core_sub), key=len))
        true_core_indices = core_indices[largest_comp]
        outlier_indices = np.setdiff1d(np.arange(N), true_core_indices)
        core_indices = true_core_indices

    # =========================================================================
    # 6. 1σ 群智半数回捞
    # =========================================================================
    if len(core_indices) > 0 and len(outlier_indices) > 0:
        core_features_w = features_weighted[core_indices]
        core_size = len(core_indices)
        pending_core = []
        strictly_outliers = []

        for idx in outlier_indices:
            opt_w = features_weighted[idx].reshape(1, -1)
            dists_to_core_nodes = euclidean_distances(opt_w, core_features_w)[0]
            close_points_count = np.sum(dists_to_core_nodes < GLOBAL_SIGMA)

            if close_points_count >= (core_size / 2.0):
                pending_core.append(idx)
            else:
                strictly_outliers.append(idx)

        if pending_core:
            core_indices = np.concatenate([core_indices, pending_core]).astype(int)
            outlier_indices = np.array(strictly_outliers, dtype=int)

    # 7. 指标测算
    if len(core_indices) > 1:
        W_core = W_full[np.ix_(core_indices, core_indices)]
        cci_2_core = np.mean(W_core[np.triu_indices(len(core_indices), k=1)])
    else:
        cci_2_core = 1.0

    danger_list = []
    if len(outlier_indices) > 0 and len(core_indices) > 0:
        core_feats = features_weighted[core_indices]
        for idx in outlier_indices:
            # 修复：复用已提取的 core_feats 进行距离计算
            dist = np.min(euclidean_distances(features_weighted[idx].reshape(1, -1), core_feats))
            danger_list.append({'id': cell_ids[idx], 'd': dist})

        danger_list.sort(key=lambda x: x['d'], reverse=True)

    return cci_1_global, cci_2_core, len(core_indices), danger_list


# ==================== 2. 单车处理任务引擎 (内存常驻优化版) ====================
def process_single_vehicle(ds_id, folder_path, save_folder):
    """
    独立处理一辆车的函数，专门适配多进程并行调用。
    采用内存常驻策略，一次性载入整车特征数据进行内部切片。
    """
    file_r0 = os.path.join(folder_path, f'Z_R0_{ds_id}.csv')
    file_drop = os.path.join(folder_path, f'Z_Drop_{ds_id}.csv')
    file_dtw = os.path.join(folder_path, f'Z_DTW_{ds_id}.csv')

    # 文件存在性验证
    if not (os.path.exists(file_r0) and os.path.exists(file_drop) and os.path.exists(file_dtw)):
        return ds_id, False, "数据缺失"

    try:
        # 【核心优化】：一次性读盘，常驻内存
        df_r0 = pd.read_csv(file_r0, index_col=0)
        df_drop = pd.read_csv(file_drop, index_col=0)
        df_dtw = pd.read_csv(file_dtw, index_col=0)

        # 提取共同的检查点交集
        checkpoints = [cp for cp in df_r0.index if cp in df_drop.index and cp in df_dtw.index]
        ds_evaluation_results = []

        for cp in checkpoints:
            # 【核心优化】：直接在内存中做切片组装，拒绝反复 I/O
            coord_df = pd.DataFrame({
                'Z_R0': df_r0.loc[cp],
                'Z_Drop': df_drop.loc[cp],
                'Z_DTW': df_dtw.loc[cp]
            }).dropna()

            # 修复：极少样本防御机制，无法构建拓扑图则跳过
            if coord_df.empty or len(coord_df) < 3:
                continue

            c1, c2, n_core, dangers = process_global_graph(coord_df)

            core_pct_val = n_core / len(coord_df)
            outlier_id_set = "[" + ", ".join([str(int(d['id'])) for d in dangers]) + "]" if dangers else "无"
            furthest_outlier_id = int(dangers[0]['id']) if dangers else "无"
            dist_of_furthest_outlier = dangers[0]['d'] if dangers else "无"

            ds_evaluation_results.append({
                '累计里程分层编号': cp,
                'CCI1': c1,
                'CCI2': c2,
                'CCI2-CCI1': c2 - c1,
                '核心群占比': core_pct_val,
                '偏离点电池单元编号集合': outlier_id_set,
                '距核心群的最远的偏离点编号': furthest_outlier_id,
                '最远偏离点距核心群的最近距离': dist_of_furthest_outlier
            })

        # 写入结果
        if ds_evaluation_results:
            df_out = pd.DataFrame(ds_evaluation_results)
            save_path = os.path.join(save_folder, f'eva{ds_id}.csv')
            df_out.to_csv(save_path, index=False, encoding='utf-8-sig')
            return ds_id, True, f"成功 (共处理 {len(checkpoints)} 个检查点)"
        else:
            return ds_id, False, "有效数据为空或跳过"

    except Exception as e:
        return ds_id, False, f"发生异常: {str(e)}"


# ==================== 3. 主程序调度器 ====================
def main():
    folder_path = r'E:\0417ALL'
    save_folder = r'E:\0417eva'

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print(f"创建保存文件夹: {save_folder}")

    if not os.path.exists(folder_path):
        print(f"错误: 找不到数据文件夹 {folder_path}")
        return

    # 扫描收集车辆 ID
    dataset_ids = set()
    for filename in os.listdir(folder_path):
        if filename.startswith('Z_R0_') and filename.endswith('.csv'):
            ds_id = filename.replace('Z_R0_', '').replace('.csv', '')
            dataset_ids.add(ds_id)

    dataset_ids = sorted(list(dataset_ids), key=lambda x: int(x) if x.isdigit() else x)

    if not dataset_ids:
        print(f"警告: 文件夹 {folder_path} 无有效数据。")
        return

    print("=" * 90)
    print(f" MST+M-KNN 评价  - 待处理: {len(dataset_ids)} 辆车")
    print("=" * 90)

    start_time = time.time()
    success_count = 0

    # 启动多进程池 (默认使用机器所有可用核心)
    with ProcessPoolExecutor() as executor:
        # 提交所有车辆的处理任务
        futures = {executor.submit(process_single_vehicle, ds_id, folder_path, save_folder): ds_id for ds_id in
                   dataset_ids}

        # 实时监听多进程完成状态并打印日志
        for future in as_completed(futures):
            ds_id = futures[future]
            try:
                processed_id, success, msg = future.result()
                if success:
                    print(f" [√] 车辆 {processed_id:>3} 处理完毕 | {msg}")
                    success_count += 1
                else:
                    print(f" [!] 车辆 {processed_id:>3} 处理失败 | 原因: {msg}")
            except Exception as e:
                print(f" [X] 车辆 {ds_id:>3} 进程崩溃 | {str(e)}")

    end_time = time.time()

    print("\n" + "=" * 90)
    print(f" 批处理完成！成功生成 {success_count} 份评估报告。")
    print(f" 总耗时: {end_time - start_time:.2f} 秒")
    print("=" * 90)


if __name__ == "__main__":
    # 多进程在 Windows 必须保护在 __main__ 块内执行
    main()