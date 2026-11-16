import os
import re
import glob
import numpy as np
import pandas as pd
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from tqdm import tqdm
from numba import njit

warnings.filterwarnings("ignore")

# ==================== 1. 全局配置常量 ====================
INPUT_DIR = r'E:\1CSV'  # 上一步清洗完毕的数据目录
OUTPUT_DIR = r'E:\1RRRR'  # 最终欧姆内阻存放目录0
NUM_CELLS = 96
L_COL = 12  # L列对应索引11，存放96串电压
I_COL = 6  # F列对应索引5，存放总电流
DT = 10
SKIP_PTS = 10

# [新增] 内阻有效性物理约束经验值
R0_MIN = 0.0003
R0_MAX = 0.0008


# ==================== 2. Numba 极限加速核心算法 ====================
@njit
def identify_R0_core_numba(I, Vt, n):
    delta_1, delta_2, Rr_window = 0.04, 0.04, 10
    lambda_max, lambda_min, e_base, h = 0.995, 0.95, 0.01, 0.5

    theta_VFF = np.zeros((4, n))
    R_inf = np.eye(4)
    R_prev = np.copy(R_inf)

    R0_VFF = np.zeros(n)
    Voc_VFF = np.zeros(n)
    Voc_VFF[0] = Vt[0] + I[0] * 0.0004
    Rr_current = 0.0004

    for k in range(1, n):
        phi = np.array([Vt[k - 1], I[k], I[k - 1], 1.0])

        if k >= 2:
            start_idx = max(1, k - Rr_window + 1)
            dU = Vt[start_idx:k + 1] - Vt[start_idx - 1:k]
            dI = I[start_idx:k + 1] - I[start_idx - 1:k]
            sum_dI2 = np.sum(dI ** 2)
            if sum_dI2 > 1e-8:
                Rr_current = abs(np.sum(dU * dI) / sum_dI2)
                if Rr_current < 0.0001: Rr_current = 0.0001
                if Rr_current > 0.05: Rr_current = 0.05

        y_pred_VFF = 0.0
        for i in range(4):
            y_pred_VFF += theta_VFF[i, k - 1] * phi[i]

        e_VFF = Vt[k] - y_pred_VFF
        alpha_k = min((e_VFF / e_base) ** 2, 50.0)
        lambda_k = lambda_min + (lambda_max - lambda_min) * (h ** alpha_k)

        outer_phi = np.empty((4, 4))
        for i in range(4):
            for j in range(4):
                outer_phi[i, j] = phi[i] * phi[j]

        R_current_mat = lambda_k * R_prev + (1 - lambda_k) * R_inf + outer_phi

        for i in range(4):
            R_current_mat[i, i] += 1e-8

        P_VFF = np.linalg.inv(R_current_mat)

        K_VFF = np.zeros(4)
        for i in range(4):
            for j in range(4):
                K_VFF[i] += P_VFF[i, j] * phi[j]

        for i in range(4):
            theta_VFF[i, k] = theta_VFF[i, k - 1] + K_VFF[i] * e_VFF

        R_prev = np.copy(R_current_mat)
        a1_V, a4_V = theta_VFF[0, k], theta_VFF[3, k]

        if 0.001 < a1_V < 0.999:
            Voc_est_V = a4_V / (1 - a1_V)
        else:
            Voc_est_V = Voc_VFF[k - 1]

        valid_V = False
        if I[k] >= 0:
            if (Vt[k] - delta_1) <= Voc_est_V <= (Voc_VFF[k - 1] + delta_2): valid_V = True
        else: 
            if (Voc_VFF[k - 1] - delta_2) <= Voc_est_V <= (Vt[k] + delta_1): valid_V = True

        if valid_V:
            Voc_VFF[k] = Voc_est_V
        else: # It is also possible to develop algorithms to adaptively adjust the feedback coefficients to achieve better results.
            Voc_Rint = Vt[k] + I[k] * Rr_current
            Voc_VFF[k] = Voc_Rint
            target_a4 = Voc_Rint * (1 - a1_V)
            theta_VFF[3, k] = (1 - 0.2) * theta_VFF[3, k] + 0.2 * target_a4

        b2_V = theta_VFF[2, k]
        if 0.001 < a1_V < 0.999:
            R0_VFF[k] = abs(b2_V / a1_V)
        else:
            R0_VFF[k] = R0_VFF[k - 1]

    return R0_VFF


@njit
def fast_density_peak_numba(data, bins=100):
    hist, bin_edges = np.histogram(data, bins=bins)
    smoothed = np.zeros(bins)
    for i in range(bins):
        val = hist[i]
        if i > 0: val += hist[i - 1]
        if i < bins - 1: val += hist[i + 1]
        smoothed[i] = val
    max_idx = np.argmax(smoothed)
    return (bin_edges[max_idx] + bin_edges[max_idx + 1]) / 2.0


def extract_best_param_fast(param_array, skip_pts=10):
    if len(param_array) <= skip_pts: return np.nan
    valid_data = param_array[skip_pts:]
    lower_bound = np.percentile(valid_data, 1)
    upper_bound = np.percentile(valid_data, 99)
    filtered_data = valid_data[(valid_data >= lower_bound) & (valid_data <= upper_bound)]

    if len(filtered_data) < 2 or np.std(filtered_data) < 1e-10:
        return np.mean(filtered_data) if len(filtered_data) > 0 else np.nan

    return fast_density_peak_numba(filtered_data, bins=100)


# ==================== 3. 数据解析及聚合 ====================
def fast_parse_voltage_matrix(raw_series, num_cells=96):
    raw_list = raw_series.dropna().astype(str).tolist()
    matrix = []

    for row in raw_list:
        parts = row.split('~')
        row_data = [float(parts[i]) if i < len(parts) and parts[i].strip() else np.nan for i in range(num_cells)]
        matrix.append(row_data)

    return np.array(matrix, dtype=np.float64)


# ==================== 4. 单文件工作节点 ====================
def process_single_file(task_info):
    layer_name, file_path = task_info
    current_file_R0s = []

    try:
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            raise ValueError("文件不存在或为空")

        header_check = pd.read_csv(file_path, nrows=0, on_bad_lines='skip', engine='c')
        if len(header_check.columns) <= max(I_COL, L_COL - 1):
            raise ValueError(f"数据列数({len(header_check.columns)})不足以提取目标列")

        try:
            df = pd.read_csv(file_path, on_bad_lines='skip', engine='c', usecols=[I_COL, L_COL - 1])
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, on_bad_lines='skip', engine='c', usecols=[I_COL, L_COL - 1], encoding='gbk')

        I_raw = np.array(df.iloc[:, 0])
        V_matrix = fast_parse_voltage_matrix(df.iloc[:, 1], NUM_CELLS)

        min_len = min(len(I_raw), len(V_matrix))
        if min_len < 50:
            raise ValueError(f"有效数据行数({min_len})不足50行")

        I = I_raw[:min_len]
        V_matrix = V_matrix[:min_len, :]
        if np.isnan(I).all():
            raise ValueError("电流全为缺失值")

        valid_cell_count = 0

        for cell_idx in range(NUM_CELLS):
            Vt = V_matrix[:, cell_idx]

            if np.isnan(Vt).sum() > len(Vt) * 0.5:
                r0 = np.nan
            else:
                try:
                    Vt_filled = pd.Series(Vt).ffill().bfill().values
                    if np.isnan(Vt_filled).all():
                        r0 = np.nan
                    else:
                        r0_array = identify_R0_core_numba(I, Vt_filled, min_len)
                        r0 = extract_best_param_fast(r0_array, SKIP_PTS)

                        # [核心修改 1] 强制约束欧姆内阻至用户指定范围
                        if not np.isnan(r0) and (r0 < R0_MIN or r0 > R0_MAX):
                            r0 = np.nan
                except Exception:
                    r0 = np.nan

            if not np.isnan(r0):
                valid_cell_count += 1
            current_file_R0s.append(r0)

        if valid_cell_count == 0:
            raise ValueError(f"所有电芯辨识失败 (均不满足 [{R0_MIN}, {R0_MAX}] 的范围约束)")

        return (layer_name, current_file_R0s, None)

    except Exception as e:
        return (layer_name, None, f"异常 {os.path.basename(file_path)} - {e}")


# ==================== 5. 车辆生命周期处理模块 ====================
def process_vehicle_folder(vehicle_dir, output_file, car_id):
    tasks = []

    # 遍历车辆文件夹下的所有纯数字命名的里程层级文件夹 (例如 '1', '16', '22')
    for layer_folder in os.listdir(vehicle_dir):
        layer_path = os.path.join(vehicle_dir, layer_folder)

        # 判断是否是合法纯数字文件夹
        if not os.path.isdir(layer_path) or not layer_folder.isdigit():
            continue

        # [核心修改 2] 检查文件夹下是否同时存在 charge*.csv 与 discharge*.csv
        charge_files = glob.glob(os.path.join(layer_path, 'charge*.csv'))
        discharge_files = glob.glob(os.path.join(layer_path, 'discharge*.csv'))

        if charge_files and discharge_files:
            # 只有当充电和放电文件同时存在时，才提取所有的放电文件加入任务列表
            for d_file in discharge_files:
                tasks.append((layer_folder, d_file))

    if not tasks:
        print(f"   车 {car_id} 无符合条件的充放电成对数据，跳过。")
        return

    print(f"   车 {car_id}: 发现 {len(tasks)} 个放电片段，开始特征提取...")
    grouped_results = defaultdict(list)
    error_logs = []

    # 性能适配 i5-13400: 多进程运算核心控制
    max_cores = max(4, os.cpu_count() - 4)

    with ProcessPoolExecutor(max_workers=max_cores) as executor:
        futures = {executor.submit(process_single_file, task): task for task in tasks}
        with tqdm(total=len(tasks), desc=f"计算进度", unit="段", colour="blue") as pbar:
            for future in as_completed(futures):
                layer_name, R0_list, err = future.result()
                if err:
                    error_logs.append(err)
                elif R0_list is not None:
                    # 归类：将属于同一个层级（layer）的内阻结果存放到一起
                    grouped_results[int(layer_name)].append(R0_list)
                pbar.update(1)

    if error_logs:
        print(f"   跳过 {len(error_logs)} 个片段 (数据异常/全被范围约束剔除)")

    # [核心修改 3] 聚合算法：对同一层级的多个放电片段提取出的有效 R0 取中位数
    final_results = {}
    for layer_int, r0_matrices in grouped_results.items():
        if r0_matrices:
            mat = np.array(r0_matrices, dtype=float)

            # 使用 np.nanmedian 忽略 NaN 计算中位数，压制全 NaN 列引起的警告
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                medians = np.nanmedian(mat, axis=0)

            final_results[layer_int] = medians

    if not final_results:
        print(f"   车 {car_id} 计算完毕，但没有符合区间要求的特征值。")
        return

    # 生成最终 DataFrame 报告，第一列为 index (里程层级，从小到大排序)
    df_output = pd.DataFrame.from_dict(final_results, orient='index', columns=list(range(1, NUM_CELLS + 1)))
    df_output.sort_index(inplace=True)
    df_output.index.name = 'CheckPoint'

    df_output.to_csv(output_file, encoding='utf-8-sig')
    print(f"   车 {car_id} 特征矩阵已保存至: {output_file}\n")


# ==================== 6. 主控入口 ====================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print(" 动力电池全生命周期欧姆内阻 (R0) 批量提取引擎启动")
    print(f"设定参数 | 数据源: {INPUT_DIR} | 输出: {OUTPUT_DIR}")
    print(f"经验约束 | R0 限定有效范围: [{R0_MIN}, {R0_MAX}] 欧姆")
    print("=" * 60)

    if not os.path.exists(INPUT_DIR):
        print(f" 找不到数据源目录: {INPUT_DIR}")
        return

    # 动态扫描 INPUT_DIR 下所有的 Lifecycle_Dataxx 文件夹
    vehicle_folders = [f for f in os.listdir(INPUT_DIR) if
                       os.path.isdir(os.path.join(INPUT_DIR, f)) and f.startswith('Lifecycle_Data')]

    # 按照车辆编号进行数字排序 (例如 Data2 应该排在 Data10 前面)
    def extract_num(name):
        nums = re.findall(r'\d+', name)
        return int(nums[0]) if nums else 0

    vehicle_folders.sort(key=extract_num)

    if not vehicle_folders:
        print(" 目录下没有找到 Lifecycle_Data 格式的车辆文件夹。")
        return

    for idx, folder_name in enumerate(vehicle_folders, 1):
        car_id = extract_num(folder_name)
        vehicle_dir = os.path.join(INPUT_DIR, folder_name)
        output_csv = os.path.join(OUTPUT_DIR, f'R0_{car_id}.csv')

        print(f" [{idx}/{len(vehicle_folders)}] 正在分析车辆 {car_id} 的生命周期数据...")
        process_vehicle_folder(vehicle_dir, output_csv, car_id)

    print("=" * 60)
    print(" 所有车辆的特征提取任务圆满完成！")


if __name__ == "__main__":
    main()