import os
import re
import glob
import numpy as np
import pandas as pd
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from tqdm import tqdm
from fastdtw import fastdtw

# 忽略全 NaN 切片求中位数时产生的警告
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==================== 1. 全局配置区域 ====================
INPUT_DIR = r'E:\1CSV'  # 数据分割与清洗后的输入目录
OUTPUT_DIR = r'E:\1DTWDTW'  # DTW 特征最终输出目录
NUM_CELLS = 96  # 电池单元数量


# ==================== 2. 辅助与解析函数 ====================
def fast_parse_voltage_matrix(df):
    """
    极速解析电压矩阵（智能寻找包含波浪号 '~' 的列）
    """
    target_col = None

    for col in df.columns:
        if len(df) > 0 and pd.notna(df.iloc[0][col]):
            sample = str(df.iloc[0][col])
            if '~' in sample and len(sample) > 5:
                target_col = col
                break

    if target_col is None:
        return None

    try:
        # 极速切分字符串
        expanded = df[target_col].astype(str).str.split('~', expand=True)
        cols_to_take = min(NUM_CELLS, expanded.shape[1])

        # 转换为数值，无法转换的变为 NaN
        v_matrix = expanded.iloc[:, :cols_to_take].apply(pd.to_numeric, errors='coerce').values

        # 如果列数不足 96，用 NaN 补齐
        if cols_to_take < NUM_CELLS:
            pad = np.full((v_matrix.shape[0], NUM_CELLS - cols_to_take), np.nan)
            v_matrix = np.hstack((v_matrix, pad))

        return v_matrix
    except Exception:
        return None


def calculate_dtw_distances(voltage_matrix):
    """计算每个单体曲线与全局【中位数曲线】的 DTW 距离"""
    distances = []

    # 基准曲线：取所有电芯的中位数，抗干扰能力强
    ref_curve = np.nanmedian(voltage_matrix, axis=1)

    valid_ref_mask = ~np.isnan(ref_curve)
    ref_curve_clean = ref_curve[valid_ref_mask]

    # 鲁棒性：基准曲线有效点过少，直接返回全 NaN
    if len(ref_curve_clean) < 10:
        return np.full(NUM_CELLS, np.nan)

    for i in range(NUM_CELLS):
        cell_curve = voltage_matrix[:, i]
        valid_mask = ~np.isnan(cell_curve)
        cell_curve_clean = cell_curve[valid_mask]

        # 如果某个单体由于传感器故障完全没数据，直接赋值 NaN，跳过计算
        if len(cell_curve_clean) > 10:
            try:
                # 计算 DTW 距离
                dtw_dist, _ = fastdtw(cell_curve_clean, ref_curve_clean)
            except Exception:
                dtw_dist = np.nan
        else:
            dtw_dist = np.nan

        distances.append(dtw_dist)

    return np.array(distances)


# ==================== 3. 并发工作节点 (Worker) ====================
def process_single_file(task_info):
    """独立处理单个充电 csv 文件的任务函数"""
    layer_int, file_path = task_info
    try:
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return (layer_int, None, "文件不存在或大小为0")

        # 极速读取
        df = pd.read_csv(file_path, on_bad_lines='skip', engine='c')

        if df.empty:
            return (layer_int, None, "数据表为空")

        v_matrix = fast_parse_voltage_matrix(df)

        if v_matrix is not None and v_matrix.shape[0] > 10:
            distances = calculate_dtw_distances(v_matrix)

            if np.all(np.isnan(distances)):
                return (layer_int, None, "该表所有电芯的DTW均计算失败(数据缺失)")

            return (layer_int, distances, None)
        else:
            return (layer_int, None, "未找到有效的波浪号电压列或行数过少")

    except Exception as e:
        return (layer_int, None, str(e))


# ==================== 4. 单车文件夹调度逻辑 ====================
def process_vehicle_folder(vehicle_dir, output_file, car_id, max_workers):
    tasks = []

    # 遍历车辆文件夹下所有的纯数字里程层级文件夹 (例如 '1', '15')
    for layer_folder in os.listdir(vehicle_dir):
        layer_path = os.path.join(vehicle_dir, layer_folder)

        # 判断是否是合法纯数字文件夹
        if not os.path.isdir(layer_path) or not layer_folder.isdigit():
            continue

        # 检查该层级下是否【同时】存在充放电片段
        charge_files = glob.glob(os.path.join(layer_path, 'charge*.csv'))
        discharge_files = glob.glob(os.path.join(layer_path, 'discharge*.csv'))

        if charge_files and discharge_files:
            # 只有同时存在时，才提取所有的【充电片段】送去算 DTW
            for c_file in charge_files:
                tasks.append((int(layer_folder), c_file))

    if not tasks:
        print(f"   车 {car_id} 无符合条件的充放电共存数据，跳过。")
        return

    print(f"   车 {car_id}: 发现 {len(tasks)} 个有效充电片段，并发计算中...")
    grouped_results = defaultdict(list)
    error_logs = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_file, task): task for task in tasks}

        with tqdm(total=len(tasks), desc=f" 车 {car_id} 计算进度", unit="文件", colour="blue") as pbar:
            for future in as_completed(futures):
                layer_int, distances, err = future.result()
                if err:
                    error_logs.append(err)
                elif distances is not None:
                    grouped_results[layer_int].append(distances)
                pbar.update(1)

    if error_logs:
        print(f"   因异常数据跳过 {len(error_logs)} 个片段。")

    final_results = {}

    # [核心修改]：如果有多个有效值，取中位数 (nanmedian)
    for layer_int, dist_matrices in grouped_results.items():
        if dist_matrices:
            mat = np.array(dist_matrices, dtype=float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                medians = np.nanmedian(mat, axis=0)
            final_results[layer_int] = medians

    if not final_results:
        print(f"   车 {car_id} 计算完毕，但无有效结果生成。")
        return

    # 生成最终 DataFrame 报告，按里程层级(CheckPoint)升序排序
    columns = list(range(1, NUM_CELLS + 1))
    df_output = pd.DataFrame.from_dict(final_results, orient='index', columns=columns)
    df_output.sort_index(inplace=True)
    df_output.index.name = 'CheckPoint'

    df_output.to_csv(output_file, encoding='utf-8-sig')
    print(f"   车 {car_id} DTW 矩阵已保存至: {output_file}\n")


# ==================== 5. 主控系统 ====================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 针对 i5-13400 算力的优化 (DTW 是 CPU 密集型任务，不要完全占满核心)
    total_cores = os.cpu_count() or 4
    allowed_workers = min(8, max(1, total_cores - 4))

    print("=" * 60)
    print(f" 动力电池 DTW 距离批量特征提取启动")
    print(f"设定参数 | 数据源: {INPUT_DIR} | 输出: {OUTPUT_DIR}")
    print(f"提取逻辑 | 充放电同在 -> 提取充电曲线 -> 计算 DTW -> 取中位数")
    print(f"硬件策略 | 检测到 {total_cores} 核心，分配 {allowed_workers} 个高负载工作进程。")
    print("=" * 60)

    if not os.path.exists(INPUT_DIR):
        print(f" 找不到数据源目录: {INPUT_DIR}")
        return

    # 动态扫描 INPUT_DIR 下所有的 Lifecycle_Dataxx 文件夹
    vehicle_folders = [f for f in os.listdir(INPUT_DIR) if
                       os.path.isdir(os.path.join(INPUT_DIR, f)) and f.startswith('Lifecycle_Data')]

    # 按照车辆编号进行数字排序
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
        output_csv = os.path.join(OUTPUT_DIR, f'DTW_{car_id}.csv')

        print(f" [{idx}/{len(vehicle_folders)}] 正在分析车辆 {car_id} 的生命周期数据...")
        process_vehicle_folder(vehicle_dir, output_csv, car_id, allowed_workers)

    print("=" * 60)
    print(" 所有车辆的 DTW 特征提取任务圆满完成！")


if __name__ == "__main__":
    main()