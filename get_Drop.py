import os
import re
import glob
import numpy as np
import pandas as pd
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from tqdm import tqdm

# 忽略全 NaN 数组求中位数时产生的警告
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==================== 1. 全局配置区域 ====================
INPUT_DIR = r'E:\1CSV'  # 数据分割与清洗后的输入目录
OUTPUT_DIR = r'E:\1DropDrop'  # Drop 特征最终输出目录
NUM_CELLS = 96  # 电池单元数量


# ==================== 2. 辅助与解析函数 ====================
def fast_parse_voltage_matrix(df):
    """
    极速且高鲁棒的电压矩阵解析 (抛弃慢速 for 循环，引入 C 引擎向量化)
    """
    target_col = None

    # 智能寻找包含波浪号 '~' 的列
    for col in df.columns:
        if len(df) > 0 and pd.notna(df.iloc[0][col]):
            sample = str(df.iloc[0][col])
            if '~' in sample and len(sample) > 5:
                target_col = col
                break

    if target_col is None:
        return None

    try:
        # 极速切分与格式转换
        expanded = df[target_col].astype(str).str.split('~', expand=True)
        cols_to_take = min(NUM_CELLS, expanded.shape[1])
        v_matrix = expanded.iloc[:, :cols_to_take].apply(pd.to_numeric, errors='coerce').values

        # 不足 96 列用 NaN 补齐
        if cols_to_take < NUM_CELLS:
            pad = np.full((v_matrix.shape[0], NUM_CELLS - cols_to_take), np.nan)
            v_matrix = np.hstack((v_matrix, pad))

        return v_matrix
    except Exception:
        return None


# ==================== 3. 并发工作节点 (Worker) ====================
def process_single_file(task_info):
    """独立处理单个充电文件的任务函数"""
    layer_int, file_path = task_info
    try:
        # 鲁棒性1：防空文件/损坏文件
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return (layer_int, None, "文件不存在或大小为0")

        # 使用 C 引擎与跳过坏行机制
        df = pd.read_csv(file_path, on_bad_lines='skip', engine='c')

        if df.empty:
            return (layer_int, None, "数据表为空")

        v_matrix = fast_parse_voltage_matrix(df)

        # 鲁棒性2：保证数据行数足够支持始末电压提取
        if v_matrix is not None and v_matrix.shape[0] > 1:
            v_start = v_matrix[0, :]
            v_end = v_matrix[-1, :]

            # 核心业务逻辑：计算绝对差值 |V_start - V_end|
            v_diff = np.abs(v_start - v_end)

            # 鲁棒性3：检查是否全部电芯数据均失效
            if np.all(np.isnan(v_diff)):
                return (layer_int, None, "所有电芯的首尾数据均缺失(NaN)")

            return (layer_int, v_diff, None)
        else:
            return (layer_int, None, "未找到有效的电压列或数据不足2行")

    except Exception as e:
        return (layer_int, None, f"解析异常: {str(e)}")


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
            # 只有同时存在时，才提取所有的【充电片段】送去算 Drop (压升)
            for c_file in charge_files:
                tasks.append((int(layer_folder), c_file))

    if not tasks:
        print(f"   车 {car_id} 无符合条件的充放电共存数据，跳过。")
        return

    print(f"  车 {car_id}: 发现 {len(tasks)} 个有效充电片段，并发计算中...")
    grouped_results = defaultdict(list)
    error_logs = []

    # 启动进程池
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_file, task): task for task in tasks}

        with tqdm(total=len(tasks), desc=f" 车 {car_id} 压升计算", unit="文件", colour="cyan") as pbar:
            for future in as_completed(futures):
                layer_int, v_diff, err = future.result()
                if err:
                    error_logs.append(err)
                elif v_diff is not None:
                    grouped_results[layer_int].append(v_diff)
                pbar.update(1)

    # 打印被成功跳过的异常日志
    if error_logs:
        print(f"   因异常数据跳过 {len(error_logs)} 个片段。")

    final_results = {}

    # [核心修改]：如果有多个有效值，取中位数 (nanmedian)
    for layer_int, diff_matrices in grouped_results.items():
        if diff_matrices:
            mat = np.array(diff_matrices, dtype=float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                medians = np.nanmedian(mat, axis=0)
            final_results[layer_int] = medians

    if not final_results:
        print(f"   车 {car_id} 计算完毕，但无有效结果生成。")
        return

    # 导出为 CSV，按里程层级(CheckPoint)升序排序
    columns = list(range(1, NUM_CELLS + 1))
    df_output = pd.DataFrame.from_dict(final_results, orient='index', columns=columns)
    df_output.sort_index(inplace=True)
    df_output.index.name = 'CheckPoint'

    df_output.to_csv(output_file, encoding='utf-8-sig')
    print(f"  车 {car_id} Drop(压升)矩阵已保存至: {output_file}\n")


# ==================== 5. 主控系统 ====================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 针对 i5-13400 + 16G 算力优化
    total_cores = os.cpu_count() or 4
    allowed_workers = min(8, max(1, total_cores - 4))

    print("=" * 60)
    print(f" 动力电池充电压升 (Drop) 批量特征提取启动")
    print(f"设定参数 | 数据源: {INPUT_DIR} | 输出: {OUTPUT_DIR}")
    print(f"提取逻辑 | 充放电同在 -> 提取充电曲线 -> |V_start - V_end| -> 取中位数")
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
        output_csv = os.path.join(OUTPUT_DIR, f'Drop_{car_id}.csv')

        print(f" [{idx}/{len(vehicle_folders)}] 正在分析车辆 {car_id} 的生命周期数据...")
        process_vehicle_folder(vehicle_dir, output_csv, car_id, allowed_workers)

    print("=" * 60)
    print(" 所有车辆的 Drop 特征提取任务圆满完成！")


if __name__ == "__main__":
    main()