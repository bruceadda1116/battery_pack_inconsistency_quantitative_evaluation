import pandas as pd
import numpy as np
import os
import time
import glob
import re
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed


# ==================== 1. 核心读取与预处理 ====================
def read_csv_smart(file_path):
    try:
        header_row = 0
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for i in range(50):
                    if 'terminaltime' in f.readline().lower():
                        header_row = i
                        break
        except:
            pass

        try:
            df = pd.read_csv(file_path, skiprows=header_row, encoding='utf-8', on_bad_lines='skip', low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, skiprows=header_row, encoding='gbk', on_bad_lines='skip', low_memory=False)
        except TypeError:
            df = pd.read_csv(file_path, skiprows=header_row, encoding='gbk', error_bad_lines=False, low_memory=False)

        df.columns = [str(c).strip().lower() for c in df.columns]

        # 硬件优化：向下转换数据类型
        for col in ['soc', 'totalodometer', 'chargestatus']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')

        return df
    except Exception:
        return None


# ==================== 2. SOC 截取算法 ====================
def cut_charge_segment(soc_values, target_start=40, target_end=80):
    start_mask = soc_values >= target_start
    if not start_mask.any(): return None
    rel_start_idx = start_mask.argmax()

    end_mask = soc_values >= target_end
    if not end_mask.any(): return None
    rel_end_idx = end_mask.argmax()

    if rel_start_idx > rel_end_idx: return None
    return rel_start_idx, rel_end_idx


def cut_discharge_segment(soc_values, target_start=80, target_end=40):
    start_mask = soc_values <= target_start
    if not start_mask.any(): return None
    rel_start_idx = start_mask.argmax()

    end_mask = soc_values <= target_end
    if not end_mask.any(): return None
    rel_end_idx = end_mask.argmax()

    if rel_start_idx > rel_end_idx: return None
    return rel_start_idx, rel_end_idx


# ==================== 3. 内存清洗模块 ====================
def clean_dataframe_in_memory(df):
    if df.shape[1] <= 11:
        return pd.DataFrame()

    col_11 = df.iloc[:, 11]
    str_col = col_11.astype(str).str.strip().str.lower()
    valid_mask = col_11.notna() & (str_col != "") & (str_col != "nan")

    return df[valid_mask]


# ==================== 4. 单车数据处理 ====================
def process_single_vehicle(file_path, base_output_dir):
    try:
        filename = os.path.basename(file_path)
        match = re.search(r'vin(\d+)\.csv', filename, re.IGNORECASE)
        if not match: return None
        car_id = match.group(1)

        target_dir = os.path.join(base_output_dir, f'Lifecycle_Data{car_id}')

        df = read_csv_smart(file_path)
        if df is None or df.empty: return None

        req_cols = ['terminaltime', 'chargestatus', 'soc', 'totalodometer']
        if not all(col in df.columns for col in req_cols): return None

        df['soc'] = df['soc'].ffill().bfill()
        df = df.sort_values('terminaltime').reset_index(drop=True)

        conditions = [(df['chargestatus'] == 1), (df['chargestatus'] == 3)]
        df['state'] = np.select(conditions, ['Charge', 'Discharge'], default='Rest')
        df['segment_id'] = (df['state'] != df['state'].shift()).cumsum()

        # [逻辑修改 1]：先创建列表，把清洗合格的片段全部收集起来
        valid_charges = []
        valid_discharges = []

        # --- 第一阶段：提取并在内存中清洗 ---
        for seg_id, group in df.groupby('segment_id'):
            state = group['state'].iat[0]
            if state not in ['Charge', 'Discharge'] or group['soc'].isnull().all():
                continue

            soc_vals = group['soc'].values
            start_idx = group.index[0]

            if state == 'Charge' and soc_vals[0] <= 40 and soc_vals[-1] >= 80:
                cut_res = cut_charge_segment(soc_vals, 40, 80)
                if cut_res:
                    s_rel, e_rel = cut_res
                    real_start, real_end = start_idx + s_rel, start_idx + e_rel

                    # 抓取截取片段最后一行的里程
                    end_odo = df.at[real_end, 'totalodometer']

                    # 切割并在内存中清洗
                    raw_seg = df.iloc[real_start:real_end + 1]
                    clean_seg = clean_dataframe_in_memory(raw_seg)

                    # 如果清洗后没变成空壳，则录入有效列表
                    if not clean_seg.empty:
                        valid_charges.append((clean_seg, end_odo))

            elif state == 'Discharge' and soc_vals[0] >= 80 and soc_vals[-1] <= 40:
                cut_res = cut_discharge_segment(soc_vals, 80, 40)
                if cut_res:
                    s_rel, e_rel = cut_res
                    real_start, real_end = start_idx + s_rel, start_idx + e_rel

                    # 抓取截取片段最后一行的里程
                    end_odo = df.at[real_end, 'totalodometer']

                    raw_seg = df.iloc[real_start:real_end + 1]
                    clean_seg = clean_dataframe_in_memory(raw_seg)

                    if not clean_seg.empty:
                        valid_discharges.append((clean_seg, end_odo))

        # 拦截：如果全车连一个有效片段都没有，直接返回
        if not valid_charges and not valid_discharges:
            return None

        # --- 第二阶段：基于里程终点计算层级并分装 ---
        layer_data = defaultdict(lambda: {'charge': [], 'discharge': []})

        # [逻辑修改 2]：向上取整的层级计算函数
        def get_layer(odo_val):
            if pd.isna(odo_val) or odo_val <= 0:
                return 1  # 缺失或0公里归入第1层
            return max(1, math.ceil(odo_val / 1000.0))

        for c_df, odo in valid_charges:
            layer = get_layer(odo)
            layer_data[layer]['charge'].append(c_df)

        for d_df, odo in valid_discharges:
            layer = get_layer(odo)
            layer_data[layer]['discharge'].append(d_df)

        # --- 第三阶段：落地保存 ---
        c_count, d_count = 0, 0
        for layer, data in layer_data.items():
            layer_dir = os.path.join(target_dir, str(layer))
            layer_dir_created = False

            for idx, c_df in enumerate(data['charge'], start=1):
                if not layer_dir_created:
                    os.makedirs(layer_dir, exist_ok=True)
                    layer_dir_created = True
                c_df.to_csv(os.path.join(layer_dir, f'charge{idx:02d}.csv'), index=False, encoding='utf-8-sig')
                c_count += 1

            for idx, d_df in enumerate(data['discharge'], start=1):
                if not layer_dir_created:
                    os.makedirs(layer_dir, exist_ok=True)
                    layer_dir_created = True
                d_df.to_csv(os.path.join(layer_dir, f'discharge{idx:02d}.csv'), index=False, encoding='utf-8-sig')
                d_count += 1

        return f"车 {car_id}: 最终保留有效 充电 {c_count} 个, 放电 {d_count} 个"

    except Exception as e:
        return f"车 {os.path.basename(file_path)} 异常: {str(e)}"


# ==================== 5. 主控台 ====================
if __name__ == "__main__":
    t_start = time.time()

    INPUT_DIR = r'F:\excle'
    OUTPUT_DIR = r'E:\1CSV'

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_csv_files = glob.glob(os.path.join(INPUT_DIR, 'vin*.csv'))

    if not all_csv_files:
        print(f"在 {INPUT_DIR} 目录下未找到任何格式为 vin*.csv 的文件！")
        exit()

    print(f" 严格匹配到 {len(all_csv_files)} 个源数据文件。")
    print(f" 启动 [精准挑选 -> 取整分层 -> 自动清洗] 并发引擎...")

    max_workers = min(6, os.cpu_count() - 2)

    success_count = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_vehicle, file_path, OUTPUT_DIR): file_path for file_path in
                   all_csv_files}

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result:
                print(f"[{i}/{len(all_csv_files)}]  {result}")
                success_count += 1
            else:
                print(f"[{i}/{len(all_csv_files)}]  未提取到有效片段跳过。")

    print(f"\n{'=' * 40}")
    print(f" 任务完成，成功处理并清洗了 {success_count} 个文件的数据。")
    print(f" 总耗时: {(time.time() - t_start) / 60:.2f} 分钟")