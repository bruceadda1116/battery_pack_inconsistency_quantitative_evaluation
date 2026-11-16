import os
import glob
import pandas as pd


def batch_standardize_features():
    # ================= 配置区域 =================
    # 设置文件夹路径
    folder_path = r'E:\BASE'

    # 将你计算好的全局参数硬编码存入字典，方便集中管理
    # 结构: '特征前缀': {'median': 中位数, 'range80': 80%极差}
    global_params = {
        'R0': {'median': 0.00044, 'range80': 0.0002},
        'Drop': {'median': 0.291, 'range80': 0.077},
        'DTW': {'median': 0.158, 'range80': 0.371}
    }

    # ================= 处理逻辑 =================
    print(f"开始扫描文件夹: {folder_path}\n")

    # 遍历字典中的三种特征类型
    for feature_name, params in global_params.items():
        # 匹配对应特征的所有文件，例如 R0_*.csv
        file_pattern = os.path.join(folder_path, f'{feature_name}_*.csv')
        files = glob.glob(file_pattern)

        # 过滤掉可能已经处理过并带有 Z_ 前缀的文件（防止重复运行时的干扰）
        files_to_process = [f for f in files if not os.path.basename(f).startswith('Z_')]

        if not files_to_process:
            print(f"未找到需要处理的 {feature_name} 特征表格。")
            continue

        print(f"--- 正在处理 {feature_name} 特征 (共 {len(files_to_process)} 个文件) ---")

        for file_path in files_to_process:
            file_name = os.path.basename(file_path)
            # 构造输出文件名：Z_原文件名
            output_name = f"Z_{file_name}"
            output_path = os.path.join(folder_path, output_name)

            try:
                # 1. 读取原始数据
                df = pd.read_csv(file_path)

                # 鲁棒性检查
                if df.empty or df.shape[1] < 2:
                    print(f"  [跳过] {file_name}: 表格为空或缺乏有效数据列。")
                    continue

                # 2. 分离标签列与特征数据列
                # 第一列是检查点编号 (Lxx)，单独提取出来
                label_col = df.iloc[:, 0]
                # 后面的列全是电池的特征数值
                feature_data = df.iloc[:, 1:]

                # 3. 核心计算：Range80 标准化
                # Pandas 支持直接对整个 DataFrame 进行四则运算，忽略 NaN
                normalized_data = (feature_data - params['median']) / params['range80']

                # 4. 重新拼接标签列和标准化后的数据
                # axis=1 表示按列拼接
                df_normalized = pd.concat([label_col, normalized_data], axis=1)

                # 5. 保存至原文件夹
                df_normalized.to_csv(output_path, index=False)
                print(f"  [成功] {file_name} -> {output_name}")

            except Exception as e:
                print(f"  [错误] 处理 {file_name} 时发生异常: {e}")

    print("\n所有特征参数的标准化处理已全部完成！")


if __name__ == "__main__":
    batch_standardize_features()