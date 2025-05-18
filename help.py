import os
import glob
import pandas as pd

# Поменяйте на реальный путь к вашей папке agg
AGG_DIR = '/Users/ruslanageev/PycharmProjects/Prophet/results/agg'

# ищем все файлы *_metrics_mean_*.csv
pattern = os.path.join(AGG_DIR, '*_metrics_mean_*.csv')
files = glob.glob(pattern)

if not files:
    print("Не найдено ни одного файла по шаблону", pattern)
    exit(1)

for path in sorted(files):
    name = os.path.basename(path)
    # split → [<dataset>, <method>.csv]
    parts = name.split('_metrics_mean_')
    dataset = parts[0]
    method  = parts[1].replace('.csv', '')

    df = pd.read_csv(path)

    # метрики
    f1_col = 'f1_fidelity'
    suf_col = 'sufficiency'
    com_col = 'comprehensiveness'

    # если в файле нет f1_fidelity, можно раскомментировать строчку ниже
    # f1_col = f1_col if f1_col in df.columns else com_col

    # для каждой метрики берём:
    #   * строку, где она максимальна
    #   * значение этой метрики и sparsity в этой строке
    #   * среднее по столбцу
    def best_and_mean(df, metric):
        idx_best = df['f1_fidelity'].idxmax()
        print(idx_best)
        # idx_best = 10
        return (
            df.loc[idx_best, metric],
            df.loc[idx_best, 'sparsity'],
            df[metric].mean()
        )

    f1_best, f1_spars_best, f1_mean    = best_and_mean(df, f1_col)
    suf_best, suf_spars_best, suf_mean = best_and_mean(df, suf_col)
    com_best, com_spars_best, com_mean = best_and_mean(df, com_col)

    print(f"=== Dataset: {dataset} | Method: {method} ===")
    print(f"{f1_col:15s}: {f1_best:.4f} @ sparsity={f1_spars_best}  (mean {f1_mean:.4f})")
    print(f"{suf_col:15s}: {suf_best:.4f} @ sparsity={suf_spars_best}  (mean {suf_mean:.4f})")
    print(f"{com_col:15s}: {com_best:.4f} @ sparsity={com_spars_best}  (mean {com_mean:.4f})")
    print()
