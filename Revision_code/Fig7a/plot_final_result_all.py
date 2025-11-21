#!/usr/bin/env python3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

# 入力ファイル
DIR="/work3/Users/oba/references/chromBERT/stem_change/group_binary_datasets/Eval_DEBUG/ft_20251104_all"
input_file = f"{DIR}/all_results.txt"

# データ読み込み
df = pd.read_csv(input_file, sep=r"\s+", header=None, names=["pair", "ACC", "AUC", "F1"])

# _rを除去
df["pair"] = df["pair"].str.replace("_r", "", regex=False)

# "vs"で分割
df[["A", "B"]] = df["pair"].str.split("vs", expand=True)


# plot指示
# 1. Cancerを除外
df = df[(df["A"] != "Cancer") & (df["B"] != "Cancer")]
# 2. 指定順序
# cells = ['ESC', 'iPSC', 'ESderived', 'bloodTcell', 'HSCBcell', 'Mesenchymal', 'Brain', 'Muscle', 'Heart', 'SmoothMuscle']
cells = ['ESC', 'iPSC', 'ESderived', 'bloodTcell', 'HSCBcell', 'Brain', 'Muscle', 'Heart', 'SmoothMuscle']


df["A"] = pd.Categorical(df["A"], categories=cells, ordered=True)
df["B"] = pd.Categorical(df["B"], categories=cells, ordered=True)

# 比較対象の全リスト
samples = cells

# 指標ごとにピボットしてheatmapを作成
metrics = ["ACC", "AUC", "F1"]

# 保存ディレクトリ
outdir = f"{DIR}/fig"
os.makedirs(outdir, exist_ok=True)

for metric in metrics:
    # 対称マトリクスを初期化
    mat = pd.DataFrame(index=samples, columns=samples, dtype=float)

    # 対応する値を埋める
    for _, row in df.iterrows():
        mat.loc[row["A"], row["B"]] = row[metric]
        mat.loc[row["B"], row["A"]] = row[metric]  # 対称にする

    # 上三角をNaNにする（下三角のみ表示）
    mask = np.triu(np.ones_like(mat, dtype=bool), k=1)
    mat_masked = mat.mask(mask)
    
    # # 対角を1.0に（任意）
    # for s in samples:
    #     mat.loc[s, s] = 1.0

    # ヒートマップを描画
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        mat_masked.astype(float),
        annot=True,
        fmt=".3f",
        cmap="Oranges",
        square=True,
        mask=mask,
        cbar=True,
        vmin=0.5,   # カラースケール下限
        vmax=1.0,   # カラースケール上限
    )
    plt.title(f"{metric} heatmap")
    plt.tight_layout()

    # 保存
    plt.savefig(f"{outdir}/{metric}_heatmap.pdf", format='pdf')
    plt.close()

    plt.close()

    # --- クラスタリング付きヒートマップ ---
    cluster = sns.clustermap(
        mat.astype(float),
        cmap="Oranges",
        annot=True,
        fmt=".3f",
        square=True,
        cbar=True,
        vmin=0.5,
        vmax=1.0,
        figsize=(8, 8),
        method="average",
        metric="euclidean",
    )
    cluster.fig.suptitle(f"{metric} clustered heatmap", y=1.02)
    cluster.fig.tight_layout()
    cluster.savefig(f"{outdir}/{metric}_heatmap_clustered.pdf",format='pdf')
    plt.close(cluster.fig)

print(f"✅ {outdir} に ACC, AUC, F1 のヒートマップを保存しました。")
