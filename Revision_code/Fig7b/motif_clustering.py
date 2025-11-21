import os
os.environ["SCIPY_ARRAY_API"] = "1"
import pandas as pd
import sys 
sys.path.append('/work3/Users/oba/references/chromBERT/stem_change/src')
import css_utility_rnakato_20251027 as crb

# init_df.csvというファイルを読み込んで、以下の処理を行う
# python motif_clustering.py -i init_df.csv -o output_DIR
# となるように、以下の処理を関数化する

# y_pred= crb.motif_init2pred_with_dendrogram(
# 				input_path='/content/init_df.csv', categorical=False, 
# 				fillna_method='ffill', linkage_method='complete', threshold=35)

# heatmap of DTW distance matrix reordered by clustering
# dtw_distance_matrix, y_pred=crb.motif_init2pred(
# 				input_path='/content/init_df.csv', categorical=False, 
# 				fillna_method="ffill", n_clusters=11, linkage_method="complete")
# import matplotlib.pyplot as plt
# import seaborn as sns
# import warnings
# from scipy.cluster.hierarchy import leaves_list, linkage, ClusterWarning
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore", category=ClusterWarning)
#     linkage_matrix = linkage(dtw_distance_matrix, method='complete')
#     ordered_indices = leaves_list(linkage_matrix)
#     reordered_matrix = dtw_distance_matrix[ordered_indices][:, ordered_indices]

# plt.figure(figsize=(6, 5))
# sns.heatmap(reordered_matrix, cmap='viridis', square=True)
# plt.title('Cluster-Reordered DTW Distance Matrix')
# plt.xlabel('Motif Index (reordered)')
# plt.ylabel('Motif Index (reordered)')
# plt.show()

# crb.motif_init2cluster_vis(
# 				input_path='/content/init_df.csv', categorical=False, n_clusters=11, 
# 				fillna_method="ffill", linkage_method="complete", random_state=2, 
# 				font_scale=0.0035,font_v_scale=10, fig_w=12, fig_h=8, node_size=1000, node_dist=0.05)

# crb.motif_init2umap(
# 				input_path='/content/init_df.csv', categorical=False,  n_clusters=11, 
# 				fillna_method="ffill", linkage_method="complete", n_neighbors=5, 
# 				min_dist=0.3, random_state=2)


def heatmap_DTW(input_path, output_path="./motif_dtw.pdf"):
	dtw_distance_matrix, y_pred=crb.motif_init2pred(
				input_path=input_path, categorical=False, 
				fillna_method="ffill", n_clusters=11, linkage_method="complete"
				)
	import matplotlib.pyplot as plt
	import seaborn as sns
	import warnings
	from scipy.cluster.hierarchy import leaves_list, linkage, ClusterWarning
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", category=ClusterWarning)
		linkage_matrix = linkage(dtw_distance_matrix, method='complete')
		ordered_indices = leaves_list(linkage_matrix)
		reordered_matrix = dtw_distance_matrix[ordered_indices][:, ordered_indices]

	plt.figure(figsize=(6, 5))
	sns.heatmap(reordered_matrix, cmap='viridis', square=True)
	plt.title('Cluster-Reordered DTW Distance Matrix')
	plt.xlabel('Motif Index (reordered)')
	plt.ylabel('Motif Index (reordered)')
	# save
	plt.savefig(output_path)
	return 

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="Motif clustering from init_df.csv")
	parser.add_argument("-i","--input_csv", type=str, help="Input CSV file (init_df.csv)")
	parser.add_argument("-o","--output_dir", type=str, help="Output directory for results")
	args = parser.parse_args()
	# Dendrogram and clustering
	y_pred= crb.motif_init2pred_with_dendrogram(
					input_path=args.input_csv, categorical=False, 
					fillna_method='ffill', linkage_method='complete', threshold=35,
					output_path=f"{args.output_dir}/motif_dendrogram.pdf"
					)
	# UMAP visualization
	umap_plt = crb.motif_init2umap(
					input_path=args.input_csv, categorical=False,  n_clusters=len(set(y_pred)), 
					fillna_method="ffill", linkage_method="complete", n_neighbors=5, 
					min_dist=0.3, random_state=2,
					output_path=f"{args.output_dir}/motif_umap.pdf"
					)
	# Cluster visualization
	clustr_plt = crb.motif_init2cluster_vis(
					input_path=args.input_csv, categorical=False, n_clusters=len(set(y_pred)), 
					fillna_method="ffill", linkage_method="complete", random_state=2, 
					font_scale=0.0035,font_v_scale=10, fig_w=12, fig_h=8, 
					node_size=1000, node_dist=0.05,
					output_path=f"{args.output_dir}/motif_cluster.pdf"
					)
	# heatmap of DTW distance matrix reordered by clustering
	heatmap_plt = heatmap_DTW(input_path=args.input_csv,
							  output_path=f"{args.output_dir}/motif_dtw.pdf")
	