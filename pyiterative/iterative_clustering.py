import scanpy as sc
import leidenalg
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from scipy.optimize import brentq
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from typing import Optional, Union, List, Tuple, Dict
import warnings
from multiprocessing import Pool, cpu_count
import logging
from pynndescent import NNDescent
from scipy.sparse import lil_matrix, csr_matrix
import igraph as ig
import concord as ccd
import torch
def Iterative_Clustering(adata, ndims=30, num_iterations=20, min_pct=0.4, min_log2_fc=2, batch_size=256, min_score=150, min_de_genes=1, min_cluster_size=4, batch_key=None):
    """
    Wrapper function to perform iterative clustering using scVI and Leiden algorithm.
    Args:
        adata: AnnData object containing the scRNA-seq data with the specified embedding in obsm.
        ndims: Number of latent dimensions to use from the embedding.
        num_iterations: Maximum number of clustering iterations.
        min_pct: Minimum percentage of cells expressing a gene to consider it for differential expression.
        min_log2_fc: Minimum log2 fold change for a gene to be considered differentially expressed.
        batch_size: Batch size for scVI differential expression.
        min_score: Minimum score for a gene to be considered differentially expressed.
        min_de_genes: Minimum number of differentially expressed genes required (returns score of 0 if below threshold).
        min_cluster_size: Minimum size of clusters to retain.
        model: scVI model object for differential expression analysis.
        embedding_key: Key in adata.obsm indicating the embedding to use (default: 'X_scVI').
    Returns:
        adata: AnnData object with updated clustering in adata.obs['leiden'].
    """
    adata.obs['leiden']='1'
    adata.obs['leiden'] = adata.obs['leiden'].astype('category')
    previous_num_clusters = 1
    for i in range(num_iterations):
        adata = Clustering_Iteration(adata, ndims=ndims, min_pct=min_pct, min_log2_fc=min_log2_fc, batch_size=batch_size, min_score=min_score, min_de_genes=min_de_genes, min_cluster_size=min_cluster_size, batch_key=batch_key)
        if len(adata.obs['leiden'].cat.categories) == previous_num_clusters:
            break
        previous_num_clusters = len(adata.obs['leiden'].cat.categories)
    return adata

def Find_Nearest_Cluster(centroids, cluster_labels, target_cluster):
    """
    Find the nearest cluster to the target cluster based on centroid distance.
    Args:
        centroids: Precomputed centroids array (n_clusters x n_dims)
        cluster_labels: List of cluster labels corresponding to centroid rows
        target_cluster: The cluster to find the nearest neighbor for
    Returns:
        nearest_cluster: The label of the nearest cluster, or None if no suitable cluster found
    """
    from sklearn.metrics import pairwise_distances
    
    # Get all clusters except the target cluster
    other_clusters = [c for c in cluster_labels if c != target_cluster]
    
    if len(other_clusters) == 0:
        return None
    
    try:
        # Find the index of target cluster and other clusters
        cluster_to_idx = {cluster: i for i, cluster in enumerate(cluster_labels)}
        
        if target_cluster not in cluster_to_idx:
            return None
            
        target_idx = cluster_to_idx[target_cluster]
        
        # Calculate distances from target cluster to all other clusters
        target_centroid = centroids[target_idx:target_idx+1]  # Keep as 2D array
        other_centroids = np.array([centroids[cluster_to_idx[c]] for c in other_clusters if c in cluster_to_idx])
        
        if len(other_centroids) == 0:
            return None
            
        # Calculate distances
        distances = pairwise_distances(target_centroid, other_centroids)[0]
        
        # Find nearest cluster
        nearest_idx = np.argmin(distances)
        nearest_cluster = other_clusters[nearest_idx]
        
        return nearest_cluster
        
    except Exception as e:
        print(f"Error finding nearest cluster for {target_cluster}: {e}")
        # Fallback: return the first available cluster
        return other_clusters[0] if other_clusters else None
def Find_Centroids(adata, cluster_key='leiden', embedding_key='X_scVI', ndims=30):
    """
    Calculates centroids in the scVI latent space for each cluster in adata.
    Args:
        adata: AnnData object containing the scRNA-seq data with obsm['X_scVI'].
        cluster_key: Key in adata.obs indicating cluster assignments.
        embedding_key: Key in adata.obsm indicating the embedding to use (e.g., 'X_scVI').
        ndims: Number of dimensions in the embedding to consider.
    Returns:
        Value array of shape (num_clusters, ndims) with centroids for each cluster.
    """
    
    centroids = adata.obsm[embedding_key].copy()
    
    centroids_df = pd.DataFrame(centroids)
    centroids_df['cluster'] = adata.obs[cluster_key].values
    
    valid_clusters = []
    for cluster in adata.obs[cluster_key].cat.categories:
        if np.sum(adata.obs[cluster_key] == cluster) > 0:
            valid_clusters.append(cluster)
    
    if not valid_clusters:
        return np.zeros((0, ndims))
        
    centroids_df = centroids_df[centroids_df['cluster'].isin(valid_clusters)]
    centroids_df = centroids_df.groupby('cluster').mean()
    
    if np.isnan(centroids_df.values).any():
        centroids_df = centroids_df.dropna()
        
    return centroids_df.values
def Clustering_Iteration(adata, ndims=30, min_pct=0.4, min_log2_fc=2, batch_size=256, min_score=150, min_de_genes=1, min_cluster_size=4, batch_key=None):
    """
    Performs one iteration of clustering and merging.
    Args:
         adata: AnnData object containing the scRNA-seq data with the specified embedding in obsm.
         ndims: Number of latent dimensions to use from the embedding.
         min_pct: Minimum percentage of cells expressing a gene to consider it for differential expression.
         min_log2_fc: Minimum log2 fold change for a gene to be considered differentially expressed.
         batch_size: Batch size for scVI differential expression.
         min_score: Minimum score for a gene to be considered differentially expressed.
         min_de_genes: Minimum number of differentially expressed genes required (returns score of 0 if below threshold).
         min_cluster_size: Minimum size of clusters to retain.
         model: scVI model object for differential expression analysis. If None, clustering will still occur but differential expression scoring will be skipped.
         embedding_key: Key in adata.obsm indicating the embedding to use (default: 'X_scVI').
    Returns:
         adata: AnnData object with updated clustering in adata.obs['leiden'].
    """
    
    clusters = adata.obs['leiden'].cat.categories.copy()
    
    for cluster in clusters:
        cluster_mask = adata.obs['leiden'] == cluster
        cluster_adata = adata[cluster_mask].copy()
        cluster_adata.layers['counts'] = cluster_adata.X.copy()
        sc.pp.normalize_total(cluster_adata, target_sum=1e4)
        sc.pp.log1p(cluster_adata)
        sc.pp.highly_variable_genes(cluster_adata, n_top_genes=3000, subset=False, flavor='seurat_v3', layer='counts')
        
        # Subset to highly variable genes for Concord
        hvg_genes = cluster_adata.var_names[cluster_adata.var['highly_variable']].tolist()
        cluster_adata_hvg = cluster_adata[:, hvg_genes].copy()
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        ccd_model = ccd.Concord(adata=cluster_adata_hvg, domain_key=batch_key, 
                                device=device, preload_dense=False, batch_size=batch_size, latent_dim=ndims,
                                encoder_dims=[int(2**(np.floor(np.sqrt(ndims))+1))]) # Use encoder_dims = 2^(floor(sqrt(ndims))+1)
        ccd_model.fit_transform(output_key='Concord')
        
        # Transfer the Concord embedding back to the original cluster_adata
        cluster_adata.obsm['Concord'] = cluster_adata_hvg.obsm['Concord']

        if cluster_adata.n_obs < min_cluster_size:
            continue
        print('Creating sNN graph...')
        if cluster_adata.n_obs < 20:
            k = int(np.floor(cluster_adata.n_obs/2))
        else:
            k = 20
        
        idx, distance = NNDescent(cluster_adata.obsm['Concord'][:, :ndims], n_neighbors=k).neighbor_graph
        idx = idx[:, 1:]  # Drop self from sNN
        n_cells = idx.shape[0]
        
        # Vectorized sNN calculation using sparse matrix operations
        # Create a binary neighbor matrix
        row_indices = np.repeat(np.arange(n_cells), k-1)
        col_indices = idx.flatten()
        data = np.ones(len(row_indices), dtype=np.float32)
        neighbor_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(n_cells, n_cells))
        
        # Compute shared neighbors: multiply neighbor matrix by its transpose
        # This gives the count of shared neighbors
        snn = neighbor_matrix.dot(neighbor_matrix.T)
        
        # Normalize by k to get the Jaccard-like similarity
        snn = snn.multiply(1.0 / k)
        
        # Make symmetric (take maximum)
        snn = snn.maximum(snn.T)
        
        # Prune edges with less than 1/15 similarity
        snn.data[snn.data < (1/15)] = 0
        snn.eliminate_zeros()
        
        cluster_adata.obsp['connectivities'] = snn
        # Convert sparse matrix to igraph directly to avoid scipy compatibility issues
        print('Performing Leiden clustering...')
        sources, targets = cluster_adata.obsp['connectivities'].nonzero()
        weights = cluster_adata.obsp['connectivities'].data
        g = ig.Graph(n=cluster_adata.n_obs, edges=list(zip(sources, targets)), 
                     edge_attrs={'weight': weights}, directed=False)
        part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=1)
        cluster_adata.obs['leiden'] = [str(c) for c in part.membership]
        cluster_adata.obs['leiden'] = cluster_adata.obs['leiden'].astype('category')
        
        cluster_adata.obs['leiden'] = cluster_adata.obs['leiden'].cat.remove_unused_categories()
        
        sub_clusters = cluster_adata.obs['leiden'].cat.categories
        nonempty_sub_clusters = [subcluster for subcluster in sub_clusters if np.sum(cluster_adata.obs['leiden'] == subcluster) > 0]
        
        if len(nonempty_sub_clusters) < 2:
            continue
            
        changes_made = True
        merged_pairs = []
        
        while changes_made:
            changes_made = False
            
            cluster_adata.obs['leiden'] = cluster_adata.obs['leiden'].cat.remove_unused_categories()
            
            sub_clusters = cluster_adata.obs['leiden'].cat.categories
            nonempty_sub_clusters = [subcluster for subcluster in sub_clusters if np.sum(cluster_adata.obs['leiden'] == subcluster) > 0]
            
            if len(nonempty_sub_clusters) < 2:
                break
            centroids = Find_Centroids(cluster_adata, cluster_key='leiden', embedding_key='Concord', ndims=ndims)
            
            if centroids.shape[0] < 2:
                break
                
            centroid_map = {subcluster: i for i, subcluster in enumerate(nonempty_sub_clusters)}
            
            min_distance = np.inf
            closest_pair = None
            
            for sub_cluster in nonempty_sub_clusters:
                if sub_cluster not in centroid_map:
                    continue
                
                # Use the refactored Find_Nearest_Cluster function
                closest_sub_cluster = Find_Nearest_Cluster(centroids, nonempty_sub_clusters, sub_cluster)
                
                if closest_sub_cluster is None:
                    continue
                
                if (sub_cluster, str(closest_sub_cluster)) in merged_pairs or (str(closest_sub_cluster), sub_cluster) in merged_pairs:
                    continue
                
                # Calculate distance between the pair for comparison
                idx = centroid_map[sub_cluster]
                closest_idx = centroid_map.get(closest_sub_cluster)
                
                if closest_idx is None or idx >= centroids.shape[0] or closest_idx >= centroids.shape[0]:
                    continue
                
                from sklearn.metrics import pairwise_distances
                distance = pairwise_distances(centroids[idx:idx+1], centroids[closest_idx:closest_idx+1])[0][0]
                
                if distance < min_distance:
                    min_distance = distance
                    closest_pair = (sub_cluster, closest_sub_cluster)
            
            if closest_pair is None:
                break
                
            sub_cluster, closest_sub_cluster = closest_pair
                
            n_cells_sub = np.sum(cluster_adata.obs['leiden'] == sub_cluster)
            n_cells_closest = np.sum(cluster_adata.obs['leiden'] == closest_sub_cluster)
            
            # Force merge if either cluster is too small (regardless of DE score)
            if n_cells_sub < min_cluster_size or n_cells_closest < min_cluster_size:
                print(f"Force merging small sub-clusters: {sub_cluster} ({n_cells_sub} cells) with {closest_sub_cluster} ({n_cells_closest} cells)")
                cluster_adata.obs.loc[cluster_adata.obs['leiden'] == closest_sub_cluster, 'leiden'] = sub_cluster
                merged_pairs.append((sub_cluster, str(closest_sub_cluster)))
                changes_made = True
                continue
            
            # Skip DE analysis if clusters are too small for reliable DE (but above min_cluster_size)
            if n_cells_sub < 3 or n_cells_closest < 3:
                merged_pairs.append((sub_cluster, str(closest_sub_cluster)))
                continue
                
            # Perform differential expression analysis for larger clusters
            bayes_de_score = DE_Score(cluster_adata, sub_cluster, closest_sub_cluster, min_pct, min_log2_fc, min_de_genes)
            
            if bayes_de_score < min_score:
                cluster_adata.obs.loc[cluster_adata.obs['leiden'] == closest_sub_cluster, 'leiden'] = sub_cluster
                merged_pairs.append((sub_cluster, str(closest_sub_cluster)))
                changes_made = True
        
        cluster_adata.obs['leiden'] = cluster_adata.obs['leiden'].cat.remove_unused_categories()
        
        # Store cluster mapping for later renaming
        final_sub_clusters = cluster_adata.obs['leiden'].cat.categories
        final_nonempty_sub_clusters = [subcluster for subcluster in final_sub_clusters if np.sum(cluster_adata.obs['leiden'] == subcluster) > 0]
        
        if len(final_nonempty_sub_clusters) > 1:
            # Sort subclusters for consistent ordering
            sorted_subclusters = sorted(final_nonempty_sub_clusters, key=lambda x: int(x))
            
            # Create hierarchical names by appending subcluster number to parent cluster
            # First, collect all temp labels and add them to categories
            temp_labels_to_add = []
            for subcluster in sorted_subclusters:
                temp_label = f"temp_{cluster}_{subcluster}"
                temp_labels_to_add.append(temp_label)
            
            # Add all temp labels to categories at once
            if temp_labels_to_add:
                new_categories = [cat for cat in temp_labels_to_add if cat not in adata.obs['leiden'].cat.categories]
                if new_categories:
                    adata.obs['leiden'] = adata.obs['leiden'].cat.add_categories(new_categories)
            
            # Now assign the temp labels
            for subcluster in sorted_subclusters:
                subcluster_mask = cluster_adata.obs['leiden'] == subcluster
                original_indices = cluster_adata.obs.index[subcluster_mask]
                # Temporarily store with cluster prefix to avoid conflicts
                temp_label = f"temp_{cluster}_{subcluster}"
                adata.obs.loc[original_indices, 'leiden'] = temp_label
    
    adata.obs['leiden'] = adata.obs['leiden'].cat.remove_unused_categories()
    
    # Final cleanup: merge any remaining clusters smaller than min_cluster_size
    final_cleanup_changes = True
    while final_cleanup_changes:
        final_cleanup_changes = False
        current_clusters = adata.obs['leiden'].cat.categories.copy()
        
        for cluster in current_clusters:
            cluster_size = np.sum(adata.obs['leiden'] == cluster)
            if cluster_size < min_cluster_size:
                # Find nearest cluster and merge
                other_clusters = [c for c in current_clusters if c != cluster and np.sum(adata.obs['leiden'] == c) > 0]
                if other_clusters:
                    # Calculate centroids for final cleanup
                    cleanup_centroids = Find_Centroids(adata, cluster_key='leiden', embedding_key='Concord', ndims=ndims)
                    nearest_cluster = Find_Nearest_Cluster(cleanup_centroids, current_clusters, cluster)
                    if nearest_cluster is not None:
                        print(f"Final cleanup: merging small cluster {cluster} ({cluster_size} cells) with nearest cluster {nearest_cluster}")
                        adata.obs.loc[adata.obs['leiden'] == cluster, 'leiden'] = nearest_cluster
                        final_cleanup_changes = True
                        break  # Start over to avoid modifying categories while iterating
        
        if final_cleanup_changes:
            adata.obs['leiden'] = adata.obs['leiden'].cat.remove_unused_categories()
    
    # Final renaming: convert temp labels to hierarchical cluster names
    adata.obs['leiden'] = adata.obs['leiden'].cat.remove_unused_categories()
    current_clusters = adata.obs['leiden'].cat.categories.copy()
    
    # Separate temp and non-temp clusters
    temp_clusters = [c for c in current_clusters if c.startswith('temp_')]
    non_temp_clusters = [c for c in current_clusters if not c.startswith('temp_')]
    
    # Group temp clusters by their parent cluster
    cluster_groups = {}
    for temp_cluster in temp_clusters:
        # Parse temp_parentcluster_subcluster
        # temp_cluster format: temp_{parent}_{subcluster}
        # where parent can be hierarchical like "1_4"
        temp_prefix = 'temp_'
        temp_body = temp_cluster[len(temp_prefix):]
        
        # Split from the right to separate subcluster number from parent
        parts = temp_body.rsplit('_', 1)
        if len(parts) == 2:
            parent_cluster = parts[0]
            subcluster = parts[1]
            if parent_cluster not in cluster_groups:
                cluster_groups[parent_cluster] = []
            cluster_groups[parent_cluster].append((temp_cluster, subcluster))
    
    # Create a mapping from old names to new names
    rename_mapping = {}
    
    # Process each parent cluster group
    for parent_cluster, temp_labels_with_subclusters in cluster_groups.items():
        # Sort by subcluster number for consistent ordering
        temp_labels_with_subclusters.sort(key=lambda x: int(x[1]))
        
        if len(temp_labels_with_subclusters) == 1:
            # Single subcluster - keep parent cluster name unchanged
            temp_label, subcluster = temp_labels_with_subclusters[0]
            new_name = parent_cluster
            rename_mapping[temp_label] = new_name
        else:
            # Multiple subclusters - append subcluster number to parent to create hierarchical name
            for i, (temp_label, subcluster) in enumerate(temp_labels_with_subclusters, 1):
                new_name = f"{parent_cluster}_{i}"
                rename_mapping[temp_label] = new_name
    
    # Add all new categories at once
    new_categories = [name for name in rename_mapping.values() if name not in adata.obs['leiden'].cat.categories]
    if new_categories:
        adata.obs['leiden'] = adata.obs['leiden'].cat.add_categories(new_categories)
    
    # Apply the renaming
    for old_name, new_name in rename_mapping.items():
        adata.obs.loc[adata.obs['leiden'] == old_name, 'leiden'] = new_name
    
    adata.obs['leiden'] = adata.obs['leiden'].cat.remove_unused_categories()
    print('Clustering iteration complete. Number of clusters:', len(adata.obs['leiden'].cat.categories))
    return adata
def DE_Score(adata, ident_1, ident_2, min_pct, min_log2_fc, min_de_genes):
    """
    Calculate differential expression score between two identities.
    Args:
        adata: AnnData object containing the scRNA-seq data.
        ident_1: First identity/group for comparison.
        ident_2: Second identity/group for comparison.
        min_pct: Minimum percentage of cells expressing a gene to consider it for differential expression.
        min_log2_fc: Minimum log2 fold change for a gene to be considered differentially expressed.
        min_de_genes: Minimum number of differentially expressed genes required (returns score of 0 if below threshold).
    Returns:
        de_score: Differential expression score sum(min(-log10(p),20)).
    """
    de_results = dge_2samples(
        adata,
        ident_1=ident_1,
        ident_2=ident_2,
        groupby='leiden',
        fc_thr=1,
        min_pct=0,
        max_pval=0.05,
        min_count=10,
        icc='i',
        df_correction=False,
        n_cores=max(1, cpu_count() - 1)
    )
    
    # Count number of DE genes meeting criteria
    de_genes = de_results[
        (abs(de_results['log2FC']) >= min_log2_fc) &
        (de_results['p.value.adj'] <= 0.05)
    ]

    de_genes = de_genes[de_genes['pct.1'] >= min_pct | de_genes['pct.2'] >= min_pct]
        
    if de_genes.shape[0] < min_de_genes:
        return 0
    
    return np.sum(np.minimum(-np.log10(de_genes['p.value.adj']), 20))
def dge_2samples(
    adata,
    features: Optional[List[str]] = None,
    ident_1: Optional[str] = None,
    ident_2: Optional[str] = None,
    groupby: str = 'leiden',
    fc_thr: float = 1.0,
    min_pct: float = 0.0,
    max_pval: float = 1.0,
    min_count: int = 30,
    icc: Union[str, float] = 'i',
    df_correction: bool = False,
    n_cores: int = 1
) -> pd.DataFrame:
    """
    Analyze differential gene expression between 2 identities using weighted t-test and chi-squared test.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with counts in .X or .raw.X
    features : list of str, optional
        List of genes to analyze. If None, all genes are analyzed.
    ident_1 : str
        First identity/group for comparison
    ident_2 : str
        Second identity/group for comparison
    groupby : str
        Column in adata.obs to use for grouping (default: 'leiden')
    fc_thr : float
        Fold-change threshold for reporting results
    min_pct : float
        Minimum fraction of cells expressing the gene in at least one group
    max_pval : float
        Maximum p-value for reporting results
    min_count : int
        Minimum aggregate count in at least one group
    icc : str or float
        Intracluster correlation coefficient method ('i' for iterative, 'A' for ANOVA, 0, or 1)
    df_correction : bool
        Apply correction for degrees of freedom (not recommended)
    n_cores : int
        Number of CPU cores for parallel processing
        
    Returns
    -------
    pd.DataFrame
        Results with columns: log2FC, p.value, p.value.adj, Chi2.p.value, pct.1, pct.2, Counts/Cell.1, Counts/Cell.2
    """
    iwt = iter_wght_ttest(
        adata, features, ident_1, ident_2, groupby, fc_thr, min_pct, 
        max_pval, min_count, icc, df_correction, n_cores
    )
    
    chi2 = chi2_test(
        adata, list(iwt.index), ident_1, ident_2, groupby, 
        fc_thr=1.0, min_pct=min_pct, max_pval=1.0, 
        min_count=min_count, n_cores=n_cores
    )
    
    # Merge results
    features_common = iwt.index.intersection(chi2.index)
    output = iwt.loc[features_common].copy()
    output['Chi2.p.value'] = chi2.loc[features_common, 'p.value']
    
    # Apply Benjamini-Hochberg correction to t-test p-values
    if len(output) > 0:
        _, pvals_adj, _, _ = multipletests(output['p.value'].values, method='fdr_bh')
        output['p.value.adj'] = pvals_adj
    
    return output


def chi2_test(
    adata,
    features: Optional[List[str]] = None,
    ident_1: Optional[str] = None,
    ident_2: Optional[str] = None,
    groupby: str = 'leiden',
    fc_thr: float = 1.0,
    min_pct: float = 0.0,
    max_pval: float = 1.0,
    min_count: int = 30,
    n_cores: int = 1
) -> pd.DataFrame:
    """
    Perform chi-squared test for differential gene expression.
    
    Returns
    -------
    pd.DataFrame
        Results with columns: log2FC, p.value, Counts/Cell.1, Counts/Cell.2
    """
    if ident_1 is None or ident_2 is None:
        raise ValueError("Both ident_1 and ident_2 must be defined")
    
    # Get gene list
    if features is None:
        gene_list = adata.var_names.tolist()
    else:
        gene_list = features
    
    # Get count matrix (prefer raw if available)
    if adata.raw is not None:
        X = adata.raw.X
        var_names = adata.raw.var_names
    else:
        X = adata.X
        var_names = adata.var_names
    
    # Subset by identities
    mask_1 = adata.obs[groupby] == ident_1
    mask_2 = adata.obs[groupby] == ident_2
    
    Ci_1 = X[mask_1, :]
    Ci_2 = X[mask_2, :]
    
    Nc_1 = Ci_1.shape[0]
    Nc_2 = Ci_2.shape[0]
    
    # Aggregate counts per gene
    if sp.issparse(Ci_1):
        AC_1 = np.array(Ci_1.sum(axis=0)).flatten()
        AC_2 = np.array(Ci_2.sum(axis=0)).flatten()
    else:
        AC_1 = Ci_1.sum(axis=0)
        AC_2 = Ci_2.sum(axis=0)
    
    TC_1 = AC_1.sum()
    TC_2 = AC_2.sum()
    
    # Create gene name to index mapping
    gene_to_idx = {gene: idx for idx, gene in enumerate(var_names)}
    
    def process_gene(gene_name):
        if gene_name not in gene_to_idx:
            return None
        
        idx = gene_to_idx[gene_name]
        ac1 = AC_1[idx]
        ac2 = AC_2[idx]
        
        # Get gene expression for min_pct calculation
        if sp.issparse(Ci_1):
            nonzero_1 = (Ci_1[:, idx].toarray().flatten() != 0).sum()
            nonzero_2 = (Ci_2[:, idx].toarray().flatten() != 0).sum()
        else:
            nonzero_1 = (Ci_1[:, idx] != 0).sum()
            nonzero_2 = (Ci_2[:, idx] != 0).sum()
        
        pct_1 = nonzero_1 / Nc_1
        pct_2 = nonzero_2 / Nc_2
        
        if (ac1 >= min_count or ac2 >= min_count) and (pct_1 > min_pct or pct_2 > min_pct):
            cont_table = np.array([
                [TC_1 - ac1, TC_2 - ac2],
                [ac1, ac2]
            ])
            
            fc = (ac1 / TC_1) / (ac2 / TC_2) if ac2 > 0 else np.nan
            
            if not np.isnan(fc) and (fc >= fc_thr or fc <= 1/fc_thr):
                _, p_value = stats.chi2_contingency(cont_table)[:2]
                
                if p_value <= max_pval:
                    return {
                        'gene': gene_name,
                        'log2FC': np.log2(fc),
                        'p.value': p_value,
                        'Counts/Cell.1': ac1 / Nc_1,
                        'Counts/Cell.2': ac2 / Nc_2
                    }
        return None
    
    print("Performing chi^2 test:")
    
    if n_cores > 1:
        with Pool(n_cores) as pool:
            results = list(tqdm(pool.imap(process_gene, gene_list), total=len(gene_list)))
    else:
        results = [process_gene(gene) for gene in tqdm(gene_list)]
    
    # Filter None results and create DataFrame
    results = [r for r in results if r is not None]
    
    if len(results) == 0:
        return pd.DataFrame(columns=['log2FC', 'p.value', 'Counts/Cell.1', 'Counts/Cell.2'])
    
    output = pd.DataFrame(results)
    output.set_index('gene', inplace=True)
    
    return output


def _process_gene_weighted_ttest(args):
    """Helper function for parallel processing in iter_wght_ttest."""
    gene_name, gene_to_idx, Ci_1, Ci_2, Xi_1, Xi_2, Ni_1, Ni_2, Nc_1, Nc_2, min_count, min_pct, fc_thr, max_pval, icc, df_correction = args
    
    if gene_name not in gene_to_idx:
        return None
    
    idx = gene_to_idx[gene_name]
    
    # Get counts for this gene
    if sp.issparse(Ci_1):
        h_1_col = Ci_1[:, idx]
        h_2_col = Ci_2[:, idx]
        h_1 = h_1_col.toarray().flatten()
        h_2 = h_2_col.toarray().flatten()
        xi_1 = Xi_1[:, idx].toarray().flatten()
        xi_2 = Xi_2[:, idx].toarray().flatten()
        
        # Use scipy.sparse method for efficient nonzero counting
        if hasattr(h_1_col, 'count_nonzero'):
            nonzero_1 = h_1_col.count_nonzero()
            nonzero_2 = h_2_col.count_nonzero()
        else:
            nonzero_1 = (h_1 != 0).sum()
            nonzero_2 = (h_2 != 0).sum()
    else:
        h_1 = Ci_1[:, idx]
        h_2 = Ci_2[:, idx]
        xi_1 = Xi_1[:, idx]
        xi_2 = Xi_2[:, idx]
        nonzero_1 = (xi_1 != 0).sum()
        nonzero_2 = (xi_2 != 0).sum()
    
    AC_1 = h_1.sum()
    AC_2 = h_2.sum()
    
    pct_1 = nonzero_1 / Nc_1
    pct_2 = nonzero_2 / Nc_2
    
    if (AC_1 >= min_count or AC_2 >= min_count) and \
       (pct_1 > min_pct or pct_2 > min_pct):
        
        wi_1 = icc_weight(h_1, Ni_1, icc)
        wi_2 = icc_weight(h_2, Ni_2, icc)
        
        fc = (xi_1 * wi_1).sum() / (xi_2 * wi_2).sum() if (xi_2 * wi_2).sum() > 0 else np.nan
        
        if not np.isnan(fc) and (fc >= fc_thr or fc <= 1/fc_thr) and \
           (nonzero_1 >= 3 or nonzero_2 >= 3):
            
            if df_correction:
                p_value = alt_wttest2(xi_1, xi_2, wi_1, wi_2)
            else:
                p_value = alt_wttest(xi_1, xi_2, wi_1, wi_2)
            
            if p_value <= max_pval:
                return {
                    'gene': gene_name,
                    'log2FC': np.log2(fc),
                    'p.value': p_value,
                    'pct.1': pct_1,
                    'pct.2': pct_2,
                    'Counts/Cell.1': AC_1 / Nc_1,
                    'Counts/Cell.2': AC_2 / Nc_2
                }
    return None


def iter_wght_ttest(
    adata,
    features: Optional[List[str]] = None,
    ident_1: Optional[str] = None,
    ident_2: Optional[str] = None,
    groupby: str = 'leiden',
    fc_thr: float = 1.0,
    min_pct: float = 0.0,
    max_pval: float = 1.0,
    min_count: int = 30,
    icc: Union[str, float] = 'i',
    df_correction: bool = False,
    n_cores: int = 1
) -> pd.DataFrame:
    """
    Perform weighted t-test with iterative weight calculation.
    
    Returns
    -------
    pd.DataFrame
        Results with columns: log2FC, p.value, p.value.adj, pct.1, pct.2, Counts/Cell.1, Counts/Cell.2
    """
    if ident_1 is None or ident_2 is None:
        raise ValueError("Both ident_1 and ident_2 must be defined")
    
    # Get gene list
    if features is None:
        gene_list = adata.var_names.tolist()
    else:
        gene_list = features
    
    # Get count matrix
    if adata.raw is not None:
        X = adata.raw.X
        var_names = adata.raw.var_names
    else:
        X = adata.X
        var_names = adata.var_names
    
    # Subset by identities
    mask_1 = adata.obs[groupby] == ident_1
    mask_2 = adata.obs[groupby] == ident_2
    
    Ci_1 = X[mask_1, :]
    Ci_2 = X[mask_2, :]
    
    Nc_1 = Ci_1.shape[0]
    Nc_2 = Ci_2.shape[0]
    
    # Calculate total counts per cell
    if sp.issparse(Ci_1):
        Ni_1 = np.array(Ci_1.sum(axis=1)).flatten()
        Ni_2 = np.array(Ci_2.sum(axis=1)).flatten()
    else:
        Ni_1 = Ci_1.sum(axis=1)
        Ni_2 = Ci_2.sum(axis=1)
    
    # Normalize counts
    if sp.issparse(Ci_1):
        Xi_1 = Ci_1.multiply(1 / Ni_1[:, np.newaxis])
        Xi_2 = Ci_2.multiply(1 / Ni_2[:, np.newaxis])
    else:
        Xi_1 = Ci_1 / Ni_1[:, np.newaxis]
        Xi_2 = Ci_2 / Ni_2[:, np.newaxis]
    
    # Create gene name to index mapping
    gene_to_idx = {gene: idx for idx, gene in enumerate(var_names)}
    
    print("Performing weighted t-test:")
    
    # Prepare arguments for parallel processing
    args_list = [(gene, gene_to_idx, Ci_1, Ci_2, Xi_1, Xi_2, Ni_1, Ni_2, Nc_1, Nc_2, 
                  min_count, min_pct, fc_thr, max_pval, icc, df_correction) 
                 for gene in gene_list]
    
    if n_cores > 1:
        with Pool(n_cores) as pool:
            results = list(tqdm(pool.imap(_process_gene_weighted_ttest, args_list), total=len(gene_list)))
    else:
        results = [_process_gene_weighted_ttest(args) for args in tqdm(args_list)]
    
    # Filter None results and create DataFrame
    results = [r for r in results if r is not None]
    
    if len(results) == 0:
        return pd.DataFrame(columns=['log2FC', 'p.value', 'p.value.adj', 'pct.1', 'pct.2', 'Counts/Cell.1', 'Counts/Cell.2'])
    
    output = pd.DataFrame(results)
    output.set_index('gene', inplace=True)
    
    # Apply Benjamini-Hochberg correction to p-values
    if len(output) > 0:
        _, pvals_adj, _, _ = multipletests(output['p.value'].values, method='fdr_bh')
        output['p.value.adj'] = pvals_adj
    
    return output


def alt_wttest(x1: np.ndarray, x2: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> float:
    """
    Alternative weighted t-test based on Margolin-Leikin variance estimator.
    
    Parameters
    ----------
    x1, x2 : array
        Data arrays for groups 1 and 2
    w1, w2 : array
        Weight arrays for groups 1 and 2
        
    Returns
    -------
    float
        P-value from weighted t-test
    """
    if len(x1) != len(w1) or len(x2) != len(w2):
        raise ValueError("Length mismatch between data and weights")
    
    # Normalize weights
    w1 = w1 / w1.sum()
    w2 = w2 / w2.sum()
    
    # Weighted means
    m1 = (x1 * w1).sum()
    m2 = (x2 * w2).sum()
    
    # Weighted variances (unbiased when w ~ 1/s^2)
    vm1 = (w1**2 * (x1 - m1)**2).sum() / (1 - (w1**2).sum())
    vm2 = (w2**2 * (x2 - m2)**2).sum() / (1 - (w2**2).sum())
    
    # Standard error
    s12 = np.sqrt(vm1 + vm2)
    
    if s12 == 0:
        return 1.0
    
    # T-statistic
    t = (m1 - m2) / s12
    
    # Degrees of freedom (Welch-Satterthwaite approximation)
    df = s12**4 / (vm1**2/(len(x1)-1) + vm2**2/(len(x2)-1))
    
    # P-value (two-tailed)
    p = 2 * stats.t.sf(np.abs(t), df=df)
    
    return p


def alt_wttest2(x1: np.ndarray, x2: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> float:
    """
    Alternative weighted t-test with effective degrees of freedom correction.
    
    Parameters
    ----------
    x1, x2 : array
        Data arrays for groups 1 and 2
    w1, w2 : array
        Weight arrays for groups 1 and 2
        
    Returns
    -------
    float
        P-value from weighted t-test
    """
    if len(x1) != len(w1) or len(x2) != len(w2):
        raise ValueError("Length mismatch between data and weights")
    
    # Normalize weights
    w1 = w1 / w1.sum()
    w2 = w2 / w2.sum()
    
    # Effective sample sizes
    n1 = 1 / (w1**2).sum()
    n2 = 1 / (w2**2).sum()
    
    # Weighted means
    m1 = (x1 * w1).sum()
    m2 = (x2 * w2).sum()
    
    # Weighted variances
    vm1 = (w1**2 * (x1 - m1)**2).sum() / (1 - 1/n1)
    vm2 = (w2**2 * (x2 - m2)**2).sum() / (1 - 1/n2)
    
    # Standard error
    s12 = np.sqrt(vm1 + vm2)
    
    if s12 == 0:
        return 1.0
    
    # T-statistic
    t = (m1 - m2) / s12
    
    # Degrees of freedom
    df = s12**4 / (vm1**2/(n1-1) + vm2**2/(n2-1))
    
    # P-value (two-tailed)
    p = 2 * stats.t.sf(np.abs(t), df=df)
    
    return p


def icc_an(h: np.ndarray, n: np.ndarray) -> float:
    """
    Calculate ANOVA intracluster correlation coefficient (ICC).
    
    Parameters
    ----------
    h : array
        Count values
    n : array
        Total counts per observation
        
    Returns
    -------
    float
        ICC value (clamped to [0, 1])
    """
    N = n.sum()
    k = len(n)
    n0 = (1/(k-1)) * (N - (n**2).sum()/N)
    
    h2_n = h**2 / n
    MSw = (1/(N-k)) * (h.sum() - h2_n.sum())
    MSb = (1/(k-1)) * (h2_n.sum() - (1/N) * h.sum()**2)
    
    denom = MSb + (n0 - 1) * MSw
    
    if denom == 0:
        icc = 0.0
    else:
        icc = (MSb - MSw) / denom
    
    # Clamp to [0, 1]
    return np.clip(icc, 0.0, 1.0)


def icc_iter(h: np.ndarray, n: np.ndarray) -> float:
    """
    Calculate iterative ICC providing more accurate variance matching.
    
    Parameters
    ----------
    h : array
        Count values
    n : array
        Total counts per observation
        
    Returns
    -------
    float
        ICC value (clamped to [0, 1])
    """
    x = h / n
    sum_n = n.sum()
    
    # Initial weights
    w0 = n / sum_n
    w0_sq_sum = (w0**2).sum()
    x0 = (x * w0).sum()
    
    # Initial variances
    VarT0 = x0 * (1 - x0) / sum_n
    VarE0 = (w0**2 * (x - x0)**2).sum() / (1 - w0_sq_sum)
    
    if VarE0 <= VarT0:
        return 0.0
    
    def f(icc, x, n):
        """Function to find root for ICC calculation."""
        wprop = n / (1 + icc * (n - 1))
        sum_wprop = wprop.sum()
        w = wprop / sum_wprop
        x1 = (x * w).sum()
        VarT = x1 * (1 - x1) / sum_wprop
        w_sq = w**2
        VarE = (w_sq * (x - x1)**2).sum() / (1 - w_sq.sum())
        return VarE - VarT
    
    try:
        icc_val = brentq(f, 0, 1, args=(x, n), xtol=1e-4/n.max())
        return min(icc_val, 1.0)
    except ValueError:
        return 0.0


def icc_weight(h: np.ndarray, n: np.ndarray, icc: Union[str, float] = 'i') -> np.ndarray:
    """
    Calculate statistical weights based on ICC.
    
    Parameters
    ----------
    h : array
        Count values
    n : array
        Total counts per observation
    icc : str or float
        ICC method: 'i' (iterative), 'A' (ANOVA), 0, or 1
        
    Returns
    -------
    array
        Normalized weights
    """
    Nc = len(n)
    
    if len(h) != Nc:
        raise ValueError("Unequal lengths of h and n vectors")
    
    if Nc < 3:
        raise ValueError("At least 3 observations are required")
    
    # If too few nonzero counts, return equal weights
    if (h != 0).sum() < 3:
        return np.ones(Nc) / Nc
    
    # Determine ICC value
    if icc == 'i':
        icc_val = icc_iter(h, n)
    elif icc == 'A':
        icc_val = icc_an(h, n)
    elif icc in [0, 1]:
        icc_val = float(icc)
    else:
        raise ValueError("Invalid icc, must be 'i', 'A', 0, or 1")
    
    # Calculate and return normalized weights
    wprop = n / (1 + icc_val * (n - 1))
    return wprop / wprop.sum()


def dge_multisample(
    adata,
    samples_1: List[str],
    samples_2: List[str],
    sample_key: str = 'sample',
    features: Optional[List[str]] = None,
    t_test: bool = False,
    min_pct: float = 0.03,
    fc_thr: float = 1.0,
    max_pval: float = 1.0,
    icc: Union[str, float] = 'i',
    df_correction: bool = False,
    n_cores: int = 1
) -> Dict:
    """
    Perform multi-sample differential gene expression analysis.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    samples_1 : list of str
        Sample identities for group 1 (at least 3)
    samples_2 : list of str
        Sample identities for group 2 (at least 3)
    sample_key : str
        Column in adata.obs containing sample identities
    features : list of str, optional
        Genes to analyze
    t_test : bool
        Use unweighted t-test for sample comparison
    min_pct : float
        Minimum expression fraction threshold
    fc_thr : float
        Fold-change threshold
    max_pval : float
        Maximum p-value threshold
    icc : str or float
        ICC method for within-sample weighting
    df_correction : bool
        Apply degrees of freedom correction
    n_cores : int
        Number of CPU cores
        
    Returns
    -------
    dict
        Dictionary with 'DGE' (results DataFrame), 'Sstats' (sample statistics), 
        and 'parameters' (analysis parameters)
    """
    if len(samples_1) < 3 or len(samples_2) < 3:
        raise ValueError("At least 3 samples per group are required")
    
    # Calculate weighted averages within samples
    print("Calculating weighted averages within samples...")
    adata_av = cnt_av(adata, sample_key, features, icc)
    
    # Generate sample matrix
    print("Generating sample matrix...")
    sample_matrix = create_sample_matrix(adata_av, samples_1, samples_2)
    
    # Perform weighted t-test on sample matrix
    print("Performing multi-sample analysis...")
    output = wt_multisample(
        sample_matrix, features, t_test, min_pct, fc_thr, max_pval, df_correction
    )
    
    # Create parameters dictionary
    params = {
        't_test': str(t_test),
        'min_pct': str(min_pct),
        'fc_thr': str(fc_thr),
        'max_pval': str(max_pval),
        'icc': str(icc),
        'df_correction': str(df_correction),
        'n_cores': str(n_cores)
    }
    
    return {
        'DGE': output['DGE'],
        'Sstats': output['Sstats'],
        'parameters': params
    }


def cnt_av(
    adata,
    sample_key: str = 'sample',
    features: Optional[List[str]] = None,
    icc: Union[str, float] = 'i'
) -> Dict:
    """
    Calculate weighted average counts and variances for each sample.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    sample_key : str
        Column in adata.obs containing sample identities
    features : list of str, optional
        Genes to analyze
    icc : str or float
        ICC method
        
    Returns
    -------
    dict
        Dictionary with 'AV_data' (per-sample statistics) and 'Sstats' (sample summaries)
    """
    if features is None:
        gene_list = adata.var_names.tolist()
    else:
        gene_list = features
    
    samples = adata.obs[sample_key].unique()
    Ns = len(samples)
    
    if Ns < 6:
        raise ValueError("At least 6 samples with different identities are required")
    
    # Get count matrix
    if adata.raw is not None:
        X = adata.raw.X
        var_names = adata.raw.var_names
    else:
        X = adata.X
        var_names = adata.var_names
    
    # Create gene to index mapping
    gene_to_idx = {gene: idx for idx, gene in enumerate(var_names) if gene in gene_list}
    gene_list_filtered = [g for g in gene_list if g in gene_to_idx]
    Ng = len(gene_list_filtered)
    
    # Initialize storage
    av_data = {}
    sstats = pd.DataFrame(
        index=['N.cells', 'N.counts', 'Counts/cell'],
        columns=samples
    )
    
    print("Averaging counts and calculating variance for each sample:")
    
    for sample in tqdm(samples):
        # Subset data for this sample
        mask = adata.obs[sample_key] == sample
        Ci = X[mask, :]
        
        # Calculate total counts per cell
        if sp.issparse(Ci):
            Ni = np.array(Ci.sum(axis=1)).flatten()
        else:
            Ni = Ci.sum(axis=1)
        
        Nc = Ci.shape[0]
        
        # Record sample statistics
        sstats.loc['N.cells', sample] = Nc
        sstats.loc['N.counts', sample] = Ni.sum()
        sstats.loc['Counts/cell', sample] = Ni.sum() / Nc
        
        # Normalize counts (% UMI)
        if sp.issparse(Ci):
            Xi_normalized = Ci.multiply(100 / Ni[:, np.newaxis])
        else:
            Xi_normalized = 100 * Ci / Ni[:, np.newaxis]
        
        # Calculate statistics for each gene
        AV = np.zeros(Ng)
        VAR = np.zeros(Ng)
        PCT = np.zeros(Ng)
        
        for i, gene in enumerate(gene_list_filtered):
            idx = gene_to_idx[gene]
            
            # Get counts for this gene
            if sp.issparse(Ci):
                h = Ci[:, idx].toarray().flatten()
                x = Xi_normalized[:, idx].toarray().flatten()
            else:
                h = Ci[:, idx]
                x = Xi_normalized[:, idx]
            
            # Calculate weights
            w = icc_weight(h, Ni, icc)
            
            # Weighted average
            AV[i] = (x * w).sum()
            
            # Variance
            if AV[i] != 0:
                VAR[i] = (w**2 * (x - AV[i])**2).sum() / (1 - (w**2).sum())
            else:
                VAR[i] = 1 / Ni.sum()**2
            
            # Expression fraction
            PCT[i] = (h != 0).sum() / Nc
        
        # Store results
        av_data[sample] = pd.DataFrame({
            'AV': AV,
            'VAR': VAR,
            'PCT': PCT
        }, index=gene_list_filtered)
    
    return {'AV_data': av_data, 'Sstats': sstats}


def create_sample_matrix(
    av_dict: Dict,
    samples_1: List[str],
    samples_2: List[str]
) -> Dict:
    """
    Create sample matrices from averaged data.
    
    Parameters
    ----------
    av_dict : dict
        Output from cnt_av function
    samples_1 : list of str
        Sample names for group 1
    samples_2 : list of str
        Sample names for group 2
        
    Returns
    -------
    dict
        Dictionary containing matrices for both groups and sample statistics
    """
    N_1 = len(samples_1)
    N_2 = len(samples_2)
    
    if N_1 < 3 or N_2 < 3:
        raise ValueError("At least 3 samples per group are required")
    
    av_data = av_dict['AV_data']
    sstats = av_dict['Sstats']
    
    # Get gene list from first sample
    gene_list = av_data[samples_1[0]].index.tolist()
    Ng = len(gene_list)
    
    # Initialize matrices
    AV_1 = pd.DataFrame(index=gene_list, columns=samples_1)
    VAR_1 = pd.DataFrame(index=gene_list, columns=samples_1)
    PCT_1 = pd.DataFrame(index=gene_list, columns=samples_1)
    
    AV_2 = pd.DataFrame(index=gene_list, columns=samples_2)
    VAR_2 = pd.DataFrame(index=gene_list, columns=samples_2)
    PCT_2 = pd.DataFrame(index=gene_list, columns=samples_2)
    
    print("Generating sample matrix:")
    
    # Fill matrices for group 1
    for sample in tqdm(samples_1, desc="Group 1"):
        AV_1[sample] = av_data[sample]['AV']
        VAR_1[sample] = av_data[sample]['VAR']
        PCT_1[sample] = av_data[sample]['PCT']
    
    # Fill matrices for group 2
    for sample in tqdm(samples_2, desc="Group 2"):
        AV_2[sample] = av_data[sample]['AV']
        VAR_2[sample] = av_data[sample]['VAR']
        PCT_2[sample] = av_data[sample]['PCT']
    
    return {
        'AV_1': AV_1,
        'VAR_1': VAR_1,
        'PCT_1': PCT_1,
        'AV_2': AV_2,
        'VAR_2': VAR_2,
        'PCT_2': PCT_2,
        'Sstats': sstats
    }


def iter_var(av: np.ndarray, va: np.ndarray) -> np.ndarray:
    """
    Calculate weights for variance vectors using iterative approach.
    
    Parameters
    ----------
    av : array
        Average expression values
    va : array
        Variance values
        
    Returns
    -------
    array
        Normalized weights
    """
    Nv = len(va)
    
    if len(av) != Nv:
        raise ValueError("Unequal lengths of average and variance vectors")
    
    if Nv < 3:
        raise ValueError("At least 3 samples are required")
    
    x = av
    Vari = va
    
    # Initial calculations
    inv_Vari = 1 / Vari
    sum_inv_Vari = inv_Vari.sum()
    VarT0 = 1 / sum_inv_Vari
    w0 = inv_Vari / sum_inv_Vari
    w0_sq_sum = (w0**2).sum()
    x0 = (x * w0).sum()
    VarE0 = (w0**2 * (x - x0)**2).sum() / (1 - w0_sq_sum)
    
    if VarE0 <= VarT0:
        Vp = 0
    else:
        def f(Vp, x, Vari):
            wprop = 1 / (Vari + Vp)
            sum_wprop = wprop.sum()
            VarT = 1 / sum_wprop
            w = wprop / sum_wprop
            x1 = (x * w).sum()
            w_sq = w**2
            VarE = (w_sq * (x - x1)**2).sum() / (1 - w_sq.sum())
            return VarE - VarT
        
        try:
            Vp1 = Vari.max()
            Vp = brentq(f, 0, Vp1, args=(x, Vari), xtol=Vari.min() * 1e-4)
            Vp = max(Vp, 0)
        except ValueError:
            Vp = 0
    
    wprop_final = 1 / (Vari + Vp)
    return wprop_final / wprop_final.sum()


def wt_multisample(
    sample_matrix: Dict,
    features: Optional[List[str]] = None,
    t_test: bool = False,
    min_pct: float = 0.03,
    fc_thr: float = 1.0,
    max_pval: float = 1.0,
    df_correction: bool = False
) -> Dict:
    """
    Perform weighted t-test on multiple samples.
    
    Parameters
    ----------
    sample_matrix : dict
        Output from create_sample_matrix
    features : list of str, optional
        Genes to analyze
    t_test : bool
        Use unweighted t-test
    min_pct : float
        Minimum expression fraction
    fc_thr : float
        Fold-change threshold
    max_pval : float
        Maximum p-value
    df_correction : bool
        Apply DF correction
        
    Returns
    -------
    dict
        Dictionary with 'DGE' and 'Sstats'
    """
    AV1 = sample_matrix['AV_1']
    var1 = sample_matrix['VAR_1']
    AV2 = sample_matrix['AV_2']
    var2 = sample_matrix['VAR_2']
    pct1 = sample_matrix['PCT_1']
    pct2 = sample_matrix['PCT_2']
    sstats = sample_matrix['Sstats']
    
    if features is None:
        gene_list = AV1.index.tolist()
    else:
        gene_list = features
    
    results = []
    
    print("Performing weighted t-test:")
    
    for gene in tqdm(gene_list):
        if gene not in AV1.index:
            continue
        
        Xi1 = AV1.loc[gene].values
        Xi2 = AV2.loc[gene].values
        VARi1 = var1.loc[gene].values
        VARi2 = var2.loc[gene].values
        PCTi1 = pct1.loc[gene].values
        PCTi2 = pct2.loc[gene].values
        
        N1 = len(Xi1)
        N2 = len(Xi2)
        
        if N1 >= 3 and N2 >= 3 and (PCTi1.mean() >= min_pct or PCTi2.mean() >= min_pct):
            # Calculate weights
            if t_test:
                wi_1 = np.ones(N1) / N1
                wi_2 = np.ones(N2) / N2
            else:
                wi_1 = iter_var(Xi1, VARi1)
                wi_2 = iter_var(Xi2, VARi2)
            
            # Weighted averages
            Xi1av = (Xi1 * wi_1).sum()
            Xi2av = (Xi2 * wi_2).sum()
            
            # Standard deviations
            Xi1sd = np.sqrt((wi_1**2 * (Xi1 - Xi1av)**2).sum() / (1 - (wi_1**2).sum()))
            Xi2sd = np.sqrt((wi_2**2 * (Xi2 - Xi2av)**2).sum() / (1 - (wi_2**2).sum()))
            
            # Fold change
            fc = Xi1av / Xi2av if Xi2av != 0 else np.nan
            
            if not np.isnan(fc) and (fc >= fc_thr or fc <= 1/fc_thr):
                # Weighted t-test
                if df_correction:
                    p_value = alt_wttest2(Xi1, Xi2, wi_1, wi_2)
                else:
                    p_value = alt_wttest(Xi1, Xi2, wi_1, wi_2)
                
                if p_value <= max_pval:
                    results.append({
                        'gene': gene,
                        'log2FC': np.log2(fc),
                        'p.value': p_value,
                        'Wtd.%UMI.1': Xi1av,
                        'Sd.%UMI.1': Xi1sd,
                        'Av.min.pct.1': PCTi1.mean(),
                        'Wtd.%UMI.2': Xi2av,
                        'Sd.%UMI.2': Xi2sd,
                        'Av.min.pct.2': PCTi2.mean()
                    })
    
    if len(results) == 0:
        dge_df = pd.DataFrame(columns=[
            'log2FC', 'p.value', 'Wtd.%UMI.1', 'Sd.%UMI.1', 'Av.min.pct.1',
            'Wtd.%UMI.2', 'Sd.%UMI.2', 'Av.min.pct.2'
        ])
    else:
        dge_df = pd.DataFrame(results)
        dge_df.set_index('gene', inplace=True)
    
    return {'DGE': dge_df, 'Sstats': sstats}


# Convenience functions for scanpy integration

def add_dge_results_to_adata(adata, results: pd.DataFrame, key: str = 'dge'):
    """
    Add DGE results to AnnData object.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    results : DataFrame
        DGE results from dge_2samples or iter_wght_ttest
    key : str
        Key to use in adata.uns for storing results
    """
    adata.uns[key] = results
    return adata