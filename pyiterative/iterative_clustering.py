import scanpy as sc
import anndata as ad
import leidenalg
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from scipy.optimize import brenth
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from multiprocessing import cpu_count
from joblib import Parallel, delayed
from sklearn.metrics import pairwise_distances
from pynndescent import NNDescent
from scipy.sparse import csr_matrix
import igraph as ig
import concord as ccd
import torch
try:
    import cupy as cp
except Exception:
    cp = None


# Check for GPU availability
try:
    _GPU_AVAILABLE = torch.cuda.is_available()
except Exception:
    _GPU_AVAILABLE = False
_DEVICE = torch.device('cuda' if _GPU_AVAILABLE else 'cpu')

# Try to import CuPy for GPU acceleration
_CUPY_AVAILABLE = False
_GPU_CACHE = {}
try:
    _CUPY_AVAILABLE = cp is not None and cp.cuda.is_available()
    if _CUPY_AVAILABLE:
        # Enable pinned memory for faster CPU-GPU transfers
        cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)
        # Cache for persistent GPU data
        _GPU_CACHE = {'n_array': None, 'n_hash': None}
except Exception:
    _CUPY_AVAILABLE = False

DEFAULT_COUNTS_LAYER = 'counts'


def _is_backed(adata):
    return bool(getattr(adata, 'isbacked', False))


def _obs_positions(index, n_obs):
    if isinstance(index, slice):
        return np.arange(n_obs)[index]
    
    index = np.asarray(index)
    if index.dtype == bool:
        return np.flatnonzero(index)
    return index


def _copy_adata(adata, backed_load_chunk_size=None, obs_idx=slice(None), var_idx=slice(None)):
    if not _is_backed(adata):
        return adata[obs_idx, var_idx].copy()
    
    if getattr(adata, 'is_view', False):
        return adata.to_memory()[obs_idx, var_idx].copy()
    
    if backed_load_chunk_size is None:
        return adata[obs_idx, var_idx].to_memory()
    
    if backed_load_chunk_size <= 0:
        raise ValueError("backed_load_chunk_size must be a positive integer or None")
    
    obs_positions = _obs_positions(obs_idx, adata.n_obs)
    chunks = []
    for start in range(0, len(obs_positions), backed_load_chunk_size):
        end = min(start + backed_load_chunk_size, len(obs_positions))
        chunks.append(adata[obs_positions[start:end], var_idx].to_memory())
    
    if not chunks:
        return adata.to_memory()
    
    if len(chunks) == 1:
        return chunks[0]
    
    return ad.concat(chunks, axis=0, join='outer', merge='same', uns_merge='same')


def _counts_layer_or_none(adata, counts_layer=DEFAULT_COUNTS_LAYER):
    if counts_layer is not None and counts_layer in adata.layers:
        return counts_layer
    return None


def _raw_counts_for_current_vars(adata):
    if adata.raw is None:
        return None
    try:
        return adata.raw[:, adata.var_names.tolist()].X
    except (KeyError, ValueError, IndexError):
        return None


def _ensure_counts_layer(adata, counts_layer=DEFAULT_COUNTS_LAYER):
    if counts_layer is not None and counts_layer not in adata.layers:
        raw_counts = _raw_counts_for_current_vars(adata)
        adata.layers[counts_layer] = raw_counts.copy() if raw_counts is not None else adata.X.copy()


def _set_x_to_counts(adata, counts_layer=DEFAULT_COUNTS_LAYER):
    layer = _counts_layer_or_none(adata, counts_layer)
    if layer is not None:
        adata.X = adata.layers[layer].copy()
        return
    
    raw_counts = _raw_counts_for_current_vars(adata)
    if raw_counts is not None:
        adata.X = raw_counts.copy()


def _get_count_matrix(adata, counts_layer=DEFAULT_COUNTS_LAYER):
    layer = _counts_layer_or_none(adata, counts_layer)
    if layer is not None:
        return adata.layers[layer], adata.var_names
    if adata.raw is not None:
        return adata.raw.X, adata.raw.var_names
    return adata.X, adata.var_names


def _run_hvg(adata, n_top_genes, counts_layer=DEFAULT_COUNTS_LAYER, x_is_log_normalized=False, context=None, backed_load_chunk_size=None):
    hvg_layer = _counts_layer_or_none(adata, counts_layer)
    hvg_adata = None
    hvg_target = adata
    
    if _is_backed(adata):
        hvg_adata = _copy_adata(adata, backed_load_chunk_size=backed_load_chunk_size)
        hvg_target = hvg_adata
        hvg_layer = _counts_layer_or_none(hvg_target, counts_layer)
    
    if hvg_layer is None:
        raw_counts = _raw_counts_for_current_vars(hvg_target)
        if raw_counts is not None:
            if hvg_adata is None:
                hvg_adata = _copy_adata(hvg_target)
            hvg_adata.X = raw_counts.copy()
            hvg_target = hvg_adata
            x_is_log_normalized = False
    
    hvg_kwargs = {
        'n_top_genes': n_top_genes,
        'subset': False,
        'flavor': 'seurat_v3',
        'span': 0.5
    }
    if hvg_layer is not None:
        hvg_kwargs['layer'] = hvg_layer
    
    try:
        sc.pp.highly_variable_genes(hvg_target, **hvg_kwargs)
    except (ValueError, RuntimeError) as e:
        if 'Extrapolation not allowed' in str(e):
            hvg_kwargs['span'] = 1.0
            try:
                sc.pp.highly_variable_genes(hvg_target, **hvg_kwargs)
            except Exception:
                if context is not None:
                    print(f"Warning: seurat_v3 HVG failed for {context} ({str(e)}), using seurat flavor instead")
                _run_seurat_hvg_fallback(hvg_target, n_top_genes, counts_layer, x_is_log_normalized)
        else:
            if context is not None:
                print(f"Warning: seurat_v3 HVG failed for {context} ({str(e)}), using seurat flavor instead")
            _run_seurat_hvg_fallback(hvg_target, n_top_genes, counts_layer, x_is_log_normalized)
    
    if hvg_adata is not None:
        adata.var['highly_variable'] = hvg_adata.var['highly_variable'].values


def _run_seurat_hvg_fallback(adata, n_top_genes, counts_layer=DEFAULT_COUNTS_LAYER, x_is_log_normalized=False):
    if x_is_log_normalized:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, subset=False, flavor='seurat')
        return
    
    hvg_adata = _copy_adata(adata)
    _set_x_to_counts(hvg_adata, counts_layer)
    sc.pp.normalize_total(hvg_adata, target_sum=1e4)
    sc.pp.log1p(hvg_adata)
    sc.pp.highly_variable_genes(hvg_adata, n_top_genes=n_top_genes, subset=False, flavor='seurat')
    adata.var['highly_variable'] = hvg_adata.var['highly_variable'].values


def _resolve_concord_chunked(adata, concord_chunked=None, source_is_backed=False):
    if concord_chunked is None:
        return source_is_backed or _is_backed(adata)
    return bool(concord_chunked)


def _make_concord_model(adata, input_feature, batch_key, batch_size, ndims, seed, concord_chunked=None, concord_chunk_size=10000, source_is_backed=False):
    use_chunked = _resolve_concord_chunked(adata, concord_chunked, source_is_backed=source_is_backed)
    if use_chunked and concord_chunk_size <= 0:
        raise ValueError("concord_chunk_size must be a positive integer")
    
    effective_batch_size = min(batch_size, adata.n_obs)
    concord_kwargs = {
        'adata': adata,
        'input_feature': input_feature,
        'domain_key': batch_key,
        'device': _DEVICE,
        'preload_dense': False,
        'batch_size': effective_batch_size,
        'latent_dim': ndims,
        'encoder_dims': [int(2**(np.floor(np.sqrt(ndims))+1))],
        'save_dir': None,
        'seed': seed
    }
    
    if use_chunked:
        concord_kwargs['chunked'] = True
        concord_kwargs['chunk_size'] = concord_chunk_size
    
    return ccd.Concord(**concord_kwargs)


def Iterative_Clustering(adata, ndims=64, num_iterations=20, min_pct=0.5, min_log2_fc=2, batch_size=256, min_score=150, min_de_genes=4, min_cluster_size=4, batch_key=None, n_cores=None, DE_batch_size=2048, icc_gpu=False, min_pval=0.05, pct_diff=0.7, seed=42, counts_layer=DEFAULT_COUNTS_LAYER, concord_chunked=None, concord_chunk_size=10000, backed_load_chunk_size=None):
    """
    Wrapper function to perform iterative clustering using scVI and Leiden algorithm.
    Args:
        adata: AnnData object containing the scRNA-seq data with the specified embedding in obsm.
        ndims: Number of latent dimensions to use from the embedding (default: 64).
        num_iterations: Maximum number of clustering iterations (default: 20).
        min_pct: Minimum percentage of cells expressing a gene to consider it for differential expression (default: 0.5).
        min_log2_fc: Minimum log2 fold change for a gene to be considered differentially expressed (default: 2).
        batch_size: Batch size for the CONCORD model (default: 256).
        min_score: Minimum score for a gene to be considered differentially expressed (default: 150).
        min_de_genes: Minimum number of differentially expressed genes required (returns score of 0 if below threshold) (default: 4).
        min_cluster_size: Minimum size of clusters to retain (default: 4).
        batch_key: Key in adata.obs indicating batch information for CONCORD model.
        n_cores: Number of CPU cores to use for parallel processing (default: max(1, cpu_count() - 1)).
        DE_batch_size: Batch size for GPU processing in dge_2samples (default: 2048).
        icc_gpu: Use GPU for ICC weight computation in dge_2samples (default: False). Set to False to force CPU.
        min_pval: Minimum BH-adjusted p-value for a gene to be considered differentially expressed (default: 0.05).
        pct_diff: Minimum percentage difference threshold for DE genes. If pct_1 > pct_2: pct_diff = (pct_1-pct_2)/pct_1. If pct_2 > pct_1: pct_diff = (pct_2-pct_1)/pct_2 (default: 0.7).
        seed: Random seed for reproducibility (default: 42).
        counts_layer: Layer containing raw counts for HVG and DE analysis. If absent, falls back to .raw and then .X (default: 'counts').
        concord_chunked: Whether to use CONCORD's chunked loader. If None, enabled when the source AnnData is backed/on disk.
        concord_chunk_size: Number of cells per CONCORD chunk when chunked loading is enabled (default: 10000).
        backed_load_chunk_size: Number of cells per chunk when materializing backed AnnData slices for pyIterative steps. If None, load slices at once.
    Returns:
        adata: AnnData object with updated clustering in adata.obs['leiden'].
    """
    if n_cores is None:
        n_cores = max(1, cpu_count() - 1)
    source_is_backed = _is_backed(adata)
    if source_is_backed:
        print("Detected backed/on-disk AnnData. CONCORD chunked loading is enabled by default; pyIterative still materializes cluster slices for HVG, graph, and DE steps.")
    # Place all cells in a single initial cluster
    adata.obs['leiden']='1'
    adata.obs['leiden'] = adata.obs['leiden'].astype('category')
    previous_num_clusters = 1
    # Iterative loop
    for i in range(num_iterations):
        adata = Clustering_Iteration(adata, ndims=ndims, min_pct=min_pct, min_log2_fc=min_log2_fc, batch_size=batch_size, min_score=min_score, min_de_genes=min_de_genes, min_cluster_size=min_cluster_size, batch_key=batch_key, n_cores=n_cores, DE_batch_size=DE_batch_size, icc_gpu=icc_gpu, min_pval=min_pval, pct_diff=pct_diff, seed=seed, counts_layer=counts_layer, concord_chunked=concord_chunked, concord_chunk_size=concord_chunk_size, backed_load_chunk_size=backed_load_chunk_size)
        if len(adata.obs['leiden'].cat.categories) == previous_num_clusters:
            break
        previous_num_clusters = len(adata.obs['leiden'].cat.categories)
    
    # Final validation: ensure all clusters have DE_score > min_score with their closest cluster
    print('Performing final validation of cluster separation...')
    final_validation_state = {
        'pairs': {},
        'cluster_versions': {},
    }
    final_validation_changes = True
    while final_validation_changes:
        final_validation_changes = False
        current_clusters = adata.obs['leiden'].cat.categories.copy()
        current_cluster_keys = {str(cluster) for cluster in current_clusters}
        cluster_versions = final_validation_state['cluster_versions']
        for cluster_key in current_cluster_keys:
            cluster_versions.setdefault(cluster_key, 0)
        for cluster_key in list(cluster_versions):
            if cluster_key not in current_cluster_keys:
                del cluster_versions[cluster_key]
        final_validation_state['pairs'] = {
            pair_key: pair_state
            for pair_key, pair_state in final_validation_state['pairs'].items()
            if set(pair_key).issubset(current_cluster_keys)
        }
        
        if len(current_clusters) < 2:
            break
        
        # Calculate centroids for all final clusters in CONCORD space.
        # Keep the full adata count matrix untouched; normalize only the HVG subset used by CONCORD.
        _run_hvg(adata, n_top_genes=2000, counts_layer=counts_layer, x_is_log_normalized=False, context='final validation', backed_load_chunk_size=backed_load_chunk_size)
        hvg_genes = adata.var_names[adata.var['highly_variable']].tolist()
        adata_hvg = _copy_adata(adata, backed_load_chunk_size=backed_load_chunk_size, var_idx=hvg_genes)
        _ensure_counts_layer(adata_hvg, counts_layer)
        _set_x_to_counts(adata_hvg, counts_layer)
        sc.pp.normalize_total(adata_hvg, target_sum=1e4)
        sc.pp.log1p(adata_hvg)
        ccd_model = _make_concord_model(adata_hvg, hvg_genes, batch_key, batch_size, ndims, seed,
                                        concord_chunked=concord_chunked, concord_chunk_size=concord_chunk_size,
                                        source_is_backed=source_is_backed)
        ccd_model.fit_transform(output_key='Concord', save_model=False)
        final_centroids = Find_Centroids(adata_hvg, cluster_key='leiden', embedding_key='Concord', ndims=ndims)
        
        if final_centroids.shape[0] < 2:
            break
        
        # Check each cluster against its nearest neighbor
        for cluster in current_clusters:
            cluster_size = np.sum(adata.obs['leiden'] == cluster)
            
            # Skip if cluster no longer exists (might have been merged in previous iteration)
            if cluster_size == 0:
                continue
            
            # Find nearest cluster
            nearest_cluster = Find_Nearest_Cluster(final_centroids, current_clusters, cluster)
            
            if nearest_cluster is None:
                continue
            
            nearest_cluster_size = np.sum(adata.obs['leiden'] == nearest_cluster)
            
            # Skip if nearest cluster no longer exists
            if nearest_cluster_size == 0:
                continue
            
            cluster_key = str(cluster)
            nearest_cluster_key = str(nearest_cluster)
            pair_key = tuple(sorted([cluster_key, nearest_cluster_key]))
            current_pair_versions = {
                cluster_key: cluster_versions.get(cluster_key, 0),
                nearest_cluster_key: cluster_versions.get(nearest_cluster_key, 0),
            }
            pair_state = final_validation_state['pairs'].get(pair_key)
            if (
                pair_state is not None
                and pair_state.get('compared', False)
                and pair_state.get('cluster_versions') == current_pair_versions
            ):
                continue
            
            # Calculate DE score between cluster and its nearest neighbor
            de_score = DE_Score(adata, cluster, nearest_cluster, min_pct, min_log2_fc, min_de_genes, DE_batch_size=DE_batch_size, n_cores=n_cores, icc_gpu=icc_gpu, min_pval=min_pval, pct_diff=pct_diff, counts_layer=counts_layer)
            
            if de_score < min_score:
                print(f"Final validation: merging cluster {cluster} ({cluster_size} cells) with nearest cluster {nearest_cluster} ({nearest_cluster_size} cells) - DE score: {de_score:.2f}")
                adata.obs.loc[adata.obs['leiden'] == cluster, 'leiden'] = nearest_cluster
                cluster_versions[nearest_cluster_key] = cluster_versions.get(nearest_cluster_key, 0) + 1
                cluster_versions.pop(cluster_key, None)
                final_validation_state['pairs'] = {
                    cached_pair_key: cached_pair_state
                    for cached_pair_key, cached_pair_state in final_validation_state['pairs'].items()
                    if cluster_key not in cached_pair_key and nearest_cluster_key not in cached_pair_key
                }
                final_validation_changes = True
                # Start over after a merge
                break
            else:
                final_validation_state['pairs'][pair_key] = {
                    'compared': True,
                    'cluster_versions': current_pair_versions.copy(),
                }
        
        if final_validation_changes:
            adata.obs['leiden'] = adata.obs['leiden'].cat.remove_unused_categories()
    
    print(f'Final validation complete. Final number of clusters: {len(adata.obs["leiden"].cat.categories)}')
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
def Find_Centroids(adata, cluster_key='leiden', embedding_key='Concord', ndims=30):
    """
    Calculates centroids in the scVI latent space for each cluster in adata.
    Args:
        adata: AnnData object containing the scRNA-seq data
        cluster_key: Key in adata.obs indicating cluster assignments.
        embedding_key: Key in adata.obsm indicating the embedding to use (e.g., 'Concord').
        ndims: Number of dimensions in the embedding to consider.
    Returns:
        Value array of shape (num_clusters, ndims) with centroids for each cluster.
    """
    
    centroids = adata.obsm[embedding_key].copy()
    
    centroids_df = pd.DataFrame(centroids)
    centroids_df['cluster'] = adata.obs[cluster_key].values
    # Filter for clusters that have at least one cell
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
def Clustering_Iteration(adata, ndims=30, min_pct=0.4, min_log2_fc=2, batch_size=256, min_score=150, min_de_genes=1, min_cluster_size=4, batch_key=None, n_cores=None, DE_batch_size=2048, icc_gpu=True, min_pval=0.05, pct_diff=0.7, seed=42, counts_layer=DEFAULT_COUNTS_LAYER, concord_chunked=None, concord_chunk_size=10000, backed_load_chunk_size=None):
    """
    Performs one iteration of clustering and merging.
    Args:
         adata: AnnData object containing the scRNA-seq data.
         ndims: Number of latent dimensions to use from the embedding.
         min_pct: Minimum percentage of cells expressing a gene to consider it for differential expression.
         min_log2_fc: Minimum log2 fold change for a gene to be considered differentially expressed.
         batch_size: Batch size for the CONCORD model.
         min_score: Minimum score for a gene to be considered differentially expressed.
         min_de_genes: Minimum number of differentially expressed genes required (returns score of 0 if below threshold).
         min_cluster_size: Minimum size of clusters to retain.
         batch_key: Key in adata.obs indicating batch information for CONCORD model.
         n_cores: Number of CPU cores to use for parallel processing. Default is max(1, cpu_count() - 1).
         DE_batch_size: Batch size for GPU processing in dge_2samples.
         icc_gpu: Use GPU for ICC weight computation in dge_2samples. Set to False to force CPU.
         min_pval: Minimum BH-adjusted p-value for a gene to be considered differentially expressed.
         pct_diff: Minimum percentage difference threshold for DE genes. If pct_1 > pct_2: pct_diff = (pct_1-pct_2)/pct_1. If pct_2 > pct_1: pct_diff = (pct_2-pct_1)/pct_2 (default: 0.7).
         seed: Random seed for reproducibility.
         counts_layer: Layer containing raw counts for HVG and DE analysis. If absent, falls back to .raw and then .X.
         concord_chunked: Whether to use CONCORD's chunked loader. If None, enabled when the source AnnData is backed/on disk.
         concord_chunk_size: Number of cells per CONCORD chunk when chunked loading is enabled.
         backed_load_chunk_size: Number of cells per chunk when materializing backed AnnData slices. If None, load slices at once.
    Returns:
         adata: AnnData object with updated clustering in adata.obs['leiden'].
    """
    if n_cores is None:
        n_cores = max(1, cpu_count() - 1)
    
    source_is_backed = _is_backed(adata)
    clusters = adata.obs['leiden'].cat.categories.copy()
    
    for cluster in clusters:
        cluster_mask = (adata.obs['leiden'] == cluster).values
        cluster_adata = _copy_adata(adata, backed_load_chunk_size=backed_load_chunk_size, obs_idx=cluster_mask)
        _ensure_counts_layer(cluster_adata, counts_layer)
        _set_x_to_counts(cluster_adata, counts_layer)
        sc.pp.normalize_total(cluster_adata, target_sum=1e4)
        sc.pp.log1p(cluster_adata)
        
        # Try to find highly variable genes with error handling
        n_genes = min(2000, cluster_adata.n_vars)
        _run_hvg(cluster_adata, n_top_genes=n_genes, counts_layer=counts_layer, x_is_log_normalized=True, context=f'cluster {cluster}', backed_load_chunk_size=backed_load_chunk_size)
        
        # Subset to highly variable genes for CONCORD
        hvg_genes = cluster_adata.var_names[cluster_adata.var['highly_variable']].tolist()
        
        # If no HVG found, skip this cluster
        if len(hvg_genes) == 0:
            print(f"Warning: No highly variable genes found for cluster {cluster}, skipping")
            continue
        
        # Check cluster size BEFORE attempting to fit CONCORD model
        if cluster_adata.n_obs <= min_cluster_size:
            print(f"Warning: Cluster {cluster} has {cluster_adata.n_obs} cells (=< {min_cluster_size}), skipping")
            continue
            
        cluster_adata_hvg = _copy_adata(cluster_adata, backed_load_chunk_size=backed_load_chunk_size, var_idx=hvg_genes)
        
        ccd_model = _make_concord_model(cluster_adata_hvg, hvg_genes, batch_key, batch_size, ndims, seed,
                                        concord_chunked=concord_chunked, concord_chunk_size=concord_chunk_size,
                                        source_is_backed=source_is_backed)
        
        try:
            ccd_model.fit_transform(output_key='Concord', save_model=False)
        except Exception as e:
            print(f"Warning: Concord fit_transform failed for cluster {cluster} ({str(e)}), skipping")
            continue
        
        # Transfer the Concord embedding back to the original cluster_adata
        cluster_adata.obsm['Concord'] = cluster_adata_hvg.obsm['Concord']
        print('Creating sNN graph...')
        if cluster_adata.n_obs < 20:
            k = int(np.floor(cluster_adata.n_obs/2))
        else:
            k = 20
        
        idx, distance = NNDescent(cluster_adata.obsm['Concord'][:, :ndims], n_neighbors=k, random_state=seed).neighbor_graph
        # Drop self from kNN
        idx = idx[:, 1:]
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
        # Leiden clustering
        part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=1, seed=seed)
        cluster_adata.obs['leiden'] = [str(c) for c in part.membership]
        cluster_adata.obs['leiden'] = cluster_adata.obs['leiden'].astype('category')
        
        cluster_adata.obs['leiden'] = cluster_adata.obs['leiden'].cat.remove_unused_categories()
        
        sub_clusters = cluster_adata.obs['leiden'].cat.categories
        nonempty_sub_clusters = [subcluster for subcluster in sub_clusters if np.sum(cluster_adata.obs['leiden'] == subcluster) > 0]
        
        if len(nonempty_sub_clusters) < 2:
            continue
            
        changes_made = True
        merged_pairs = []
        # Check for merges until no more can be made
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
            
            # Build list of all pairs with their distances
            all_pairs = []
            
            for sub_cluster in nonempty_sub_clusters:
                if sub_cluster not in centroid_map:
                    continue
                
                # Find nearest cluster
                closest_sub_cluster = Find_Nearest_Cluster(centroids, nonempty_sub_clusters, sub_cluster)
                
                if closest_sub_cluster is None:
                    continue
                
                # Skip if already tested
                if (sub_cluster, str(closest_sub_cluster)) in merged_pairs or (str(closest_sub_cluster), sub_cluster) in merged_pairs:
                    continue
                
                # Calculate distance
                idx = centroid_map[sub_cluster]
                closest_idx = centroid_map.get(closest_sub_cluster)
                
                if closest_idx is None or idx >= centroids.shape[0] or closest_idx >= centroids.shape[0]:
                    continue
                
                distance = pairwise_distances(centroids[idx:idx+1], centroids[closest_idx:closest_idx+1])[0][0]
                
                # Store pair with canonical ordering to avoid duplicates
                pair_key = tuple(sorted([sub_cluster, closest_sub_cluster]))
                all_pairs.append((distance, pair_key[0], pair_key[1]))
            
            # Remove duplicate pairs and sort by distance
            seen_pairs = set()
            unique_pairs = []
            for dist, c1, c2 in all_pairs:
                pair_key = tuple(sorted([c1, c2]))
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    unique_pairs.append((dist, c1, c2))
            
            unique_pairs.sort(key=lambda x: x[0])  # Sort by distance (closest first)
            
            if len(unique_pairs) == 0:
                break
            
            # Test pairs in order from closest to farthest
            for distance, sub_cluster, closest_sub_cluster in unique_pairs:
                # Check if pair was already tested
                if (sub_cluster, str(closest_sub_cluster)) in merged_pairs or (str(closest_sub_cluster), sub_cluster) in merged_pairs:
                    continue
                
                n_cells_sub = np.sum(cluster_adata.obs['leiden'] == sub_cluster)
                n_cells_closest = np.sum(cluster_adata.obs['leiden'] == closest_sub_cluster)
                
                # Force merge if either cluster is too small (regardless of DE score)
                if n_cells_sub < min_cluster_size or n_cells_closest < min_cluster_size:
                    print(f"Force merging small sub-clusters: {sub_cluster} ({n_cells_sub} cells) with {closest_sub_cluster} ({n_cells_closest} cells)")
                    cluster_adata.obs.loc[cluster_adata.obs['leiden'] == closest_sub_cluster, 'leiden'] = sub_cluster
                    merged_pairs.append((sub_cluster, str(closest_sub_cluster)))
                    changes_made = True
                    # Recalculate after merge
                    break
                
                # Skip DE analysis if clusters are too small for reliable DE (but above min_cluster_size)
                if n_cells_sub < 3 or n_cells_closest < 3:
                    merged_pairs.append((sub_cluster, str(closest_sub_cluster)))
                    continue
                    
                # Perform differential expression analysis for larger clusters
                bayes_de_score = DE_Score(cluster_adata, sub_cluster, closest_sub_cluster, min_pct, min_log2_fc, min_de_genes, n_cores=n_cores, DE_batch_size=DE_batch_size, icc_gpu=icc_gpu, min_pval=min_pval, pct_diff=pct_diff, counts_layer=counts_layer)
                
                if bayes_de_score < min_score:
                    cluster_adata.obs.loc[cluster_adata.obs['leiden'] == closest_sub_cluster, 'leiden'] = sub_cluster
                    merged_pairs.append((sub_cluster, str(closest_sub_cluster)))
                    changes_made = True
                    # Recalculate after merge
                    break
                else:
                    # Mark pair as tested but not merged
                    merged_pairs.append((sub_cluster, str(closest_sub_cluster)))
        
        cluster_adata.obs['leiden'] = cluster_adata.obs['leiden'].cat.remove_unused_categories()
        
        # Store cluster mapping for later renaming
        final_sub_clusters = cluster_adata.obs['leiden'].cat.categories
        final_nonempty_sub_clusters = [subcluster for subcluster in final_sub_clusters if np.sum(cluster_adata.obs['leiden'] == subcluster) > 0]
        
        if len(final_nonempty_sub_clusters) > 1:
            # Sort subclusters for consistent ordering
            sorted_subclusters = sorted(final_nonempty_sub_clusters, key=lambda x: int(x))
            
            # Create hierarchical names by appending subcluster number to parent cluster
            # Collect all temp labels and add them to categories
            temp_labels_to_add = []
            for subcluster in sorted_subclusters:
                temp_label = f"temp_{cluster}_{subcluster}"
                temp_labels_to_add.append(temp_label)
            
            # Add all temp labels to categories at once
            if temp_labels_to_add:
                new_categories = [cat for cat in temp_labels_to_add if cat not in adata.obs['leiden'].cat.categories]
                if new_categories:
                    adata.obs['leiden'] = adata.obs['leiden'].cat.add_categories(new_categories)
            
            # Assign the temp labels
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


def DE_Score(adata, ident_1, ident_2, min_pct, min_log2_fc, min_de_genes, DE_batch_size=2048, n_cores=None, icc_gpu=True, min_pval=0.05, pct_diff=0.7, counts_layer=DEFAULT_COUNTS_LAYER):
    """
    Calculate differential expression score between two identities.
    Args:
        adata: AnnData object containing the scRNA-seq data.
        ident_1: First identity/group for comparison.
        ident_2: Second identity/group for comparison.
        min_pct: Minimum percentage of cells expressing a gene to consider it for differential expression.
        min_log2_fc: Minimum log2 fold change for a gene to be considered differentially expressed.
        min_de_genes: Minimum number of differentially expressed genes required (returns score of 0 if below threshold).
        DE_batch_size: Batch size for GPU processing in dge_2samples (default: 2048).
        n_cores: Number of CPU cores to use for parallel processing. Default is max(1, cpu_count() - 1).
        icc_gpu: Use GPU for ICC weight computation in dge_2samples. Set to False to force CPU.
        min_pval: Minimum BH-adjusted p-value for a gene to be considered differentially expressed (default: 0.05).
        pct_diff: Minimum percentage difference threshold for DE genes. If pct_1 > pct_2: pct_diff = (pct_1-pct_2)/pct_1. If pct_2 > pct_1: pct_diff = (pct_2-pct_1)/pct_2 (default: 0.7).
        counts_layer: Layer containing raw counts for DE analysis. If absent, falls back to .raw and then .X.
    Returns:
        de_score: Differential expression score sum(min(-log10(p_adj),20)).
    """
    if n_cores is None:
        n_cores = max(1, cpu_count() - 1)
    # Run differential expression analysis
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
        n_cores=n_cores,
        gpu_batch_size=DE_batch_size,
        icc_gpu=icc_gpu,
        counts_layer=counts_layer
    )
    
    # Count number of DE genes meeting criteria
    de_genes = de_results[
        (abs(de_results['log2FC']) >= min_log2_fc) &
        (de_results['p.value.adj'] <= min_pval) & (de_results['Chi2.p.value'] <= 0.05)
    ]

    de_genes = de_genes[(de_genes['pct.1'] >= min_pct) | (de_genes['pct.2'] >= min_pct)]
    
    # Apply pct_diff filter
    pct_diff_values = np.where(
        de_genes['pct.1'] > de_genes['pct.2'],
        (de_genes['pct.1'] - de_genes['pct.2']) / de_genes['pct.1'],
        (de_genes['pct.2'] - de_genes['pct.1']) / de_genes['pct.2']
    )
    de_genes = de_genes[pct_diff_values >= pct_diff]
    
    # Return 0 if not enough DE genes
    if de_genes.shape[0] < min_de_genes:
        return 0
    # Calculate DE score
    return np.sum(np.minimum(-np.log10(de_genes['p.value.adj']), 20))
def dge_2samples(adata, features=None, ident_1=None, ident_2=None, groupby='leiden', fc_thr=1.0, min_pct=0.0, max_pval=1.0, min_count=30, icc='i', df_correction=False, n_cores=1, use_gpu=True, gpu_batch_size=2048, icc_gpu=True, counts_layer=DEFAULT_COUNTS_LAYER):
    """
    Analyze differential gene expression between 2 identities using weighted t-test and chi-squared test.
    Args:
        adata: AnnData object containing the scRNA-seq data with counts in a layer, .raw.X, or .X.
        features: List of genes to analyze. If None, all genes are analyzed.
        ident_1: First identity/group for comparison.
        ident_2: Second identity/group for comparison.
        groupby: Column in adata.obs to use for grouping (default: 'leiden').
        fc_thr: Fold-change threshold for reporting results.
        min_pct: Minimum fraction of cells expressing the gene in at least one group.
        max_pval: Maximum p-value for reporting results.
        min_count: Minimum aggregate count in at least one group.
        icc: Intracluster correlation coefficient method ('i' for iterative, 'A' for ANOVA, 0, or 1).
        df_correction: Apply correction for degrees of freedom (not recommended).
        n_cores: Number of CPU cores for parallel processing.
        use_gpu: Use GPU acceleration for chi-squared test (default: True if CUDA available).
        gpu_batch_size: Number of genes to process per GPU batch (default: 2048).
        icc_gpu: Use GPU for ICC weight computation (default: True). Set to False to force CPU.
        counts_layer: Layer containing raw counts. If absent, falls back to .raw and then .X.
    Returns:
        pd.DataFrame with columns: log2FC, p.value, p.value.adj, Chi2.p.value, pct.1, pct.2.
    """
    # Run iterative weighted t-test
    iwt = iter_wght_ttest(
        adata, features, ident_1, ident_2, groupby, fc_thr, min_pct, 
        max_pval, min_count, icc, df_correction, n_cores,
        use_gpu=use_gpu, gpu_batch_size=gpu_batch_size, icc_gpu=icc_gpu,
        counts_layer=counts_layer
    )
    
    # Use GPU-accelerated chi2 test if available and requested
    if use_gpu and _GPU_AVAILABLE:
        chi2 = chi2_test_gpu(
            adata, list(iwt.index), ident_1, ident_2, groupby, 
            fc_thr=1.0, min_pct=0.0, max_pval=max_pval, 
            min_count=0, batch_size=gpu_batch_size, device=None,
            counts_layer=counts_layer
        )
    else:
        # Fall back to CPU version
        chi2 = chi2_test(
            adata, list(iwt.index), ident_1, ident_2, groupby, 
            fc_thr=1.0, min_pct=0.0, max_pval=max_pval, 
            min_count=0, n_cores=n_cores, counts_layer=counts_layer
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


def _chi2_contingency_gpu(observed: torch.Tensor) -> torch.Tensor:
    """
    Vectorized chi-squared test for multiple 2x2 contingency tables on GPU.
    
    Parameters
    ----------
    observed : torch.Tensor
        Tensor of shape (n_genes, 2, 2) containing contingency tables
        
    Returns
    -------
    torch.Tensor
        P-values for each gene
    """
    # Sum along axes
    row_sums = observed.sum(dim=2, keepdim=True)  # (n_genes, 2, 1)
    col_sums = observed.sum(dim=1, keepdim=True)  # (n_genes, 1, 2)
    total = observed.sum(dim=(1, 2), keepdim=True)  # (n_genes, 1, 1)
    
    # Expected frequencies
    expected = (row_sums * col_sums) / total
    
    # Avoid division by zero
    expected = torch.clamp(expected, min=1e-10)
    
    # Chi-squared statistic
    chi2_stat = ((observed - expected) ** 2 / expected).sum(dim=(1, 2))
    
    # Degrees of freedom for 2x2 table is 1
    # Use chi-squared CDF approximation on GPU
    # For df=1, we can use the relationship with normal distribution
    # P(χ²(1) > x) = 2 * P(N(0,1) > √x)
    z = torch.sqrt(chi2_stat)
    
    # Complementary error function approximation for p-value
    # Using torch.special.erfc if available, otherwise approximate
    if hasattr(torch.special, 'erfc'):
        p_values = torch.special.erfc(z / np.sqrt(2))
    else:
        # Fallback: use torch distributions (slower)
        from torch.distributions import Chi2
        chi2_dist = Chi2(torch.tensor(1.0, device=z.device))
        p_values = 1 - chi2_dist.cdf(chi2_stat)
    
    return p_values


def chi2_test_gpu(adata, features=None, ident_1=None, ident_2=None, groupby='leiden', fc_thr=1.0, min_pct=0.0, max_pval=1.0, min_count=30, batch_size=1000, device=None, counts_layer=DEFAULT_COUNTS_LAYER):
    """
    GPU-accelerated chi-squared test for differential gene expression.
    Uses PyTorch for vectorized operations on GPU.
    Args:
        adata: AnnData object containing the scRNA-seq data.
        features: List of genes to analyze. If None, all genes are analyzed.
        ident_1: First identity/group for comparison.
        ident_2: Second identity/group for comparison.
        groupby: Column in adata.obs to use for grouping (default: 'leiden').
        fc_thr: Fold-change threshold for reporting results.
        min_pct: Minimum fraction of cells expressing the gene in at least one group.
        max_pval: Maximum p-value for reporting results.
        min_count: Minimum aggregate count in at least one group.
        batch_size: Number of genes to process per GPU batch (default: 1000).
        device: PyTorch device to use (e.g., 'cuda:0'). If None, uses default _DEVICE.
        counts_layer: Layer containing raw counts. If absent, falls back to .raw and then .X.
    Returns:
        pd.DataFrame with columns: log2FC, p.value.
    """
    if ident_1 is None or ident_2 is None:
        raise ValueError("Both ident_1 and ident_2 must be defined")
    
    if device is None:
        device = _DEVICE
    
    # Get gene list
    if features is None:
        gene_list = adata.var_names.tolist()
    else:
        gene_list = features
    
    # Get count matrix
    X, var_names = _get_count_matrix(adata, counts_layer)
    
    # Subset by identities
    mask_1 = (adata.obs[groupby] == ident_1).values
    mask_2 = (adata.obs[groupby] == ident_2).values
    
    Ci_1 = X[mask_1, :]
    Ci_2 = X[mask_2, :]
    
    # Convert to dense for GPU transfer (sparse not well supported on GPU for this)
    # For large matrices, process in chunks
    if sp.issparse(Ci_1):
        Ci_1_csc = Ci_1.tocsc()
        Ci_2_csc = Ci_2.tocsc()
    else:
        Ci_1_csc = Ci_1
        Ci_2_csc = Ci_2
    
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
    
    # Create gene name to index mapping and pre-filter valid genes
    gene_to_idx = {gene: idx for idx, gene in enumerate(var_names)}
    valid_genes = [(gene, gene_to_idx[gene]) for gene in gene_list if gene in gene_to_idx]
    
    if len(valid_genes) == 0:
        return pd.DataFrame(columns=['log2FC', 'p.value'])
    
    print(f"Performing chi^2 test on GPU ({device}):")
    
    results = []
    
    # Process in batches
    for batch_start in tqdm(range(0, len(valid_genes), batch_size)):
        batch_end = min(batch_start + batch_size, len(valid_genes))
        batch_genes = valid_genes[batch_start:batch_end]
        batch_indices = [idx for _, idx in batch_genes]
        batch_names = [name for name, _ in batch_genes]
        
        # Extract batch data
        if sp.issparse(Ci_1_csc):
            h1_batch_sparse = Ci_1_csc[:, batch_indices]
            h2_batch_sparse = Ci_2_csc[:, batch_indices]
            nonzero_1 = _sparse_csc_nonzero_counts(h1_batch_sparse)
            nonzero_2 = _sparse_csc_nonzero_counts(h2_batch_sparse)
        else:
            h1_batch = Ci_1_csc[:, batch_indices]
            h2_batch = Ci_2_csc[:, batch_indices]
            nonzero_1 = np.count_nonzero(h1_batch, axis=0)
            nonzero_2 = np.count_nonzero(h2_batch, axis=0)
        
        pct_1 = nonzero_1 / Nc_1
        pct_2 = nonzero_2 / Nc_2
        
        # Get aggregate counts for batch
        ac1_batch = AC_1[batch_indices]
        ac2_batch = AC_2[batch_indices]
        
        # Filter genes by criteria
        valid_mask = ((ac1_batch >= min_count) | (ac2_batch >= min_count)) & \
                     ((pct_1 > min_pct) | (pct_2 > min_pct)) & \
                     (ac2_batch > 0)
        
        if not valid_mask.any():
            continue
        
        # Compute fold changes
        fc_batch = (ac1_batch / (TC_1)+np.finfo(np.float32).tiny) / (ac2_batch / (TC_2)+np.finfo(np.float32).tiny)
        fc_mask = (fc_batch >= fc_thr) | (fc_batch <= 1/fc_thr)
        valid_mask = valid_mask & fc_mask
        
        if not valid_mask.any():
            continue
        
        # Filter to valid genes
        valid_idx = np.where(valid_mask)[0]
        ac1_valid = ac1_batch[valid_idx]
        ac2_valid = ac2_batch[valid_idx]
        fc_valid = fc_batch[valid_idx]
        
        # Build contingency tables for GPU
        cont_tables = np.stack([
            np.stack([TC_1 - ac1_valid, TC_2 - ac2_valid], axis=1),
            np.stack([ac1_valid, ac2_valid], axis=1)
        ], axis=1)
        
        # Transfer to GPU
        cont_tables_gpu = torch.from_numpy(cont_tables).float().to(device)
        
        # Compute p-values on GPU
        with torch.no_grad():
            p_values_gpu = _chi2_contingency_gpu(cont_tables_gpu)
            p_values = p_values_gpu.cpu().numpy()
        
        # Filter by max_pval and add results
        for i, p_val in enumerate(p_values):
            if p_val <= max_pval:
                orig_idx = valid_idx[i]
                results.append({
                    'gene': batch_names[orig_idx],
                    'log2FC': np.log2(fc_valid[i]+np.finfo(np.float32).tiny),
                    'p.value': float(p_val)
                })
    
    if len(results) == 0:
        return pd.DataFrame(columns=['log2FC', 'p.value'])
    
    output = pd.DataFrame(results)
    output.set_index('gene', inplace=True)
    
    return output


def chi2_test(adata, features=None, ident_1=None, ident_2=None, groupby='leiden', fc_thr=1.0, min_pct=0.0, max_pval=1.0, min_count=30, n_cores=1, counts_layer=DEFAULT_COUNTS_LAYER):
    """
    Perform chi-squared test for differential gene expression.
    Args:
        adata: AnnData object containing the scRNA-seq data.
        features: List of genes to analyze. If None, all genes are analyzed.
        ident_1: First identity/group for comparison.
        ident_2: Second identity/group for comparison.
        groupby: Column in adata.obs to use for grouping (default: 'leiden').
        fc_thr: Fold-change threshold for reporting results.
        min_pct: Minimum fraction of cells expressing the gene in at least one group.
        max_pval: Maximum p-value for reporting results.
        min_count: Minimum aggregate count in at least one group.
        n_cores: Number of CPU cores for parallel processing.
        counts_layer: Layer containing raw counts. If absent, falls back to .raw and then .X.
    Returns:
        pd.DataFrame with columns: log2FC, p.value.
    """
    if ident_1 is None or ident_2 is None:
        raise ValueError("Both ident_1 and ident_2 must be defined")
    
    # Get gene list
    if features is None:
        gene_list = adata.var_names.tolist()
    else:
        gene_list = features
    
    # Get count matrix
    X, var_names = _get_count_matrix(adata, counts_layer)
    
    # Subset by identities
    mask_1 = (adata.obs[groupby] == ident_1).values
    mask_2 = (adata.obs[groupby] == ident_2).values
    
    Ci_1 = X[mask_1, :]
    Ci_2 = X[mask_2, :]
    
    # Convert to CSC for efficient column access
    if sp.issparse(Ci_1):
        Ci_1 = Ci_1.tocsc()
        Ci_2 = Ci_2.tocsc()
    
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
    
    # Create gene name to index mapping and pre-filter valid genes
    gene_to_idx = {gene: idx for idx, gene in enumerate(var_names)}
    valid_genes = [(gene, gene_to_idx[gene]) for gene in gene_list if gene in gene_to_idx]
    
    if len(valid_genes) == 0:
        return pd.DataFrame(columns=['log2FC', 'p.value'])
    
    is_sparse = sp.issparse(Ci_1)
    
    def process_gene(args):
        gene_name, idx = args
        ac1 = AC_1[idx]
        ac2 = AC_2[idx]
        
        # Get gene expression for min_pct calculation
        if is_sparse:
            nonzero_1 = _sparse_column_nonzero_count(Ci_1, idx)
            nonzero_2 = _sparse_column_nonzero_count(Ci_2, idx)
        else:
            nonzero_1 = np.count_nonzero(Ci_1[:, idx])
            nonzero_2 = np.count_nonzero(Ci_2[:, idx])
        
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
                        'p.value': p_value
                    }
        return None
    
    print("Performing chi^2 test:")
    
    if n_cores > 1:
        results = Parallel(n_jobs=n_cores, backend='loky')(
            delayed(process_gene)(gene_args) for gene_args in tqdm(valid_genes)
        )
    else:
        results = [process_gene(args) for args in tqdm(valid_genes)]
    
    # Filter None results and create DataFrame
    results = [r for r in results if r is not None]
    
    if len(results) == 0:
        return pd.DataFrame(columns=['log2FC', 'p.value'])
    
    output = pd.DataFrame(results)
    output.set_index('gene', inplace=True)
    
    return output


def _sparse_column_view(matrix_csc, idx):
    """Return nonzero row indices and values for one CSC column."""
    start = matrix_csc.indptr[idx]
    end = matrix_csc.indptr[idx + 1]
    rows = matrix_csc.indices[start:end]
    values = matrix_csc.data[start:end]
    if values.size and np.any(values == 0):
        keep = values != 0
        rows = rows[keep]
        values = values[keep]
    return rows, values


def _sparse_column_nonzero_count(matrix_csc, idx):
    start = matrix_csc.indptr[idx]
    end = matrix_csc.indptr[idx + 1]
    values = matrix_csc.data[start:end]
    if values.size and np.any(values == 0):
        return int(np.count_nonzero(values))
    return int(end - start)


def _sparse_csc_nonzero_counts(matrix_csc):
    counts = np.diff(matrix_csc.indptr).astype(np.int64, copy=False)
    if matrix_csc.data.size and np.any(matrix_csc.data == 0):
        counts = np.array([
            np.count_nonzero(matrix_csc.data[matrix_csc.indptr[i]:matrix_csc.indptr[i + 1]])
            for i in range(matrix_csc.shape[1])
        ], dtype=np.int64)
    return counts


def _sparse_column_sum(matrix_csc, idx):
    start = matrix_csc.indptr[idx]
    end = matrix_csc.indptr[idx + 1]
    return matrix_csc.data[start:end].sum()


def _sparse_weighted_moments(matrix_csc, idx, n, weights):
    """Weighted mean and variance for x=h/n stored as one sparse count column."""
    weights = np.asarray(weights, dtype=np.float64)
    weight_sum = weights.sum()
    if weight_sum <= 0:
        raise ValueError("Weights must sum to a positive value")
    weights = weights / weight_sum
    total_w2 = np.square(weights).sum()
    var_denom = 1 - total_w2

    rows, values = _sparse_column_view(matrix_csc, idx)
    if rows.size:
        n_rows = n[rows]
        valid = n_rows > 0
        rows = rows[valid]
        values = values[valid]
        n_rows = n_rows[valid]
        x = values / n_rows
        w = weights[rows]
        w2 = np.square(w)
        mean = np.sum(x * w)
        sparse_x_w2 = np.sum(x * w2)
        sparse_x2_w2 = np.sum(np.square(x) * w2)
    else:
        mean = 0.0
        sparse_x_w2 = 0.0
        sparse_x2_w2 = 0.0

    if var_denom <= 0:
        variance = 0.0
    else:
        var_num = sparse_x2_w2 - 2 * mean * sparse_x_w2 + mean**2 * total_w2
        variance = max(var_num / var_denom, 0.0)
    n_eff = 1 / total_w2 if total_w2 > 0 else len(weights)
    return mean, variance, n_eff


def weighted_ttest_sparse_columns(Ci_1, Ci_2, idx, Ni_1, Ni_2, wi_1, wi_2, df_correction=False):
    """Compute weighted t-test from sparse count columns without materializing dense expression."""
    m1, vm1, n_eff_1 = _sparse_weighted_moments(Ci_1, idx, Ni_1, wi_1)
    m2, vm2, n_eff_2 = _sparse_weighted_moments(Ci_2, idx, Ni_2, wi_2)
    s12 = np.sqrt(vm1 + vm2)
    if s12 == 0:
        return m1, m2, 1.0

    t = (m1 - m2) / s12
    if df_correction:
        df_1 = max(n_eff_1 - 1, 1)
        df_2 = max(n_eff_2 - 1, 1)
    else:
        df_1 = max(Ci_1.shape[0] - 1, 1)
        df_2 = max(Ci_2.shape[0] - 1, 1)
    denom = vm1**2 / df_1 + vm2**2 / df_2
    df = s12**4 / denom if denom > 0 else 1
    p_value = 2 * stats.t.sf(np.abs(t), df=df)
    return m1, m2, p_value


def icc_an_sparse(matrix_csc, idx, n):
    """ANOVA ICC from one sparse count column without densifying the column."""
    n = np.asarray(n, dtype=np.float64)
    N = n.sum()
    k = len(n)
    if N <= 0 or k < 2:
        return 0.0

    rows, values = _sparse_column_view(matrix_csc, idx)
    valid = n[rows] > 0 if rows.size else np.array([], dtype=bool)
    rows = rows[valid]
    values = values[valid]
    n0 = (1 / (k - 1)) * (N - (np.square(n).sum() / N))
    h_sum = values.sum()
    h2_n_sum = np.sum(np.square(values) / n[rows]) if rows.size else 0.0
    MSw = (1 / (N - k)) * (h_sum - h2_n_sum) if N != k else 0.0
    MSb = (1 / (k - 1)) * (h2_n_sum - (h_sum**2 / N))
    denom = MSb + (n0 - 1) * MSw
    icc = 0.0 if denom == 0 else (MSb - MSw) / denom
    return np.clip(icc, 0.0, 1.0)


def icc_iter_sparse(matrix_csc, idx, n):
    """Iterative ICC from one sparse count column without densifying the column."""
    n = np.asarray(n, dtype=np.float64)
    sum_n = n.sum()
    if sum_n <= 0:
        return 0.0

    rows, values = _sparse_column_view(matrix_csc, idx)
    if rows.size:
        valid = n[rows] > 0
        rows = rows[valid]
        values = values[valid]

    sum_h = values.sum()
    sum_n_h = np.sum(n[rows] * values) if rows.size else 0.0
    sum_h2 = np.sum(np.square(values)) if rows.size else 0.0
    sum_n_sq = np.square(n).sum()
    x0 = sum_h / sum_n
    w0_sq_sum = sum_n_sq / sum_n**2
    var_denom = 1 - w0_sq_sum
    if var_denom <= 0:
        return 0.0

    VarT0 = x0 * (1 - x0) / sum_n
    VarE0 = (sum_h2 / sum_n**2 - 2 * x0 * sum_n_h / sum_n**2 + x0**2 * w0_sq_sum) / var_denom
    if VarE0 <= VarT0:
        return 0.0

    def f(icc):
        wprop = np.divide(
            n,
            1 + icc * (n - 1),
            out=np.zeros_like(n, dtype=np.float64),
            where=n > 0
        )
        sum_wprop = wprop.sum()
        if sum_wprop <= 0:
            return 0.0
        weights = wprop / sum_wprop
        total_w2 = np.square(weights).sum()
        if rows.size:
            x = values / n[rows]
            w = weights[rows]
            w2 = np.square(w)
            x1 = np.sum(x * w)
            sparse_x_w2 = np.sum(x * w2)
            sparse_x2_w2 = np.sum(np.square(x) * w2)
        else:
            x1 = 0.0
            sparse_x_w2 = 0.0
            sparse_x2_w2 = 0.0
        denom = 1 - total_w2
        if denom <= 0:
            return 0.0
        VarT = x1 * (1 - x1) / sum_wprop
        VarE = (sparse_x2_w2 - 2 * x1 * sparse_x_w2 + x1**2 * total_w2) / denom
        return VarE - VarT

    try:
        icc_val = brenth(f, 0, 1, xtol=1e-4 / max(n.max(), 1))
        return min(icc_val, 1.0)
    except ValueError:
        return 0.0


def icc_weight_sparse(matrix_csc, idx, n, icc='i'):
    """Calculate ICC weights from one sparse count column without densifying counts."""
    Nc = len(n)
    if matrix_csc.shape[0] != Nc:
        raise ValueError("Sparse matrix row count must match n length")
    if Nc < 3:
        raise ValueError("At least 3 observations are required")

    if _sparse_column_nonzero_count(matrix_csc, idx) < 3:
        return np.ones(Nc) / Nc

    if icc == 'i':
        icc_val = icc_iter_sparse(matrix_csc, idx, n)
    elif icc == 'A':
        icc_val = icc_an_sparse(matrix_csc, idx, n)
    elif icc in [0, 1]:
        icc_val = float(icc)
    else:
        raise ValueError("Invalid icc, must be 'i', 'A', 0, or 1")

    n = np.asarray(n, dtype=np.float64)
    wprop = np.divide(
        n,
        1 + icc_val * (n - 1),
        out=np.zeros_like(n, dtype=np.float64),
        where=n > 0
    )
    wprop_sum = wprop.sum()
    if wprop_sum <= 0:
        return np.ones(Nc) / Nc
    return wprop / wprop_sum


def _process_gene_weighted_ttest(args):
    """Helper function for parallel processing in iter_wght_ttest."""
    gene_name, idx, Ci_1, Ci_2, Xi_1, Xi_2, Ni_1, Ni_2, Nc_1, Nc_2, min_count, min_pct, fc_thr, max_pval, icc, df_correction, is_sparse = args
    
    # Get counts for this gene
    if is_sparse:
        AC_1 = _sparse_column_sum(Ci_1, idx)
        AC_2 = _sparse_column_sum(Ci_2, idx)
        nonzero_1 = _sparse_column_nonzero_count(Ci_1, idx)
        nonzero_2 = _sparse_column_nonzero_count(Ci_2, idx)
    else:
        h_1 = Ci_1[:, idx]
        h_2 = Ci_2[:, idx]
        xi_1 = Xi_1[:, idx]
        xi_2 = Xi_2[:, idx]
        nonzero_1 = np.count_nonzero(xi_1)
        nonzero_2 = np.count_nonzero(xi_2)
        AC_1 = h_1.sum()
        AC_2 = h_2.sum()
    
    pct_1 = nonzero_1 / Nc_1
    pct_2 = nonzero_2 / Nc_2
    
    if (AC_1 >= min_count or AC_2 >= min_count) and \
       (pct_1 > min_pct or pct_2 > min_pct):
        if is_sparse:
            wi_1 = icc_weight_sparse(Ci_1, idx, Ni_1, icc)
            wi_2 = icc_weight_sparse(Ci_2, idx, Ni_2, icc)
            mean_1, mean_2, p_value = weighted_ttest_sparse_columns(
                Ci_1, Ci_2, idx, Ni_1, Ni_2, wi_1, wi_2, df_correction
            )
            fc = mean_1 / mean_2 if mean_2 > 0 else np.nan
        else:
            wi_1 = icc_weight(h_1, Ni_1, icc)
            wi_2 = icc_weight(h_2, Ni_2, icc)
            fc = (xi_1 * wi_1).sum() / (xi_2 * wi_2).sum() if (xi_2 * wi_2).sum() > 0 else np.nan
        
        if not np.isnan(fc) and (fc >= fc_thr or fc <= 1/fc_thr) and \
           (nonzero_1 >= 3 or nonzero_2 >= 3):
            
            if not is_sparse:
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
                    'pct.2': pct_2
                }
    return None


def _weighted_ttest_gpu(x1_batch, x2_batch, w1_batch, w2_batch):
    """
    Helper function to compute weighted t-test statistics on GPU using PyTorch.
    Args:
        x1_batch: Tensor of shape (n_genes, n_cells_1) with expression values for group 1.
        x2_batch: Tensor of shape (n_genes, n_cells_2) with expression values for group 2.
        w1_batch: Tensor of shape (n_genes, n_cells_1) with weights for group 1.
        w2_batch: Tensor of shape (n_genes, n_cells_2) with weights for group 2.
    Returns:
        p_values: Tensor of shape (n_genes,) with p-values for each gene.
    """
    # Normalize weights per gene
    w1_sum = w1_batch.sum(dim=1, keepdim=True)
    w2_sum = w2_batch.sum(dim=1, keepdim=True)
    w1_norm = w1_batch / w1_sum
    w2_norm = w2_batch / w2_sum
    
    # Weighted means
    m1 = (x1_batch * w1_norm).sum(dim=1)
    m2 = (x2_batch * w2_norm).sum(dim=1)
    
    # Weighted variances
    w1_sq_sum = (w1_norm ** 2).sum(dim=1)
    w2_sq_sum = (w2_norm ** 2).sum(dim=1)
    
    vm1 = (w1_norm**2 * (x1_batch - m1.unsqueeze(1))**2).sum(dim=1) / (1 - w1_sq_sum)
    vm2 = (w2_norm**2 * (x2_batch - m2.unsqueeze(1))**2).sum(dim=1) / (1 - w2_sq_sum)
    
    # Standard error
    s12 = torch.sqrt(vm1 + vm2)
    s12 = torch.clamp(s12, min=1e-10)
    
    # T-statistic
    t = (m1 - m2) / s12
    
    # Degrees of freedom (approximate)
    df = x1_batch.shape[1] + x2_batch.shape[1] - 2
    
    # P-value using scipy.stats (CPU) for Student's t distribution
    t_cpu = t.detach().cpu().numpy()
    p_values_cpu = 2 * stats.t.sf(np.abs(t_cpu), df)
    p_values = torch.from_numpy(p_values_cpu).to(t.device, dtype=t.dtype)
    
    return p_values


def iter_wght_ttest_gpu(adata, features=None, ident_1=None, ident_2=None, groupby='leiden', fc_thr=1.0, min_pct=0.0, max_pval=1.0, min_count=30, icc='i', df_correction=False, batch_size=500, device=None, n_cores=1, icc_gpu=True, counts_layer=DEFAULT_COUNTS_LAYER):
    """
    GPU-accelerated weighted t-test with iterative weight calculation.
    Uses PyTorch for vectorized operations. ICC weights computed based on icc_gpu setting,
    and t-test statistics computed in batches on GPU for ~3-5x speedup.
    Args:
        adata: AnnData object containing the scRNA-seq data.
        features: List of genes to analyze. If None, all genes are analyzed.
        ident_1: First identity/group for comparison.
        ident_2: Second identity/group for comparison.
        groupby: Column in adata.obs to use for grouping (default: 'leiden').
        fc_thr: Fold-change threshold for reporting results.
        min_pct: Minimum fraction of cells expressing the gene in at least one group.
        max_pval: Maximum p-value for reporting results.
        min_count: Minimum aggregate count in at least one group.
        icc: Intracluster correlation coefficient method.
        df_correction: Apply correction for degrees of freedom.
        batch_size: Number of genes to process in parallel on GPU (default: 500).
        device: Device to use. If None, uses CUDA if available.
        n_cores: Number of CPU cores for parallel ICC weight computation (default: 1).
        icc_gpu: Use GPU for ICC weight computation (default: True). Set to False to force CPU.
        counts_layer: Layer containing raw counts. If absent, falls back to .raw and then .X.
    Returns:
        pd.DataFrame with columns: log2FC, p.value, p.value.adj, pct.1, pct.2.
    """
    if ident_1 is None or ident_2 is None:
        raise ValueError("Both ident_1 and ident_2 must be defined")
    
    if device is None:
        device = _DEVICE
    
    # Get gene list
    if features is None:
        gene_list = adata.var_names.tolist()
    else:
        gene_list = features
    
    # Get count matrix
    X, var_names = _get_count_matrix(adata, counts_layer)
    
    # Subset by identities
    mask_1 = (adata.obs[groupby] == ident_1).values
    mask_2 = (adata.obs[groupby] == ident_2).values
    
    Ci_1 = X[mask_1, :]
    Ci_2 = X[mask_2, :]
    
    # Convert to CSC format for faster column access
    if sp.issparse(Ci_1):
        Ci_1 = Ci_1.tocsc()
        Ci_2 = Ci_2.tocsc()
    
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
        Xi_1 = Xi_1.tocsc()
        Xi_2 = Xi_2.tocsc()
    else:
        Xi_1 = Ci_1 / Ni_1[:, np.newaxis]
        Xi_2 = Ci_2 / Ni_2[:, np.newaxis]
    
    # Create gene name to index mapping and filter genes that exist
    gene_to_idx = {gene: idx for idx, gene in enumerate(var_names)}
    valid_genes = [(gene, gene_to_idx[gene]) for gene in gene_list if gene in gene_to_idx]
    
    if len(valid_genes) == 0:
        return pd.DataFrame(columns=['log2FC', 'p.value', 'p.value.adj', 'pct.1', 'pct.2'])
    
    print(f"Performing weighted t-test on GPU ({device}):")
    
    is_sparse = sp.issparse(Ci_1)
    results = []
    
    # Process in batches
    for batch_start in tqdm(range(0, len(valid_genes), batch_size)):
        batch_end = min(batch_start + batch_size, len(valid_genes))
        batch_genes = valid_genes[batch_start:batch_end]
        batch_indices = [idx for _, idx in batch_genes]
        batch_names = [name for name, _ in batch_genes]
        
        # Extract batch data
        if is_sparse:
            h1_batch_sparse = Ci_1[:, batch_indices]
            h2_batch_sparse = Ci_2[:, batch_indices]
            nonzero_1 = np.diff(h1_batch_sparse.indptr)
            nonzero_2 = np.diff(h2_batch_sparse.indptr)
            AC_1_batch = np.asarray(h1_batch_sparse.sum(axis=0)).ravel()
            AC_2_batch = np.asarray(h2_batch_sparse.sum(axis=0)).ravel()
        else:
            h1_list = [Ci_1[:, idx] for idx in batch_indices]
            h2_list = [Ci_2[:, idx] for idx in batch_indices]
            xi1_list = [Xi_1[:, idx] for idx in batch_indices]
            xi2_list = [Xi_2[:, idx] for idx in batch_indices]

            # Stack into arrays
            h1_batch = np.stack(h1_list, axis=0)
            h2_batch = np.stack(h2_list, axis=0)
            xi1_batch = np.stack(xi1_list, axis=0)
            xi2_batch = np.stack(xi2_list, axis=0)

            # Compute statistics per gene
            nonzero_1 = np.count_nonzero(h1_batch, axis=1)
            nonzero_2 = np.count_nonzero(h2_batch, axis=1)
            AC_1_batch = h1_batch.sum(axis=1)
            AC_2_batch = h2_batch.sum(axis=1)
        pct_1 = nonzero_1 / Nc_1
        pct_2 = nonzero_2 / Nc_2
        
        # Filter by criteria
        valid_mask = ((AC_1_batch >= min_count) | (AC_2_batch >= min_count)) & \
                     ((pct_1 > min_pct) | (pct_2 > min_pct))
        
        if not valid_mask.any():
            continue
        
        valid_idx = np.where(valid_mask)[0]
        if is_sparse:
            valid_abs_indices = [batch_indices[i] for i in valid_idx]
            w1_list = None
            w2_list = None
            if icc_gpu and _CUPY_AVAILABLE and (icc == 'i' or icc in [0, 1]):
                try:
                    w1_list = compute_icc_weights_sparse_gpu(Ci_1, Ni_1, valid_abs_indices, icc, icc_gpu=icc_gpu)
                    w2_list = compute_icc_weights_sparse_gpu(Ci_2, Ni_2, valid_abs_indices, icc, icc_gpu=icc_gpu)
                except Exception:
                    w1_list = None
                    w2_list = None
            if w1_list is None or w2_list is None:
                w1_list = [icc_weight_sparse(Ci_1, idx, Ni_1, icc) for idx in valid_abs_indices]
                w2_list = [icc_weight_sparse(Ci_2, idx, Ni_2, icc) for idx in valid_abs_indices]
        else:
            # Compute ICC weights on CPU with parallelization
            valid_h1 = [h1_batch[i] for i in range(len(batch_genes)) if valid_mask[i]]
            valid_h2 = [h2_batch[i] for i in range(len(batch_genes)) if valid_mask[i]]
        
            w1_list = compute_icc_weights_parallel(valid_h1, Ni_1, icc, n_cores, icc_gpu=icc_gpu)
            w2_list = compute_icc_weights_parallel(valid_h2, Ni_2, icc, n_cores, icc_gpu=icc_gpu)
        
        if len(w1_list) == 0:
            continue
        if is_sparse:
            for i, abs_idx in enumerate(valid_abs_indices):
                mean_1, mean_2, p_val = weighted_ttest_sparse_columns(
                    Ci_1, Ci_2, abs_idx, Ni_1, Ni_2, w1_list[i], w2_list[i], df_correction
                )
                fc = mean_1 / mean_2 if mean_2 > 0 else np.nan
                orig_idx = valid_idx[i]
                if not np.isnan(fc) and (fc >= fc_thr or fc <= 1/fc_thr) and \
                   ((nonzero_1[orig_idx] >= 3) or (nonzero_2[orig_idx] >= 3)) and \
                   p_val <= max_pval:
                    results.append({
                        'gene': batch_names[orig_idx],
                        'log2FC': np.log2(fc + np.finfo(np.float32).tiny),
                        'p.value': float(p_val),
                        'pct.1': pct_1[orig_idx],
                        'pct.2': pct_2[orig_idx]
                    })
            continue
        
        # Get valid data
        xi1_valid = xi1_batch[valid_idx]
        xi2_valid = xi2_batch[valid_idx]
        w1_valid = np.stack(w1_list, axis=0)
        w2_valid = np.stack(w2_list, axis=0)
        
        # Compute fold changes
        fc_batch = (xi1_valid * w1_valid).sum(axis=1) / np.maximum((xi2_valid * w2_valid).sum(axis=1), 1e-10)
        fc_mask = (fc_batch >= fc_thr) | (fc_batch <= 1/fc_thr)
        fc_mask = fc_mask & ((nonzero_1[valid_idx] >= 3) | (nonzero_2[valid_idx] >= 3))
        
        if not fc_mask.any():
            continue
        
        # Final filtering
        fc_valid_idx = np.where(fc_mask)[0]
        xi1_final = xi1_valid[fc_valid_idx]
        xi2_final = xi2_valid[fc_valid_idx]
        w1_final = w1_valid[fc_valid_idx]
        w2_final = w2_valid[fc_valid_idx]
        fc_final = fc_batch[fc_valid_idx]
        
        # Transfer to GPU and compute p-values
        xi1_gpu = torch.from_numpy(xi1_final).float().to(device)
        xi2_gpu = torch.from_numpy(xi2_final).float().to(device)
        w1_gpu = torch.from_numpy(w1_final).float().to(device)
        w2_gpu = torch.from_numpy(w2_final).float().to(device)
        
        with torch.no_grad():
            if df_correction:
                # Use alt_wttest2 on CPU for df correction (less common)
                p_values = np.array([alt_wttest2(xi1_final[i], xi2_final[i], 
                                                 w1_final[i], w2_final[i]) 
                                    for i in range(len(xi1_final))])
            else:
                p_values_gpu = _weighted_ttest_gpu(xi1_gpu, xi2_gpu, w1_gpu, w2_gpu)
                p_values = p_values_gpu.cpu().numpy()
        
        # Add results
        for i, p_val in enumerate(p_values):
            if p_val <= max_pval:
                orig_idx = valid_idx[fc_valid_idx[i]]
                results.append({
                    'gene': batch_names[orig_idx],
                    'log2FC': np.log2(fc_final[i] + np.finfo(np.float32).tiny),
                    'p.value': float(p_val),
                    'pct.1': pct_1[orig_idx],
                    'pct.2': pct_2[orig_idx]
                })
    
    if len(results) == 0:
        return pd.DataFrame(columns=['log2FC', 'p.value', 'p.value.adj', 'pct.1', 'pct.2'])
    
    output = pd.DataFrame(results)
    output.set_index('gene', inplace=True)
    
    # Apply Benjamini-Hochberg correction to p-values
    if len(output) > 0:
        _, pvals_adj, _, _ = multipletests(output['p.value'].values, method='fdr_bh')
        output['p.value.adj'] = pvals_adj
    
    return output


def iter_wght_ttest(adata, features=None, ident_1=None, ident_2=None, groupby='leiden', fc_thr=1.0, min_pct=0.0, max_pval=1.0, min_count=30, icc='i', df_correction=False, n_cores=1, use_gpu=True, gpu_batch_size=500, icc_gpu=True, counts_layer=DEFAULT_COUNTS_LAYER):
    """
    Perform weighted t-test with iterative weight calculation.
    Args:
        adata: AnnData object containing the scRNA-seq data.
        features: List of genes to analyze. If None, all genes are analyzed.
        ident_1: First identity/group for comparison.
        ident_2: Second identity/group for comparison.
        groupby: Column in adata.obs to use for grouping (default: 'leiden').
        fc_thr: Fold-change threshold for reporting results.
        min_pct: Minimum fraction of cells expressing the gene in at least one group.
        max_pval: Maximum p-value for reporting results.
        min_count: Minimum aggregate count in at least one group.
        icc: Intracluster correlation coefficient method.
        df_correction: Apply correction for degrees of freedom.
        n_cores: Number of CPU cores for parallel processing.
        use_gpu: Use GPU acceleration if available (default: True).
        gpu_batch_size: Number of genes to process per GPU batch (default: 500).
        icc_gpu: Use GPU for ICC weight computation (default: True). Set to False to force CPU.
        counts_layer: Layer containing raw counts. If absent, falls back to .raw and then .X.
    Returns:
        pd.DataFrame with columns: log2FC, p.value, p.value.adj, pct.1, pct.2.
    """
    # Use GPU version if requested and available
    if use_gpu and _GPU_AVAILABLE:
        return iter_wght_ttest_gpu(
            adata, features, ident_1, ident_2, groupby, fc_thr, min_pct,
            max_pval, min_count, icc, df_correction, gpu_batch_size, device=None, n_cores=n_cores, icc_gpu=icc_gpu,
            counts_layer=counts_layer
        )
    
    # Fall back to CPU version
    if ident_1 is None or ident_2 is None:
        raise ValueError("Both ident_1 and ident_2 must be defined")
    
    # Get gene list
    if features is None:
        gene_list = adata.var_names.tolist()
    else:
        gene_list = features
    
    # Get count matrix
    X, var_names = _get_count_matrix(adata, counts_layer)
    
    # Subset by identities
    mask_1 = (adata.obs[groupby] == ident_1).values
    mask_2 = (adata.obs[groupby] == ident_2).values
    
    Ci_1 = X[mask_1, :]
    Ci_2 = X[mask_2, :]
    
    # Convert to CSC format for faster column access
    if sp.issparse(Ci_1):
        Ci_1 = Ci_1.tocsc()
        Ci_2 = Ci_2.tocsc()
    
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
        # Convert to CSC format for efficient column access (already done for Ci_1/Ci_2)
        Xi_1 = Xi_1.tocsc()
        Xi_2 = Xi_2.tocsc()
    else:
        Xi_1 = Ci_1 / Ni_1[:, np.newaxis]
        Xi_2 = Ci_2 / Ni_2[:, np.newaxis]
    
    # Create gene name to index mapping and filter genes that exist
    gene_to_idx = {gene: idx for idx, gene in enumerate(var_names)}
    valid_genes = [(gene, gene_to_idx[gene]) for gene in gene_list if gene in gene_to_idx]
    
    if len(valid_genes) == 0:
        return pd.DataFrame(columns=['log2FC', 'p.value', 'p.value.adj', 'pct.1', 'pct.2'])
    
    print("Performing weighted t-test:")
    
    # Check if we should use sparse or dense format
    is_sparse = sp.issparse(Ci_1)
    
    # Prepare arguments for parallel processing
    args_list = [(gene, idx, Ci_1, Ci_2, Xi_1, Xi_2, Ni_1, Ni_2, Nc_1, Nc_2, 
                  min_count, min_pct, fc_thr, max_pval, icc, df_correction, is_sparse) 
                 for gene, idx in valid_genes]
    
    if n_cores > 1:
        results = Parallel(n_jobs=n_cores, backend='loky')(
            delayed(_process_gene_weighted_ttest)(args) for args in tqdm(args_list)
        )
    else:
        results = [_process_gene_weighted_ttest(args) for args in tqdm(args_list)]
    
    # Filter None results and create DataFrame
    results = [r for r in results if r is not None]
    
    if len(results) == 0:
        return pd.DataFrame(columns=['log2FC', 'p.value', 'p.value.adj', 'pct.1', 'pct.2'])
    
    output = pd.DataFrame(results)
    output.set_index('gene', inplace=True)
    
    # Apply Benjamini-Hochberg correction to p-values
    if len(output) > 0:
        _, pvals_adj, _, _ = multipletests(output['p.value'].values, method='fdr_bh')
        output['p.value.adj'] = pvals_adj
    
    return output


def alt_wttest(x1, x2, w1, w2):
    """
    Alternative weighted t-test based on Margolin-Leikin variance estimator.
    Args:
        x1, x2: Data arrays for groups 1 and 2.
        w1, w2: Weight arrays for groups 1 and 2.
    Returns:
        P-value from weighted t-test.
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


def alt_wttest2(x1, x2, w1, w2):
    """
    Alternative weighted t-test with effective degrees of freedom correction.
    Args:
        x1, x2: Data arrays for groups 1 and 2.
        w1, w2: Weight arrays for groups 1 and 2.
    Returns:
        P-value from weighted t-test.
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
    Args:
        h: Count values.
        n: Total counts per observation.
    Returns:
        ICC value (clamped to [0, 1]).
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


def icc_iter(h, n):
    """
    Calculate iterative ICC providing more accurate variance matching.
    Args:
        h: Count values.
        n: Total counts per observation.
    Returns:
        ICC value (clamped to [0, 1]).
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
        icc_val = brenth(f, 0, 1, args=(x, n), xtol=1e-4/n.max())
        return min(icc_val, 1.0)
    except ValueError:
        return 0.0


def compute_icc_weights_parallel(h_list, n, icc, n_cores=1, icc_gpu=True):
    """
    Compute ICC weights for multiple genes, with GPU acceleration if available.
    Uses CuPy for GPU-accelerated batch processing when available and beneficial.
    Falls back to CPU sequential processing otherwise.
    Args:
        h_list: List of count arrays, one per gene.
        n: Total counts per observation (same for all genes).
        icc: ICC method: 'i' (iterative), 'A' (ANOVA), 0, or 1.
        n_cores: Kept for API compatibility (not used).
        icc_gpu: Use GPU for ICC computation if available (default: True).
    Returns:
        List of weight arrays, one per gene.
    """
    # Use GPU batch processing if enabled, CuPy is available, and we have enough genes
    # Custom CUDA kernel makes GPU worthwhile even for smaller batches
    if icc_gpu and _CUPY_AVAILABLE:
        try:
            if icc in [0, 1] and len(h_list) >= 10:
                return _compute_icc_weights_gpu_batch(h_list, n, icc)
            elif icc == 'i' and len(h_list) >= 50:
                return _compute_icc_weights_gpu_rootfinding(h_list, n, icc)
        except Exception:
            # Fallback to CPU if GPU processing fails
            pass
    
    # Sequential CPU processing
    return [icc_weight(h, n, icc) for h in h_list]


def compute_icc_weights_sparse_gpu(Ci_csc, n, gene_indices, icc, icc_gpu=True):
    """
    Compute ICC weights for sparse CSC count columns without densifying counts.
    Currently accelerates fixed ICC values and iterative ICC ('i') with CuPy.
    Falls back to the caller for unsupported modes.
    """
    if not gene_indices:
        return []
    if not sp.issparse(Ci_csc):
        raise ValueError("compute_icc_weights_sparse_gpu requires a sparse matrix")
    if not icc_gpu or not _CUPY_AVAILABLE:
        raise RuntimeError("CuPy GPU ICC is not available")

    n = np.asarray(n, dtype=np.float32)
    if icc in [0, 1]:
        return _compute_fixed_icc_weights_gpu(len(gene_indices), n, float(icc))
    if icc != 'i':
        raise ValueError("Sparse GPU ICC currently supports only icc='i', 0, or 1")

    batch_csc = Ci_csc[:, gene_indices].tocsc()
    return _compute_icc_weights_sparse_gpu_iterative(batch_csc, n)


def _compute_icc_weights_gpu_batch(h_list, n, icc_val):
    """
    GPU-accelerated batch computation of ICC weights for multiple genes.
    Optimized with cached n array and pinned memory for minimal I/O overhead.
    Args:
        h_list: List of count arrays, one per gene (genes x samples).
        n: Total counts per observation (same for all genes).
        icc_val: Fixed ICC value (0 or 1).
    Returns:
        List of weight arrays, one per gene.
    """
    return _compute_fixed_icc_weights_gpu(len(h_list), n, float(icc_val))
    
    
def _compute_fixed_icc_weights_gpu(n_genes, n, icc_val):
    """Return repeated fixed-ICC weights computed on GPU."""
    n_gpu = _get_cached_n_gpu(n)
    denom = 1 + float(icc_val) * (n_gpu - 1)
    denom = cp.where(n_gpu > 0, denom, 1.0)
    wprop = cp.where(n_gpu > 0, n_gpu / denom, 0.0)
    weights = wprop / wprop.sum()
    weights_cpu = cp.asnumpy(weights)
    return [weights_cpu.copy() for _ in range(n_genes)]


def _compute_icc_weights_gpu_rootfinding(h_list, n, icc_method):
    """
    GPU-accelerated batch root-finding for ICC computation across multiple genes.
    Optimized with cached n array, pinned memory, and minimized transfers.
    Args:
        h_list: List of count arrays, one per gene (genes x samples).
        n: Total counts per observation (same for all genes).
        icc_method: ICC method. Only 'i' is routed here; 'A' uses the CPU fallback.
    Returns:
        List of weight arrays, one per gene.
    """
    # Use cached n array on GPU (eliminates repeated transfers)
    n_hash = hash(n.tobytes())
    if _GPU_CACHE.get('n_hash') == n_hash:
        n_gpu = _GPU_CACHE['n_array']
    else:
        n_gpu = cp.asarray(n, dtype=cp.float32)
        _GPU_CACHE['n_array'] = n_gpu
        _GPU_CACHE['n_hash'] = n_hash
    
    # Stack and transfer h using contiguous array for optimal transfer speed
    h_stacked = np.ascontiguousarray(np.stack(h_list, axis=0), dtype=np.float32)
    h_matrix = cp.asarray(h_stacked)
    n_genes = h_matrix.shape[0]
    
    # Compute x = h/n for all genes
    x_matrix = h_matrix / n_gpu
    sum_n = n_gpu.sum()
    
    # Initial weights (same for all genes)
    w0 = n_gpu / sum_n
    w0_sq_sum = (w0**2).sum()
    
    # Compute initial values for each gene
    x0 = (x_matrix * w0).sum(axis=1)
    VarT0 = x0 * (1 - x0) / sum_n
    VarE0 = ((w0**2) * (x_matrix - x0[:, None])**2).sum(axis=1) / (1 - w0_sq_sum)
    
    # Initialize ICC values
    icc_vals = cp.zeros(n_genes, dtype=cp.float32)
    
    # Only solve for genes where VarE0 > VarT0
    needs_solving = VarE0 > VarT0
    
    if needs_solving.any():
        # Vectorized bisection for genes that need solving
        icc_vals[needs_solving] = _gpu_bisection_icc(
            x_matrix[needs_solving],
            n_gpu,
            xtol=1e-4 / n_gpu.max()
        )
        # Clamp to [0, 1]
        icc_vals = cp.clip(icc_vals, 0.0, 1.0)
    
    # Compute final weights for all genes
    wprop = n_gpu / (1 + icc_vals[:, None] * (n_gpu - 1))
    wprop_sum = wprop.sum(axis=1, keepdims=True)
    weights = wprop / wprop_sum
    
    # Transfer back to CPU efficiently and return list of views
    weights_cpu = cp.asnumpy(weights)
    return list(weights_cpu)


def _get_cached_n_gpu(n):
    """Cache total-count vectors on GPU across ICC calls."""
    n = np.asarray(n, dtype=np.float32)
    n_hash = hash(n.tobytes())
    if _GPU_CACHE.get('n_hash') == n_hash:
        return _GPU_CACHE['n_array']

    n_gpu = cp.asarray(n, dtype=cp.float32)
    _GPU_CACHE['n_array'] = n_gpu
    _GPU_CACHE['n_hash'] = n_hash
    return n_gpu


_SPARSE_ICC_BISECTION_KERNEL = r'''
__device__ float sparse_icc_eval_one(
    const long long start,
    const long long end,
    const int* indices,
    const float* data,
    const float* n_vals,
    const int n_cells,
    const float icc
) {
    float wprop_sum = 0.0f;
    float wprop_sq_sum = 0.0f;

    for (int i = 0; i < n_cells; i++) {
        const float n = n_vals[i];
        if (n <= 0.0f) continue;
        const float denom = 1.0f + icc * (n - 1.0f);
        if (denom <= 0.0f) continue;
        const float wprop = n / denom;
        wprop_sum += wprop;
        wprop_sq_sum += wprop * wprop;
    }

    if (wprop_sum <= 0.0f) return 0.0f;

    const float inv_wprop_sum = 1.0f / wprop_sum;
    const float w_sq_sum = wprop_sq_sum * inv_wprop_sum * inv_wprop_sum;
    const float var_e_denom = 1.0f - w_sq_sum;
    if (var_e_denom <= 1e-12f) return 0.0f;

    float x1 = 0.0f;
    float sparse_x_w2 = 0.0f;
    float sparse_x2_w2 = 0.0f;

    for (long long p = start; p < end; p++) {
        const float h = data[p];
        if (h == 0.0f) continue;
        const int row = indices[p];
        const float n = n_vals[row];
        if (n <= 0.0f) continue;
        const float denom = 1.0f + icc * (n - 1.0f);
        if (denom <= 0.0f) continue;
        const float wprop = n / denom;
        const float w = wprop * inv_wprop_sum;
        const float w2 = w * w;
        const float x = h / n;
        x1 += x * w;
        sparse_x_w2 += x * w2;
        sparse_x2_w2 += x * x * w2;
    }

    const float var_t = x1 * (1.0f - x1) / wprop_sum;
    const float var_e_num = sparse_x2_w2 - 2.0f * x1 * sparse_x_w2 + x1 * x1 * w_sq_sum;
    const float var_e = var_e_num / var_e_denom;
    return var_e - var_t;
}

extern "C" __global__
void sparse_icc_bisection_kernel(
    const long long* indptr,
    const int* indices,
    const float* data,
    const float* n_vals,
    float* icc_out,
    const int n_genes,
    const int n_cells,
    const float sum_n,
    const float sum_n_sq,
    const float xtol,
    const int max_iter
) {
    const int gene_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (gene_idx >= n_genes) return;

    const long long start = indptr[gene_idx];
    const long long end = indptr[gene_idx + 1];

    int nonzero_count = 0;
    float sum_h = 0.0f;
    float sum_n_h = 0.0f;
    float sum_h2 = 0.0f;

    for (long long p = start; p < end; p++) {
        const float h = data[p];
        if (h == 0.0f) continue;
        const int row = indices[p];
        const float n = n_vals[row];
        if (n <= 0.0f) continue;
        nonzero_count += 1;
        sum_h += h;
        sum_n_h += n * h;
        sum_h2 += h * h;
    }

    if (nonzero_count < 3) {
        icc_out[gene_idx] = -1.0f;
        return;
    }

    const float sum_n2 = sum_n * sum_n;
    const float x0 = sum_h / sum_n;
    const float w0_sq_sum = sum_n_sq / sum_n2;
    const float var_e0_denom = 1.0f - w0_sq_sum;
    if (var_e0_denom <= 1e-12f) {
        icc_out[gene_idx] = 0.0f;
        return;
    }

    const float sparse_x_w0_sq = sum_n_h / sum_n2;
    const float sparse_x2_w0_sq = sum_h2 / sum_n2;
    const float var_t0 = x0 * (1.0f - x0) / sum_n;
    const float var_e0_num = sparse_x2_w0_sq - 2.0f * x0 * sparse_x_w0_sq + x0 * x0 * w0_sq_sum;
    const float var_e0 = var_e0_num / var_e0_denom;
    float fa = var_e0 - var_t0;

    if (fa <= 0.0f || !isfinite(fa)) {
        icc_out[gene_idx] = 0.0f;
        return;
    }

    float fb = sparse_icc_eval_one(start, end, indices, data, n_vals, n_cells, 1.0f);
    if (!isfinite(fb) || fa * fb > 0.0f) {
        icc_out[gene_idx] = 0.0f;
        return;
    }

    float a = 0.0f;
    float b = 1.0f;

    for (int iter = 0; iter < max_iter; iter++) {
        const float c = 0.5f * (a + b);
        const float fc = sparse_icc_eval_one(start, end, indices, data, n_vals, n_cells, c);

        if (!isfinite(fc)) {
            break;
        }
        if (fa * fc > 0.0f) {
            a = c;
            fa = fc;
        } else {
            b = c;
        }
        if ((b - a) < xtol) {
            break;
        }
    }

    icc_out[gene_idx] = 0.5f * (a + b);
}
'''

_compiled_sparse_icc_kernel = None


def _get_compiled_sparse_icc_kernel():
    global _compiled_sparse_icc_kernel
    if _compiled_sparse_icc_kernel is None:
        _compiled_sparse_icc_kernel = cp.RawKernel(
            _SPARSE_ICC_BISECTION_KERNEL,
            'sparse_icc_bisection_kernel'
        )
    return _compiled_sparse_icc_kernel


def _compute_icc_weights_sparse_gpu_iterative(Ci_csc, n):
    """
    Compute iterative ICC weights from sparse CSC columns using a CuPy RawKernel.
    The sparse count matrix is never converted to a dense gene-by-cell matrix.
    """
    if not sp.isspmatrix_csc(Ci_csc):
        Ci_csc = Ci_csc.tocsc()

    n = np.asarray(n, dtype=np.float32)
    n_genes = Ci_csc.shape[1]
    n_cells = Ci_csc.shape[0]
    if n_genes == 0:
        return []
    if n_cells != n.shape[0]:
        raise ValueError("Sparse matrix row count must match n length")
    if n.sum() <= 0:
        raise ValueError("Total counts must be positive for sparse GPU ICC")

    indptr_gpu = cp.asarray(Ci_csc.indptr.astype(np.int64, copy=False))
    indices_gpu = cp.asarray(Ci_csc.indices.astype(np.int32, copy=False))
    data_gpu = cp.asarray(Ci_csc.data.astype(np.float32, copy=False))
    n_gpu = _get_cached_n_gpu(n)
    icc_vals = cp.empty(n_genes, dtype=cp.float32)

    block_size = 128
    grid_size = (n_genes + block_size - 1) // block_size
    xtol = np.float32(1e-4 / max(float(np.max(n)), 1.0))
    kernel = _get_compiled_sparse_icc_kernel()
    kernel(
        (grid_size,), (block_size,),
        (
            indptr_gpu,
            indices_gpu,
            data_gpu,
            n_gpu,
            icc_vals,
            np.int32(n_genes),
            np.int32(n_cells),
            np.float32(n.sum()),
            np.float32(np.square(n).sum()),
            xtol,
            np.int32(30),
        )
    )
    cp.cuda.Stream.null.synchronize()

    effective_icc = cp.maximum(icc_vals, 0.0)
    denom = 1 + effective_icc[:, None] * (n_gpu[None, :] - 1)
    denom = cp.where(n_gpu[None, :] > 0, denom, 1.0)
    wprop = cp.where(n_gpu[None, :] > 0, n_gpu[None, :] / denom, 0.0)
    weights = wprop / wprop.sum(axis=1, keepdims=True)

    equal_weight_mask = icc_vals < 0
    if bool(cp.any(equal_weight_mask)):
        weights[equal_weight_mask, :] = 1.0 / n_cells

    weights_cpu = cp.asnumpy(weights)
    return list(weights_cpu)


# Custom CUDA kernel for ultra-fast parallel bisection (using global memory)
_ICC_BISECTION_KERNEL = r'''
extern "C" __global__
void icc_bisection_kernel(
    const float* x_matrix,  // (n_genes, n_samples)
    const float* n_vals,    // (n_samples,)
    float* icc_out,         // (n_genes,)
    const int n_genes,
    const int n_samples,
    const float xtol,
    const int max_iter
) {
    int gene_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (gene_idx >= n_genes) return;
    
    const float* x = &x_matrix[gene_idx * n_samples];
    
    // Bisection bounds
    float a = 0.0f;
    float b = 0.5f;  // ICC rarely > 0.5
    
    // Bisection iterations
    for (int iter = 0; iter < max_iter; iter++) {
        float c = (a + b) * 0.5f;
        
        // Evaluate objective at a
        float sum_wprop_a = 0.0f;
        #pragma unroll 4
        for (int i = 0; i < n_samples; i++) {
            sum_wprop_a += n_vals[i] / (1.0f + a * (n_vals[i] - 1.0f));
        }
        float x1_a = 0.0f;
        float w_sq_sum_a = 0.0f;
        #pragma unroll 4
        for (int i = 0; i < n_samples; i++) {
            float w = (n_vals[i] / (1.0f + a * (n_vals[i] - 1.0f))) / sum_wprop_a;
            x1_a += x[i] * w;
            w_sq_sum_a += w * w;
        }
        float VarT_a = x1_a * (1.0f - x1_a) / sum_wprop_a;
        float VarE_a = 0.0f;
        #pragma unroll 4
        for (int i = 0; i < n_samples; i++) {
            float w = (n_vals[i] / (1.0f + a * (n_vals[i] - 1.0f))) / sum_wprop_a;
            float diff = x[i] - x1_a;
            VarE_a += w * w * diff * diff;
        }
        VarE_a /= (1.0f - w_sq_sum_a);
        float fa = VarE_a - VarT_a;
        
        // Evaluate objective at c
        float sum_wprop_c = 0.0f;
        #pragma unroll 4
        for (int i = 0; i < n_samples; i++) {
            sum_wprop_c += n_vals[i] / (1.0f + c * (n_vals[i] - 1.0f));
        }
        float x1_c = 0.0f;
        float w_sq_sum_c = 0.0f;
        #pragma unroll 4
        for (int i = 0; i < n_samples; i++) {
            float w = (n_vals[i] / (1.0f + c * (n_vals[i] - 1.0f))) / sum_wprop_c;
            x1_c += x[i] * w;
            w_sq_sum_c += w * w;
        }
        float VarT_c = x1_c * (1.0f - x1_c) / sum_wprop_c;
        float VarE_c = 0.0f;
        #pragma unroll 4
        for (int i = 0; i < n_samples; i++) {
            float w = (n_vals[i] / (1.0f + c * (n_vals[i] - 1.0f))) / sum_wprop_c;
            float diff = x[i] - x1_c;
            VarE_c += w * w * diff * diff;
        }
        VarE_c /= (1.0f - w_sq_sum_c);
        float fc = VarE_c - VarT_c;
        
        // Update bounds
        if (fa * fc > 0.0f) {
            a = c;
        } else {
            b = c;
        }
        
        // Check convergence
        if (b - a < xtol) break;
    }
    
    // Output result
    icc_out[gene_idx] = (a + b) * 0.5f;
}
'''

# Compile kernel on first use
_compiled_kernel = None

def _get_compiled_kernel():
    global _compiled_kernel
    if _compiled_kernel is None:
        _compiled_kernel = cp.RawKernel(_ICC_BISECTION_KERNEL, 'icc_bisection_kernel')
    return _compiled_kernel


def _gpu_bisection_icc(x_matrix, n_gpu, xtol=1e-6, max_iter=30):
    """
    Ultra-fast GPU bisection using custom CUDA kernel.
    Each gene gets its own GPU thread for true parallel processing.
    Args:
        x_matrix: Gene expression ratios (cupy array), shape (n_genes, n_samples).
        n_gpu: Total counts per sample (cupy array), shape (n_samples,).
        xtol: Tolerance for convergence.
        max_iter: Maximum iterations.
    Returns:
        ICC values for each gene (cupy array), shape (n_genes,).
    """
    n_genes, n_samples = x_matrix.shape
    
    # Output array
    icc_out = cp.empty(n_genes, dtype=cp.float32)
    
    # Optimal block size for most GPUs
    block_size = 128
    grid_size = (n_genes + block_size - 1) // block_size
    
    try:
        kernel = _get_compiled_kernel()
        kernel(
            (grid_size,), (block_size,),
            (x_matrix, n_gpu, icc_out, n_genes, n_samples, xtol, max_iter)
        )
        cp.cuda.Stream.null.synchronize()  # Ensure kernel completes
    except Exception:
        # Fallback to Python implementation if kernel fails
        return _gpu_bisection_icc_fallback(x_matrix, n_gpu, xtol, max_iter)
    
    return icc_out


def _gpu_bisection_icc_fallback(x_matrix, n_gpu, xtol=1e-6, max_iter=30):
    """
    Fallback pure CuPy implementation if custom kernel fails.
    Args:
        x_matrix: Gene expression ratios (cupy array), shape (n_genes, n_samples).
        n_gpu: Total counts per sample (cupy array), shape (n_samples,).
        xtol: Tolerance for convergence.
        max_iter: Maximum iterations.
    Returns:
        ICC values for each gene (cupy array), shape (n_genes,).
    """
    n_genes = x_matrix.shape[0]
    n_minus_1 = n_gpu - 1
    a = cp.zeros(n_genes, dtype=cp.float32)
    b = cp.full(n_genes, 0.5, dtype=cp.float32)
    
    for iteration in range(max_iter):
        c = (a + b) * 0.5
        
        # Eval at a
        wprop_a = n_gpu / (1 + a[:, None] * n_minus_1)
        sum_wprop_a = wprop_a.sum(axis=1, keepdims=True)
        w_a = wprop_a / sum_wprop_a
        x1_a = (x_matrix * w_a).sum(axis=1)
        VarT_a = x1_a * (1 - x1_a) / sum_wprop_a.squeeze()
        w_sq_a = w_a * w_a
        VarE_a = (w_sq_a * (x_matrix - x1_a[:, None])**2).sum(axis=1) / (1 - w_sq_a.sum(axis=1))
        fa = VarE_a - VarT_a
        
        # Eval at c
        wprop_c = n_gpu / (1 + c[:, None] * n_minus_1)
        sum_wprop_c = wprop_c.sum(axis=1, keepdims=True)
        w_c = wprop_c / sum_wprop_c
        x1_c = (x_matrix * w_c).sum(axis=1)
        VarT_c = x1_c * (1 - x1_c) / sum_wprop_c.squeeze()
        w_sq_c = w_c * w_c
        VarE_c = (w_sq_c * (x_matrix - x1_c[:, None])**2).sum(axis=1) / (1 - w_sq_c.sum(axis=1))
        fc = VarE_c - VarT_c
        
        same_sign = (fa * fc) > 0
        a = cp.where(same_sign, c, a)
        b = cp.where(same_sign, b, c)
        
        if cp.all((b - a) < xtol):
            break
    
    return (a + b) * 0.5


def icc_weight(h, n, icc='i'):
    """
    Calculate statistical weights based on ICC.
    Args:
        h: Count values.
        n: Total counts per observation.
        icc: ICC method: 'i' (iterative), 'A' (ANOVA), 0, or 1.
    Returns:
        Normalized weights.
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
