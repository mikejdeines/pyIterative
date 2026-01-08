# pyIterative
A package for iterative clustering based on CONCORD and weighted t-tests.

To install:
```
git clone https://github.com/mikejdeines/pyIterative
cd pyIterative
pip install .
```

All cells begin in the same cluster.

The following steps are performed on each cluster for each iteration:

1. Select 2000 HVGs using scanpy's seurat_v3 method (fallback on seurat method).
2. Integrate and project into a low-rank approximation of size ndims using CONCORD.
3. Create a shared nearest-neighbors graph with connectivities equal to the Jaccard similarity of kNN.
4. Perform Leiden clustering at a resolution of 1.0.
5. Calculate the centroids of the clusters in the CONCORD latent space.
6. Find differentially-expressed genes between the clusters with the smallest Euclidean distance between centroids.
7. If the clusters do not have a DE score > min_score or one cluster is smaller than min_cluster_size, merge them together.
8. Repeat until each cluster has been compared to its closest neighbor.

Clustering along a branch is terminated if the cluster size is <= min_cluster_size or if CONCORD fails.

Clustering iterations are performed until no new clusters are found.

After the final iteration, all clusters are checked to make sure they are seperable by DE score.

If you find this package useful, please cite the following papers:
Margolin, G. et al. 2025.10.20.683496 (2025)
Zhu, Q. et al. Nat Biotechnol 1–15 (2026)
