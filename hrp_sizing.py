"""
HRP (Hierarchical Risk Parity) Position Sizing + PCA Eigen Portfolio Detection.
From "Machine Learning for Algorithmic Trading" ch13 by Stefan Jansen.

HRP: Cluster assets by correlation, allocate inversely to cluster variance.
     Prevents overconcentration in correlated coins.
PCA: Detect market regimes via eigenportfolio behavior.
     PC1 > 50% = high correlation regime (risk-on/risk-off dominates).
"""

import numpy as np
from collections import defaultdict
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import logging

log = logging.getLogger(__name__)


def get_hrp_weights(returns_dict: dict, lookback: int = 60) -> dict:
    """
    Compute Hierarchical Risk Parity weights for a set of coins.
    
    Args:
        returns_dict: {coin: np.array of log returns (1D, length >= lookback)}
        lookback: number of return periods to use
        
    Returns:
        {coin: weight} where weights sum to 1.0
    """
    coins = list(returns_dict.keys())
    n = len(coins)
    
    if n < 2:
        return {coins[0]: 1.0} if coins else {}
    
    # Build returns matrix and covariance
    rets = np.column_stack([returns_dict[c][-lookback:] for c in coins])
    cov = np.cov(rets.T)
    
    # Correlation distance matrix
    corr = np.corrcoef(rets.T)
    dist = np.sqrt(0.5 * (1 - corr))
    
    # Hierarchical clustering
    condensed_dist = squareform(dist, checks=False)
    link = linkage(condensed_dist, method='single')
    
    # Quasi-diagonalization: sort assets by cluster distance
    sort_idx = _quasi_diagonalize(link, n)
    sorted_coins = [coins[i] for i in sort_idx]
    
    # Recursive bisection: allocate inversely to cluster variance
    weights = pd_series_hrp(cov, sorted_coins, sort_idx)
    
    result = {}
    for coin, w in zip(sorted_coins, weights):
        result[coin] = float(w)
    
    return result


def _quasi_diagonalize(link: np.ndarray, n_items: int) -> list:
    """Sort clustered items by distance (quasi-diagonalization)."""
    link = link.astype(int)
    sort_idx = [link[-1, 0], link[-1, 1]]
    num_items = link[-1, 3]
    
    while max(sort_idx) >= num_items:
        sort_idx_new = []
        for idx in sort_idx:
            if idx >= num_items:
                sort_idx_new.extend([link[idx - num_items, 0], link[idx - num_items, 1]])
            else:
                sort_idx_new.append(idx)
        sort_idx = sort_idx_new
    
    return sort_idx


def pd_series_hrp(cov: np.ndarray, sorted_coins: list, sort_idx: list = None) -> np.ndarray:
    """Compute HRP weights via recursive bisection."""
    n = len(sorted_coins)
    weights = np.ones(n)
    clusters = [list(range(n))]
    
    while clusters:
        new_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            # Split cluster in half
            mid = len(cluster) // 2
            left = cluster[:mid]
            right = cluster[mid:]
            
            if len(left) >= 1 and len(right) >= 1:
                # Inverse variance weighting between sub-clusters
                var_left = _cluster_variance(cov, left)
                var_right = _cluster_variance(cov, right)
                
                alpha = 1 - var_left / (var_left + var_right + 1e-10)
                
                # Scale weights within each sub-cluster
                weights[left] *= alpha
                weights[right] *= (1 - alpha)
                
                new_clusters.append(left)
                new_clusters.append(right)
            elif left:
                new_clusters.append(left)
            elif right:
                new_clusters.append(right)
        
        clusters = new_clusters
    
    # Normalize to sum to 1
    weights /= weights.sum()
    return weights


def _cluster_variance(cov: np.ndarray, cluster_indices: list) -> float:
    """Compute minimum variance portfolio variance for a cluster."""
    cov_sub = cov[np.ix_(cluster_indices, cluster_indices)]
    # Inverse variance weights
    ivp = 1.0 / np.diag(cov_sub)
    ivp /= ivp.sum()
    return float(ivp @ cov_sub @ ivp)


# ── PCA Eigen Portfolio Analysis ──

class EigenRegime:
    """PCA-based market regime detection from eigen portfolio behavior."""
    
    def __init__(self, lookback: int = 60):
        self.lookback = lookback
        self.pc1_ratio = 0.0  # % variance explained by PC1
        self.regime = "neutral"  # high_corr, normal, fragmented
        self.eigen_weights = {}  # {coin: weight} for top eigen portfolio
        
    def update(self, returns_dict: dict, top_n: int = 30):
        """
        Update eigen portfolio analysis.
        
        Args:
            returns_dict: {coin: np.array of returns}
            top_n: use top N coins by volume
        """
        coins = list(returns_dict.keys())[:top_n]
        if len(coins) < 5:
            return
        
        # Build returns matrix
        rets = np.column_stack([returns_dict[c][-self.lookback:] for c in coins])
        cov = np.cov(rets.T)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        eigenvalues = eigenvalues[::-1]  # Descending
        eigenvectors = eigenvectors[:, ::-1]
        
        # PC1 explained variance ratio
        total_var = eigenvalues.sum()
        if total_var > 0:
            self.pc1_ratio = eigenvalues[0] / total_var
        
        # Regime classification
        if self.pc1_ratio > 0.55:
            self.regime = "high_correlation"  # CryptoGAT: PC1=55% in crypto
        elif self.pc1_ratio > 0.30:
            self.regime = "normal"
        else:
            self.regime = "fragmented"
        
        # Top eigen portfolio weights (market factor)
        top_eigen = eigenvectors[:, 0]
        top_eigen = top_eigen / top_eigen.sum()  # Normalize
        self.eigen_weights = {coins[i]: float(top_eigen[i]) 
                              for i in range(len(coins))}


# Singleton
eigen_regime = EigenRegime()
