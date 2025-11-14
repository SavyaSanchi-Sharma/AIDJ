"""
Feature selection and dimensionality reduction for music features.
Implements statistical filtering, relevance scoring, and ML-based selection.
"""

import numpy as np
import pandas as pd
import structlog
from typing import Dict, List, Optional, Tuple, Any, Set
from sklearn.feature_selection import (
    VarianceThreshold, 
    SelectKBest, 
    f_regression, 
    mutual_info_regression,
    RFE,
    SelectFromModel
)
from sklearn.decomposition import PCA, FastICA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.manifold import TSNE
import pickle
from dataclasses import dataclass
from pathlib import Path

logger = structlog.get_logger(__name__)


@dataclass
class FeatureSelectionConfig:
    """Configuration for feature selection pipeline."""
    
    # Statistical filtering
    variance_threshold: float = 0.01
    correlation_threshold: float = 0.95
    
    # DJ relevance scoring
    dj_weight: float = 0.3
    technical_weight: float = 0.4
    metadata_weight: float = 0.3
    
    # Dimensionality reduction
    target_dimensions: int = 512
    pca_components: int = 256
    ica_components: int = 128
    
    # Feature selection methods
    k_best_features: int = 800
    recursive_elimination: bool = True
    lasso_alpha: float = 0.1
    
    # Output settings
    save_importance_scores: bool = True
    feature_names_file: str = "selected_features.json"


class DJRelevanceScorer:
    """
    Scores features based on their relevance to DJ decision-making.
    Uses domain knowledge to weight different feature categories.
    """
    
    def __init__(self):
        """Initialize with DJ-specific importance weights."""
        
        # DJ decision factors (based on DJ interviews and studies)
        self.dj_importance_weights = {
            # Critical for mixing
            'tempo': 1.0,
            'key': 1.0, 
            'energy': 0.95,
            'camelot': 1.0,
            'bpm': 1.0,
            
            # Very important for track selection
            'genre': 0.85,
            'mood': 0.8,
            'danceability': 0.9,
            'valence': 0.75,
            'loudness': 0.7,
            
            # Important for harmonic mixing
            'chroma': 0.8,
            'tonnetz': 0.75,
            'harmonic': 0.8,
            'key_clarity': 0.85,
            'tonal_stability': 0.7,
            
            # Rhythm and timing
            'beat': 0.9,
            'onset': 0.8,
            'syncopation': 0.6,
            'rhythmic': 0.75,
            'tempo_stability': 0.8,
            
            # Energy and dynamics
            'rms': 0.7,
            'spectral_centroid': 0.6,
            'spectral_rolloff': 0.55,
            'dynamic_range': 0.65,
            'attack': 0.5,
            'decay': 0.45,
            
            # Audience response factors
            'popularity': 0.6,
            'release_age': 0.4,
            'chart_position': 0.5,
            'social_metrics': 0.45,
            
            # Technical quality
            'bitrate': 0.3,
            'sample_rate': 0.25,
            'audio_quality': 0.35,
            
            # Less important for real-time DJing
            'mfcc': 0.3,
            'spectral_bandwidth': 0.25,
            'zero_crossing': 0.2,
            'spectral_flatness': 0.2,
            'irregularity': 0.15,
            
            # Generally not used in DJ decisions
            'phase': 0.1,
            'kurtosis': 0.1,
            'skewness': 0.1,
            'file_size': 0.05,
            'creation_time': 0.05
        }
        
        # Category-based weights
        self.category_weights = {
            'rhythm_comprehensive': 0.9,
            'harmonic_advanced': 0.85,
            'energy_comprehensive': 0.8,
            'essentia_comprehensive': 0.75,
            'spotify': 0.7,
            'multi_resolution_spectral': 0.4,
            'psychoacoustic': 0.3,
            'structural': 0.5,
            'timbral_complexity': 0.35,
            'statistical': 0.2,
            'metadata': 0.6,
            'derived': 0.55
        }
        
        logger.info("DJRelevanceScorer initialized", 
                   importance_weights=len(self.dj_importance_weights))
    
    def score_feature(self, feature_name: str, category: str = "") -> float:
        """
        Score a single feature's relevance to DJ decision-making.
        
        Args:
            feature_name: Name of the feature
            category: Feature category (optional)
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        
        # Start with category-based score
        category_score = self.category_weights.get(category, 0.5)
        
        # Look for keyword matches in feature name
        feature_lower = feature_name.lower()
        keyword_score = 0.0
        
        for keyword, weight in self.dj_importance_weights.items():
            if keyword in feature_lower:
                keyword_score = max(keyword_score, weight)
        
        # Combine scores (weighted average)
        if keyword_score > 0:
            final_score = 0.7 * keyword_score + 0.3 * category_score
        else:
            final_score = category_score
        
        return min(final_score, 1.0)
    
    def score_feature_vector(self, feature_names: List[str], 
                            feature_categories: Optional[Dict[str, str]] = None) -> np.ndarray:
        """
        Score an entire feature vector.
        
        Args:
            feature_names: List of feature names
            feature_categories: Optional mapping of feature names to categories
            
        Returns:
            Array of relevance scores
        """
        
        scores = []
        categories = feature_categories or {}
        
        for feature_name in feature_names:
            category = categories.get(feature_name, "")
            score = self.score_feature(feature_name, category)
            scores.append(score)
        
        return np.array(scores)


class ComprehensiveFeatureSelector:
    """
    Comprehensive feature selection pipeline for music analysis.
    
    Combines multiple selection strategies:
    1. Statistical filtering (variance, correlation)
    2. Domain knowledge (DJ relevance scoring)
    3. ML-based selection (feature importance, mutual information)
    4. Dimensionality reduction (PCA, ICA)
    """
    
    def __init__(self, config: FeatureSelectionConfig):
        """Initialize feature selector with configuration."""
        
        self.config = config
        self.dj_scorer = DJRelevanceScorer()
        self.scaler = StandardScaler()
        
        # Selection components
        self.variance_selector = None
        self.correlation_selector = None
        self.k_best_selector = None
        self.recursive_selector = None
        self.lasso_selector = None
        
        # Dimensionality reduction
        self.pca = None
        self.ica = None
        
        # State tracking
        self.is_fitted = False
        self.selected_feature_names = []
        self.feature_importance_scores = {}
        self.selection_history = []
        
        logger.info("ComprehensiveFeatureSelector initialized", 
                   target_dimensions=config.target_dimensions)
    
    def fit_select(self, 
                   X: np.ndarray, 
                   y: Optional[np.ndarray] = None,
                   feature_names: Optional[List[str]] = None,
                   feature_categories: Optional[Dict[str, str]] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Fit selectors and transform features in one step.
        
        Args:
            X: Feature matrix (samples x features)
            y: Optional target values for supervised selection
            feature_names: Names of features
            feature_categories: Feature category mapping
            
        Returns:
            Tuple of (selected_features, selected_feature_names)
        """
        
        logger.info("Starting comprehensive feature selection", 
                   input_features=X.shape[1], 
                   target_dimensions=self.config.target_dimensions)
        
        # Prepare feature names and categories
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        current_features = X.copy()
        current_names = feature_names.copy()
        
        # Step 1: Statistical filtering
        current_features, current_names = self._apply_statistical_filtering(
            current_features, current_names)
        
        # Step 2: DJ relevance scoring
        current_features, current_names = self._apply_dj_relevance_filtering(
            current_features, current_names, feature_categories)
        
        # Step 3: ML-based feature selection (if target provided)
        if y is not None:
            current_features, current_names = self._apply_ml_feature_selection(
                current_features, current_names, y)
        
        # Step 4: Dimensionality reduction
        current_features, current_names = self._apply_dimensionality_reduction(
            current_features, current_names)
        
        # Store results
        self.selected_feature_names = current_names
        self.is_fitted = True
        
        # Save feature importance if requested
        if self.config.save_importance_scores:
            self._save_feature_importance()
        
        logger.info("Feature selection completed", 
                   final_features=current_features.shape[1],
                   reduction_ratio=f"{X.shape[1]}→{current_features.shape[1]} "
                                  f"({current_features.shape[1]/X.shape[1]:.2%})")
        
        return current_features, current_names
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform new data using fitted selectors.
        
        Args:
            X: Feature matrix to transform
            
        Returns:
            Transformed feature matrix
        """
        
        if not self.is_fitted:
            raise ValueError("FeatureSelector must be fitted before transform")
        
        current_features = X.copy()
        
        # Apply all transformations in order
        if self.variance_selector:
            current_features = self.variance_selector.transform(current_features)
        
        if self.correlation_selector:
            current_features = current_features[:, self.correlation_selector]
        
        if self.k_best_selector:
            current_features = self.k_best_selector.transform(current_features)
        
        if self.recursive_selector:
            current_features = self.recursive_selector.transform(current_features)
        
        if self.lasso_selector:
            current_features = self.lasso_selector.transform(current_features)
        
        # Scale before dimensionality reduction
        current_features = self.scaler.transform(current_features)
        
        if self.pca:
            pca_features = self.pca.transform(current_features)
        else:
            pca_features = current_features
        
        if self.ica:
            ica_features = self.ica.transform(current_features)
            # Combine PCA and ICA features
            current_features = np.hstack([pca_features, ica_features])
        else:
            current_features = pca_features
        
        return current_features
    
    def _apply_statistical_filtering(self, X: np.ndarray, 
                                   feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Apply statistical filtering (variance and correlation)."""
        
        logger.info("Applying statistical filtering", features=X.shape[1])
        
        # Variance threshold filtering
        self.variance_selector = VarianceThreshold(threshold=self.config.variance_threshold)
        X_var = self.variance_selector.fit_transform(X)
        
        # Update feature names
        variance_mask = self.variance_selector.get_support()
        names_after_variance = [name for i, name in enumerate(feature_names) if variance_mask[i]]
        
        # Correlation filtering
        correlation_indices = self._remove_correlated_features(X_var, names_after_variance)
        X_corr = X_var[:, correlation_indices]
        names_after_corr = [names_after_variance[i] for i in correlation_indices]
        
        self.correlation_selector = correlation_indices
        
        removed_count = X.shape[1] - X_corr.shape[1]
        self.selection_history.append({
            'step': 'statistical_filtering',
            'features_before': X.shape[1],
            'features_after': X_corr.shape[1],
            'features_removed': removed_count
        })
        
        logger.info("Statistical filtering completed", 
                   features_removed=removed_count,
                   features_remaining=X_corr.shape[1])
        
        return X_corr, names_after_corr
    
    def _apply_dj_relevance_filtering(self, X: np.ndarray, 
                                    feature_names: List[str],
                                    feature_categories: Optional[Dict[str, str]]) -> Tuple[np.ndarray, List[str]]:
        """Apply DJ relevance scoring and filtering."""
        
        logger.info("Applying DJ relevance filtering", features=X.shape[1])
        
        # Calculate relevance scores
        relevance_scores = self.dj_scorer.score_feature_vector(feature_names, feature_categories)
        
        # Store importance scores
        for name, score in zip(feature_names, relevance_scores):
            self.feature_importance_scores[name] = {
                'dj_relevance': float(score),
                'category': feature_categories.get(name, 'unknown') if feature_categories else 'unknown'
            }
        
        # Select top features based on relevance + random sampling of lower-scored features
        high_relevance_threshold = 0.6
        high_relevance_mask = relevance_scores >= high_relevance_threshold
        
        # Always keep high-relevance features
        high_relevance_indices = np.where(high_relevance_mask)[0]
        
        # Randomly sample from lower-relevance features
        low_relevance_indices = np.where(~high_relevance_mask)[0]
        low_relevance_scores = relevance_scores[low_relevance_indices]
        
        # Weighted sampling based on scores
        remaining_slots = self.config.k_best_features - len(high_relevance_indices)
        if remaining_slots > 0 and len(low_relevance_indices) > 0:
            sampling_probs = low_relevance_scores / np.sum(low_relevance_scores)
            sampled_count = min(remaining_slots, len(low_relevance_indices))
            
            sampled_indices = np.random.choice(
                low_relevance_indices,
                size=sampled_count,
                replace=False,
                p=sampling_probs
            )
        else:
            sampled_indices = []
        
        # Combine selected indices
        selected_indices = np.concatenate([high_relevance_indices, sampled_indices])
        
        X_relevance = X[:, selected_indices]
        names_relevance = [feature_names[i] for i in selected_indices]
        
        removed_count = X.shape[1] - X_relevance.shape[1]
        self.selection_history.append({
            'step': 'dj_relevance_filtering',
            'features_before': X.shape[1],
            'features_after': X_relevance.shape[1],
            'features_removed': removed_count,
            'high_relevance_kept': len(high_relevance_indices),
            'low_relevance_sampled': len(sampled_indices)
        })
        
        logger.info("DJ relevance filtering completed",
                   features_removed=removed_count,
                   high_relevance_kept=len(high_relevance_indices),
                   low_relevance_sampled=len(sampled_indices))
        
        return X_relevance, names_relevance
    
    def _apply_ml_feature_selection(self, X: np.ndarray, 
                                  feature_names: List[str],
                                  y: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Apply ML-based feature selection methods."""
        
        logger.info("Applying ML-based feature selection", features=X.shape[1])
        
        current_features = X.copy()
        current_names = feature_names.copy()
        
        # K-best selection using mutual information
        k_best = min(self.config.k_best_features, current_features.shape[1])
        self.k_best_selector = SelectKBest(score_func=mutual_info_regression, k=k_best)
        X_kbest = self.k_best_selector.fit_transform(current_features, y)
        
        kbest_mask = self.k_best_selector.get_support()
        names_kbest = [name for i, name in enumerate(current_names) if kbest_mask[i]]
        
        # Recursive feature elimination with Random Forest
        if self.config.recursive_elimination and X_kbest.shape[1] > self.config.target_dimensions:
            rf_estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            n_features_to_select = min(self.config.target_dimensions, X_kbest.shape[1])
            
            self.recursive_selector = RFE(estimator=rf_estimator, 
                                        n_features_to_select=n_features_to_select)
            X_rfe = self.recursive_selector.fit_transform(X_kbest, y)
            
            rfe_mask = self.recursive_selector.get_support()
            names_rfe = [name for i, name in enumerate(names_kbest) if rfe_mask[i]]
            
            current_features = X_rfe
            current_names = names_rfe
        else:
            current_features = X_kbest
            current_names = names_kbest
        
        # L1-based feature selection (Lasso)
        if current_features.shape[1] > self.config.target_dimensions:
            lasso = LassoCV(cv=5, random_state=42, n_jobs=-1, max_iter=1000)
            lasso.fit(current_features, y)
            
            self.lasso_selector = SelectFromModel(lasso, prefit=True, threshold='median')
            X_lasso = self.lasso_selector.transform(current_features)
            
            lasso_mask = self.lasso_selector.get_support()
            names_lasso = [name for i, name in enumerate(current_names) if lasso_mask[i]]
            
            current_features = X_lasso
            current_names = names_lasso
        
        removed_count = X.shape[1] - current_features.shape[1]
        self.selection_history.append({
            'step': 'ml_feature_selection',
            'features_before': X.shape[1],
            'features_after': current_features.shape[1],
            'features_removed': removed_count
        })
        
        logger.info("ML-based feature selection completed",
                   features_removed=removed_count,
                   features_remaining=current_features.shape[1])
        
        return current_features, current_names
    
    def _apply_dimensionality_reduction(self, X: np.ndarray, 
                                      feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Apply dimensionality reduction techniques."""
        
        logger.info("Applying dimensionality reduction", features=X.shape[1])
        
        # Scale features before dimensionality reduction
        X_scaled = self.scaler.fit_transform(X)
        
        # PCA
        n_pca_components = min(self.config.pca_components, X_scaled.shape[1], X_scaled.shape[0])
        self.pca = PCA(n_components=n_pca_components, random_state=42)
        X_pca = self.pca.fit_transform(X_scaled)
        
        pca_names = [f"pca_{i}" for i in range(X_pca.shape[1])]
        
        # ICA (optional)
        if self.config.ica_components > 0:
            n_ica_components = min(self.config.ica_components, X_scaled.shape[1], X_scaled.shape[0])
            self.ica = FastICA(n_components=n_ica_components, random_state=42, max_iter=1000)
            X_ica = self.ica.fit_transform(X_scaled)
            
            ica_names = [f"ica_{i}" for i in range(X_ica.shape[1])]
            
            # Combine PCA and ICA
            X_combined = np.hstack([X_pca, X_ica])
            combined_names = pca_names + ica_names
        else:
            X_combined = X_pca
            combined_names = pca_names
        
        # Truncate to target dimensions if needed
        if X_combined.shape[1] > self.config.target_dimensions:
            X_final = X_combined[:, :self.config.target_dimensions]
            final_names = combined_names[:self.config.target_dimensions]
        else:
            X_final = X_combined
            final_names = combined_names
        
        self.selection_history.append({
            'step': 'dimensionality_reduction',
            'features_before': X.shape[1],
            'features_after': X_final.shape[1],
            'pca_components': X_pca.shape[1],
            'ica_components': X_ica.shape[1] if hasattr(X_ica, 'shape') else 0,
            'final_dimensions': X_final.shape[1]
        })
        
        # Store PCA explained variance info
        if self.pca:
            self.feature_importance_scores['pca_explained_variance_ratio'] = self.pca.explained_variance_ratio_.tolist()
            self.feature_importance_scores['pca_cumulative_variance'] = np.cumsum(self.pca.explained_variance_ratio_).tolist()
        
        logger.info("Dimensionality reduction completed",
                   final_dimensions=X_final.shape[1],
                   pca_variance_explained=f"{np.sum(self.pca.explained_variance_ratio_):.2%}")
        
        return X_final, final_names
    
    def _remove_correlated_features(self, X: np.ndarray, 
                                  feature_names: List[str]) -> List[int]:
        """Remove highly correlated features."""
        
        if X.shape[1] == 0:
            return []
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(X.T)
        
        # Handle NaN values (replace with 0)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        
        # Find highly correlated pairs
        upper_triangle = np.triu(np.abs(corr_matrix), k=1)
        high_corr_pairs = np.where(upper_triangle > self.config.correlation_threshold)
        
        # Determine which features to remove
        features_to_remove = set()
        
        for i, j in zip(high_corr_pairs[0], high_corr_pairs[1]):
            if i not in features_to_remove and j not in features_to_remove:
                # Remove the feature with lower DJ relevance (if available)
                score_i = self.feature_importance_scores.get(feature_names[i], {}).get('dj_relevance', 0.5)
                score_j = self.feature_importance_scores.get(feature_names[j], {}).get('dj_relevance', 0.5)
                
                if score_i >= score_j:
                    features_to_remove.add(j)
                else:
                    features_to_remove.add(i)
        
        # Return indices of features to keep
        keep_indices = [i for i in range(X.shape[1]) if i not in features_to_remove]
        
        logger.info("Correlation filtering",
                   high_corr_pairs=len(high_corr_pairs[0]),
                   features_removed=len(features_to_remove))
        
        return keep_indices
    
    def _save_feature_importance(self):
        """Save feature importance scores and selection history."""
        
        try:
            import json
            
            # Prepare data for saving
            save_data = {
                'config': {
                    'variance_threshold': self.config.variance_threshold,
                    'correlation_threshold': self.config.correlation_threshold,
                    'target_dimensions': self.config.target_dimensions,
                    'k_best_features': self.config.k_best_features
                },
                'selected_feature_names': self.selected_feature_names,
                'feature_importance_scores': self.feature_importance_scores,
                'selection_history': self.selection_history
            }
            
            # Save to file
            output_path = Path(self.config.feature_names_file)
            with open(output_path, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            logger.info("Feature importance saved", file=str(output_path))
            
        except Exception as e:
            logger.warning("Failed to save feature importance", error=str(e))
    
    def get_feature_importance_summary(self) -> Dict[str, Any]:
        """Get a summary of feature selection results."""
        
        return {
            'total_steps': len(self.selection_history),
            'selection_history': self.selection_history,
            'final_feature_count': len(self.selected_feature_names),
            'selected_features': self.selected_feature_names[:20],  # First 20 for preview
            'pca_variance_explained': self.feature_importance_scores.get('pca_cumulative_variance', [])[-5:] if self.feature_importance_scores.get('pca_cumulative_variance') else []
        }


# Convenience function for quick feature selection
def select_music_features(X: np.ndarray, 
                         y: Optional[np.ndarray] = None,
                         feature_names: Optional[List[str]] = None,
                         target_dimensions: int = 512) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """
    Quick feature selection for music data.
    
    Args:
        X: Feature matrix
        y: Optional target values
        feature_names: Feature names
        target_dimensions: Target number of dimensions
        
    Returns:
        Tuple of (selected_features, selected_names, selection_summary)
    """
    
    config = FeatureSelectionConfig(target_dimensions=target_dimensions)
    selector = ComprehensiveFeatureSelector(config)
    
    X_selected, names_selected = selector.fit_select(X, y, feature_names)
    summary = selector.get_feature_importance_summary()
    
    return X_selected, names_selected, summary