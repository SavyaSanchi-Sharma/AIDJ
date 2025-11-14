"""
Music embedding engine for similarity search and recommendation.
Combines feature extraction, selection, and vector storage with pgvector.
"""

import numpy as np
import asyncio
import structlog
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import pickle
import json
from dataclasses import dataclass
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import faiss

from .feature_extractor import ComprehensiveFeatureExtractor, MetadataConfig
from .feature_selector import ComprehensiveFeatureSelector, FeatureSelectionConfig
from .metadata_extractor import extract_comprehensive_metadata

logger = structlog.get_logger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    
    # Feature extraction
    sample_rate: int = 44100
    extract_metadata: bool = True
    
    # Feature selection
    target_dimensions: int = 512
    use_feature_selection: bool = True
    
    # Neural embedding
    embedding_dim: int = 256
    hidden_dims: List[int] = None
    dropout_rate: float = 0.3
    
    # Training
    batch_size: int = 32
    learning_rate: float = 0.001
    max_epochs: int = 100
    early_stopping_patience: int = 10
    
    # Similarity search
    use_faiss: bool = True
    faiss_index_type: str = "IVF"  # "IVF", "HNSW", "Flat"
    n_clusters: int = 100
    
    # Storage
    save_embeddings: bool = True
    embeddings_cache_dir: str = "embeddings_cache"
    model_save_path: str = "music_embedding_model.pth"

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [1024, 512, 256]


class MusicEmbeddingNet(nn.Module):
    """
    Neural network for learning music embeddings.
    Uses contrastive learning to place similar tracks closer together.
    """
    
    def __init__(self, input_dim: int, embedding_dim: int = 256, 
                 hidden_dims: List[int] = None, dropout_rate: float = 0.3):
        """
        Initialize embedding network.
        
        Args:
            input_dim: Input feature dimension
            embedding_dim: Output embedding dimension
            hidden_dims: Hidden layer dimensions
            dropout_rate: Dropout rate for regularization
        """
        
        super(MusicEmbeddingNet, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [1024, 512, 256]
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # Build encoder layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Final embedding layer
        layers.append(nn.Linear(prev_dim, embedding_dim))
        
        self.encoder = nn.Sequential(*layers)
        
        # L2 normalization for embeddings
        self.normalize = True
        
        logger.info("MusicEmbeddingNet initialized",
                   input_dim=input_dim,
                   embedding_dim=embedding_dim,
                   hidden_dims=hidden_dims,
                   total_params=sum(p.numel() for p in self.parameters()))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        
        embedding = self.encoder(x)
        
        if self.normalize:
            embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for training music embeddings.
    Pulls similar tracks together and pushes dissimilar tracks apart.
    """
    
    def __init__(self, temperature: float = 0.1, margin: float = 1.0):
        """
        Initialize contrastive loss.
        
        Args:
            temperature: Temperature parameter for softmax
            margin: Margin for negative samples
        """
        
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(self, embeddings: torch.Tensor, similarity_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            embeddings: Batch of embeddings [batch_size, embedding_dim]
            similarity_labels: Similarity matrix [batch_size, batch_size]
                1 for similar pairs, 0 for dissimilar pairs
        
        Returns:
            Contrastive loss value
        """
        
        batch_size = embeddings.size(0)
        
        # Compute pairwise cosine similarities
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())
        
        # Apply temperature
        similarity_matrix = similarity_matrix / self.temperature
        
        # Create masks for positive and negative pairs
        positive_mask = similarity_labels.bool()
        negative_mask = ~positive_mask
        
        # Remove diagonal (self-similarity)
        identity_mask = torch.eye(batch_size, device=embeddings.device).bool()
        positive_mask = positive_mask & ~identity_mask
        negative_mask = negative_mask & ~identity_mask
        
        # InfoNCE loss
        if positive_mask.sum() > 0:
            # For each positive pair, compute loss against all negatives
            pos_similarities = similarity_matrix[positive_mask]
            
            # Compute denominator (sum over all negatives for each positive)
            neg_exp_sims = torch.exp(similarity_matrix[negative_mask])
            
            # InfoNCE loss
            loss = -torch.mean(pos_similarities) + torch.log(torch.mean(neg_exp_sims))
        else:
            # Fallback to simple margin loss if no positive pairs
            pos_pairs = similarity_matrix[positive_mask]
            neg_pairs = similarity_matrix[negative_mask]
            
            if len(pos_pairs) > 0 and len(neg_pairs) > 0:
                loss = torch.clamp(self.margin - pos_pairs.mean() + neg_pairs.mean(), min=0)
            else:
                loss = torch.tensor(0.0, device=embeddings.device)
        
        return loss


class MusicEmbeddingEngine:
    """
    Complete music embedding engine.
    
    Handles feature extraction, selection, embedding generation,
    and similarity search with vector storage.
    """
    
    def __init__(self, config: EmbeddingConfig, metadata_config: Optional[MetadataConfig] = None):
        """Initialize embedding engine with configurations."""
        
        self.config = config
        self.metadata_config = metadata_config or MetadataConfig()
        
        # Core components
        self.feature_extractor = ComprehensiveFeatureExtractor(
            sample_rate=config.sample_rate,
            metadata_config=self.metadata_config
        )
        
        if config.use_feature_selection:
            selection_config = FeatureSelectionConfig(
                target_dimensions=config.target_dimensions
            )
            self.feature_selector = ComprehensiveFeatureSelector(selection_config)
        else:
            self.feature_selector = None
        
        # Neural network components
        self.embedding_net = None
        self.scaler = StandardScaler()
        self.loss_function = ContrastiveLoss()
        
        # Vector search components
        self.faiss_index = None
        self.embeddings_cache = {}
        self.track_metadata = {}
        
        # Training state
        self.is_trained = False
        self.training_history = []
        
        # Create cache directory
        Path(config.embeddings_cache_dir).mkdir(exist_ok=True)
        
        logger.info("MusicEmbeddingEngine initialized",
                   embedding_dim=config.embedding_dim,
                   use_feature_selection=config.use_feature_selection,
                   use_faiss=config.use_faiss)
    
    async def extract_and_embed_track(self, audio_file_path: str, 
                                    track_id: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extract features and generate embedding for a single track.
        
        Args:
            audio_file_path: Path to audio file
            track_id: Optional track identifier
            
        Returns:
            Tuple of (embedding_vector, metadata)
        """
        
        if track_id is None:
            track_id = Path(audio_file_path).stem
        
        logger.info("Processing track", track_id=track_id, file_path=audio_file_path)
        
        try:
            # Check cache first
            if track_id in self.embeddings_cache:
                logger.info("Using cached embedding", track_id=track_id)
                return self.embeddings_cache[track_id], self.track_metadata.get(track_id, {})
            
            # Extract comprehensive features
            features_dict = await self.feature_extractor.extract_maximum_features(audio_file_path)
            
            # Get feature vector
            if "comprehensive_feature_vector" in features_dict:
                feature_vector = np.array(features_dict["comprehensive_feature_vector"])
            else:
                # Fallback to flattening all features
                feature_vector = self._flatten_all_features(features_dict)
            
            # Apply feature selection if available
            if self.feature_selector and self.feature_selector.is_fitted:
                feature_vector = self.feature_selector.transform(feature_vector.reshape(1, -1))[0]
            
            # Generate embedding
            embedding = self._generate_embedding(feature_vector)
            
            # Store in cache
            if self.config.save_embeddings:
                self.embeddings_cache[track_id] = embedding
                self.track_metadata[track_id] = {
                    'file_path': audio_file_path,
                    'features': features_dict,
                    'extraction_time': datetime.now().isoformat()
                }
            
            logger.info("Track processing completed", 
                       track_id=track_id,
                       embedding_dim=len(embedding),
                       feature_dim=len(feature_vector))
            
            return embedding, features_dict
            
        except Exception as e:
            logger.error("Track processing failed", 
                        track_id=track_id,
                        file_path=audio_file_path,
                        error=str(e))
            raise
    
    def train_embedding_network(self, 
                               training_data: List[Dict[str, Any]], 
                               validation_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Train the embedding network using contrastive learning.
        
        Args:
            training_data: List of training examples with features and similarity labels
            validation_data: Optional validation data
            
        Returns:
            Training history and metrics
        """
        
        logger.info("Starting embedding network training", 
                   training_samples=len(training_data))
        
        # Prepare training data
        X_train, y_train = self._prepare_training_data(training_data)
        
        if validation_data:
            X_val, y_val = self._prepare_training_data(validation_data)
        else:
            X_val, y_val = None, None
        
        # Apply feature selection
        if self.config.use_feature_selection:
            X_train, selected_names = self.feature_selector.fit_select(X_train, y_train)
            
            if X_val is not None:
                X_val = self.feature_selector.transform(X_val)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
        else:
            X_val_scaled = None
        
        # Initialize network
        input_dim = X_train_scaled.shape[1]
        self.embedding_net = MusicEmbeddingNet(
            input_dim=input_dim,
            embedding_dim=self.config.embedding_dim,
            hidden_dims=self.config.hidden_dims,
            dropout_rate=self.config.dropout_rate
        )
        
        # Setup training
        optimizer = torch.optim.Adam(
            self.embedding_net.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-5
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.max_epochs):
            # Training phase
            train_loss = self._train_epoch(X_train_scaled, y_train, optimizer)
            
            # Validation phase
            if X_val_scaled is not None:
                val_loss = self._validate_epoch(X_val_scaled, y_val)
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    if self.config.model_save_path:
                        self._save_model()
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info("Early stopping triggered", epoch=epoch)
                    break
                    
                logger.info("Training epoch completed", 
                           epoch=epoch,
                           train_loss=train_loss,
                           val_loss=val_loss)
            else:
                logger.info("Training epoch completed", 
                           epoch=epoch,
                           train_loss=train_loss)
            
            # Store history
            self.training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss if X_val_scaled is not None else None
            })
        
        self.is_trained = True
        
        # Build search index
        if self.config.use_faiss:
            self._build_faiss_index()
        
        logger.info("Training completed", 
                   total_epochs=len(self.training_history),
                   best_val_loss=best_val_loss)
        
        return {
            'training_history': self.training_history,
            'best_val_loss': best_val_loss,
            'final_train_loss': self.training_history[-1]['train_loss'] if self.training_history else None
        }
    
    def find_similar_tracks(self, 
                          query_embedding: np.ndarray, 
                          k: int = 10, 
                          threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Find similar tracks using vector similarity search.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of similar tracks to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of similar tracks with similarity scores
        """
        
        if not self.embeddings_cache:
            logger.warning("No embeddings in cache")
            return []
        
        if self.config.use_faiss and self.faiss_index is not None:
            return self._faiss_search(query_embedding, k, threshold)
        else:
            return self._cosine_similarity_search(query_embedding, k, threshold)
    
    def find_similar_tracks_by_id(self, track_id: str, k: int = 10) -> List[Dict[str, Any]]:
        """Find tracks similar to a specific track by ID."""
        
        if track_id not in self.embeddings_cache:
            logger.warning("Track not found in cache", track_id=track_id)
            return []
        
        query_embedding = self.embeddings_cache[track_id]
        similar_tracks = self.find_similar_tracks(query_embedding, k + 1)  # +1 to exclude self
        
        # Remove the query track itself
        similar_tracks = [track for track in similar_tracks if track['track_id'] != track_id]
        
        return similar_tracks[:k]
    
    def _generate_embedding(self, feature_vector: np.ndarray) -> np.ndarray:
        """Generate embedding from feature vector."""
        
        if self.embedding_net is None or not self.is_trained:
            # Return normalized feature vector if no trained network
            normalized_features = feature_vector / (np.linalg.norm(feature_vector) + 1e-8)
            
            # Pad or truncate to embedding dimension
            if len(normalized_features) > self.config.embedding_dim:
                return normalized_features[:self.config.embedding_dim]
            else:
                padded = np.zeros(self.config.embedding_dim)
                padded[:len(normalized_features)] = normalized_features
                return padded
        
        # Use trained network
        self.embedding_net.eval()
        
        with torch.no_grad():
            # Scale and convert to tensor
            scaled_features = self.scaler.transform(feature_vector.reshape(1, -1))
            features_tensor = torch.tensor(scaled_features, dtype=torch.float32)
            
            # Generate embedding
            embedding = self.embedding_net(features_tensor)
            
            return embedding.cpu().numpy().flatten()
    
    def _flatten_all_features(self, features_dict: Dict[str, Any]) -> np.ndarray:
        """Flatten nested feature dictionary into vector."""
        
        def flatten_recursive(obj, prefix=""):
            flattened = []
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_prefix = f"{prefix}_{key}" if prefix else key
                    flattened.extend(flatten_recursive(value, new_prefix))
            elif isinstance(obj, (list, np.ndarray)):
                for i, item in enumerate(obj):
                    if isinstance(item, (int, float, np.number)):
                        flattened.append(float(item))
                    else:
                        new_prefix = f"{prefix}_{i}" if prefix else str(i)
                        flattened.extend(flatten_recursive(item, new_prefix))
            elif isinstance(obj, (int, float, np.number)):
                flattened.append(float(obj))
            elif isinstance(obj, str):
                # Convert string to hash-based feature
                flattened.append(float(hash(obj) % 1000) / 1000.0)
            
            return flattened
        
        flattened = flatten_recursive(features_dict)
        
        # Pad or truncate to reasonable size
        target_size = 1500
        if len(flattened) > target_size:
            flattened = flattened[:target_size]
        else:
            flattened.extend([0.0] * (target_size - len(flattened)))
        
        return np.array(flattened, dtype=np.float32)
    
    def _prepare_training_data(self, data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from list of examples."""
        
        features = []
        labels = []
        
        for example in data:
            if 'features' in example:
                feature_vec = self._flatten_all_features(example['features'])
                features.append(feature_vec)
            
            if 'similarity_label' in example:
                labels.append(example['similarity_label'])
            else:
                labels.append(0.0)  # Default to no similarity
        
        return np.array(features, dtype=np.float32), np.array(labels, dtype=np.float32)
    
    def _train_epoch(self, X: np.ndarray, y: np.ndarray, optimizer) -> float:
        """Train for one epoch."""
        
        self.embedding_net.train()
        
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )
        
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True
        )
        
        total_loss = 0.0
        batch_count = 0
        
        for batch_features, batch_labels in dataloader:
            optimizer.zero_grad()
            
            # Generate embeddings
            embeddings = self.embedding_net(batch_features)
            
            # Create similarity matrix for batch
            batch_size = embeddings.size(0)
            similarity_matrix = torch.zeros(batch_size, batch_size)
            
            # Simple similarity: same label = similar
            for i in range(batch_size):
                for j in range(batch_size):
                    if i != j and batch_labels[i] == batch_labels[j]:
                        similarity_matrix[i, j] = 1.0
            
            # Compute loss
            loss = self.loss_function(embeddings, similarity_matrix)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        return total_loss / max(batch_count, 1)
    
    def _validate_epoch(self, X: np.ndarray, y: np.ndarray) -> float:
        """Validate for one epoch."""
        
        self.embedding_net.eval()
        
        with torch.no_grad():
            features_tensor = torch.tensor(X, dtype=torch.float32)
            labels_tensor = torch.tensor(y, dtype=torch.float32)
            
            embeddings = self.embedding_net(features_tensor)
            
            # Create similarity matrix
            batch_size = embeddings.size(0)
            similarity_matrix = torch.zeros(batch_size, batch_size)
            
            for i in range(batch_size):
                for j in range(batch_size):
                    if i != j and labels_tensor[i] == labels_tensor[j]:
                        similarity_matrix[i, j] = 1.0
            
            loss = self.loss_function(embeddings, similarity_matrix)
            
            return loss.item()
    
    def _build_faiss_index(self):
        """Build FAISS index for fast similarity search."""
        
        if not self.embeddings_cache:
            logger.warning("No embeddings to index")
            return
        
        # Convert embeddings to matrix
        embeddings_matrix = np.array(list(self.embeddings_cache.values()))
        embedding_dim = embeddings_matrix.shape[1]
        
        # Create FAISS index
        if self.config.faiss_index_type == "IVF":
            quantizer = faiss.IndexFlatL2(embedding_dim)
            self.faiss_index = faiss.IndexIVFFlat(
                quantizer, 
                embedding_dim, 
                min(self.config.n_clusters, len(embeddings_matrix))
            )
            
            # Train index
            self.faiss_index.train(embeddings_matrix)
            
        elif self.config.faiss_index_type == "HNSW":
            self.faiss_index = faiss.IndexHNSWFlat(embedding_dim, 32)
            
        else:  # Flat
            self.faiss_index = faiss.IndexFlatL2(embedding_dim)
        
        # Add embeddings
        self.faiss_index.add(embeddings_matrix)
        
        logger.info("FAISS index built", 
                   index_type=self.config.faiss_index_type,
                   total_embeddings=len(embeddings_matrix))
    
    def _faiss_search(self, query_embedding: np.ndarray, 
                     k: int, threshold: float) -> List[Dict[str, Any]]:
        """Search using FAISS index."""
        
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search
        distances, indices = self.faiss_index.search(query_embedding, k)
        
        # Convert to results
        results = []
        track_ids = list(self.embeddings_cache.keys())
        
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0 and idx < len(track_ids):
                similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity
                
                if similarity >= threshold:
                    results.append({
                        'track_id': track_ids[idx],
                        'similarity': similarity,
                        'distance': distance,
                        'rank': i + 1
                    })
        
        return results
    
    def _cosine_similarity_search(self, query_embedding: np.ndarray, 
                                k: int, threshold: float) -> List[Dict[str, Any]]:
        """Search using cosine similarity."""
        
        if not self.embeddings_cache:
            return []
        
        # Get all embeddings
        track_ids = list(self.embeddings_cache.keys())
        embeddings_matrix = np.array([self.embeddings_cache[tid] for tid in track_ids])
        
        # Compute similarities
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1), 
            embeddings_matrix
        )[0]
        
        # Get top-k results above threshold
        results = []
        
        for i, similarity in enumerate(similarities):
            if similarity >= threshold:
                results.append({
                    'track_id': track_ids[i],
                    'similarity': float(similarity),
                    'rank': len(results) + 1
                })
        
        # Sort by similarity and return top-k
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return results[:k]
    
    def _save_model(self):
        """Save the trained model and supporting data."""
        
        if self.embedding_net is None:
            return
        
        save_data = {
            'model_state_dict': self.embedding_net.state_dict(),
            'scaler': pickle.dumps(self.scaler),
            'config': {
                'embedding_dim': self.config.embedding_dim,
                'hidden_dims': self.config.hidden_dims,
                'input_dim': self.embedding_net.input_dim
            },
            'training_history': self.training_history
        }
        
        torch.save(save_data, self.config.model_save_path)
        
        logger.info("Model saved", path=self.config.model_save_path)
    
    def load_model(self, model_path: str):
        """Load a trained model."""
        
        save_data = torch.load(model_path, map_location='cpu')
        
        # Recreate network
        config = save_data['config']
        self.embedding_net = MusicEmbeddingNet(
            input_dim=config['input_dim'],
            embedding_dim=config['embedding_dim'],
            hidden_dims=config['hidden_dims']
        )
        
        # Load state
        self.embedding_net.load_state_dict(save_data['model_state_dict'])
        self.scaler = pickle.loads(save_data['scaler'])
        self.training_history = save_data.get('training_history', [])
        
        self.is_trained = True
        
        logger.info("Model loaded", path=model_path)