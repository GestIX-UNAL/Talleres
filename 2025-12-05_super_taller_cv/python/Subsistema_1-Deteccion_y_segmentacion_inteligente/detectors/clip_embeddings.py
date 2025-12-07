import torch
import clip
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import json

class CLIPEmbeddingSystem:
    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str = None
    ):
        """
        Initialize CLIP embedding system
        
        Args:
            model_name: CLIP model variant
            device: 'cuda' or 'cpu'
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading CLIP model {model_name} on {self.device}...")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        
        print("CLIP model loaded successfully")
    
    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """
        Encode single image to embedding vector
        
        Args:
            image: BGR image from OpenCV
            
        Returns:
            embedding: Normalized embedding vector
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Preprocess and encode
        image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy().flatten()
    
    def encode_text(self, texts: List[str]) -> np.ndarray:
        """
        Encode text queries to embedding vectors
        
        Args:
            texts: List of text strings
            
        Returns:
            embeddings: Array of normalized text embeddings
        """
        text_tokens = clip.tokenize(texts).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy()
    
    def compute_similarity(
        self,
        image_embedding: np.ndarray,
        text_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between image and text embeddings
        
        Args:
            image_embedding: Single image embedding
            text_embeddings: Array of text embeddings
            
        Returns:
            similarities: Cosine similarity scores
        """
        # Ensure proper shape
        if image_embedding.ndim == 1:
            image_embedding = image_embedding.reshape(1, -1)
        
        # Compute dot product (cosine similarity for normalized vectors)
        similarities = np.dot(text_embeddings, image_embedding.T).flatten()
        
        return similarities
    
    def process_image_directory(
        self,
        image_dir: str,
        output_dir: Optional[str] = None
    ) -> Tuple[List[np.ndarray], List[str]]:
        """
        Process all images in directory and extract embeddings
        
        Args:
            image_dir: Directory containing images
            output_dir: Directory to save results
            
        Returns:
            embeddings: List of embedding vectors
            filenames: List of corresponding filenames
        """
        image_path = Path(image_dir)
        image_files = list(image_path.glob("*.jpg")) + \
                     list(image_path.glob("*.png")) + \
                     list(image_path.glob("*.jpeg"))
        
        if not image_files:
            raise ValueError(f"No images found in {image_dir}")
        
        print(f"Processing {len(image_files)} images...")
        
        embeddings = []
        filenames = []
        
        for img_file in image_files:
            # Read image
            image = cv2.imread(str(img_file))
            if image is None:
                print(f"Warning: Could not read {img_file}")
                continue
            
            # Extract embedding
            embedding = self.encode_image(image)
            embeddings.append(embedding)
            filenames.append(img_file.name)
            
            print(f"Processed: {img_file.name}")
        
        # Save embeddings
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            embeddings_array = np.array(embeddings)
            np.save(output_path / "embeddings.npy", embeddings_array)
            
            with open(output_path / "filenames.json", 'w') as f:
                json.dump(filenames, f, indent=2)
            
            print(f"Saved embeddings to {output_dir}")
        
        return embeddings, filenames
    
    def visualize_embeddings_pca(
        self,
        embeddings: List[np.ndarray],
        labels: List[str],
        output_path: Optional[str] = None
    ):
        """
        Visualize embeddings using PCA
        
        Args:
            embeddings: List of embedding vectors
            labels: List of labels for each embedding
            output_path: Path to save visualization
        """
        embeddings_array = np.array(embeddings)
        
        # Apply PCA
        print("Applying PCA...")
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings_array)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=range(len(embeddings)),
            cmap='viridis',
            s=100,
            alpha=0.6
        )
        
        # Add labels
        for i, label in enumerate(labels):
            plt.annotate(
                label,
                (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                fontsize=8,
                alpha=0.7
            )
        
        plt.colorbar(scatter, label='Image Index')
        plt.title(f'CLIP Embeddings Visualization (PCA)\nVariance Explained: {pca.explained_variance_ratio_.sum():.2%}')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        plt.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved PCA visualization to {output_path}")
        
        plt.show()
    
    def visualize_embeddings_tsne(
        self,
        embeddings: List[np.ndarray],
        labels: List[str],
        output_path: Optional[str] = None,
        perplexity: int = 30
    ):
        """
        Visualize embeddings using t-SNE
        
        Args:
            embeddings: List of embedding vectors
            labels: List of labels for each embedding
            output_path: Path to save visualization
            perplexity: t-SNE perplexity parameter
        """
        embeddings_array = np.array(embeddings)
        
        # Apply t-SNE
        print("Applying t-SNE (this may take a while)...")
        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, len(embeddings) - 1),
            random_state=42
        )
        embeddings_2d = tsne.fit_transform(embeddings_array)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=range(len(embeddings)),
            cmap='plasma',
            s=100,
            alpha=0.6
        )
        
        # Add labels
        for i, label in enumerate(labels):
            plt.annotate(
                label,
                (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                fontsize=8,
                alpha=0.7
            )
        
        plt.colorbar(scatter, label='Image Index')
        plt.title('CLIP Embeddings Visualization (t-SNE)')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved t-SNE visualization to {output_path}")
        
        plt.show()
    
    def image_search(
        self,
        query_texts: List[str],
        image_embeddings: List[np.ndarray],
        image_names: List[str],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search images using text queries
        
        Args:
            query_texts: List of text queries
            image_embeddings: Precomputed image embeddings
            image_names: Corresponding image filenames
            top_k: Number of top results to return
            
        Returns:
            results: List of search results for each query
        """
        # Encode queries
        text_embeddings = self.encode_text(query_texts)
        
        results = []
        
        for query_idx, query in enumerate(query_texts):
            # Compute similarities
            similarities = []
            for img_emb in image_embeddings:
                sim = self.compute_similarity(img_emb, text_embeddings[query_idx:query_idx+1])
                similarities.append(sim[0])
            
            # Get top-k
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            query_results = {
                'query': query,
                'results': [
                    {
                        'filename': image_names[idx],
                        'similarity': float(similarities[idx]),
                        'rank': rank + 1
                    }
                    for rank, idx in enumerate(top_indices)
                ]
            }
            
            results.append(query_results)
        
        return results


def main():
    """Demo usage"""
    # Initialize CLIP system
    clip_system = CLIPEmbeddingSystem(model_name="ViT-B/32")
    
    # Example 1: Process images from directory
    image_dir = "data/input/"
    embeddings, filenames = clip_system.process_image_directory(
        image_dir=image_dir,
        output_dir="results/embeddings/"
    )
    
    # Example 2: Visualize with PCA
    clip_system.visualize_embeddings_pca(
        embeddings=embeddings,
        labels=filenames,
        output_path="results/embeddings/pca_visualization.png"
    )
    
    # Example 3: Visualize with t-SNE
    clip_system.visualize_embeddings_tsne(
        embeddings=embeddings,
        labels=filenames,
        output_path="results/embeddings/tsne_visualization.png"
    )
    
    # Example 4: Image search
    queries = [
        "a cat sitting on a couch",
        "a person wearing sunglasses",
        "a beautiful landscape with mountains"
    ]
    
    search_results = clip_system.image_search(
        query_texts=queries,
        image_embeddings=embeddings,
        image_names=filenames,
        top_k=3
    )
    
    print("\nSearch Results:")
    for result in search_results:
        print(f"\nQuery: '{result['query']}'")
        for match in result['results']:
            print(f"  {match['rank']}. {match['filename']} (similarity: {match['similarity']:.3f})")


if __name__ == "__main__":
    main()