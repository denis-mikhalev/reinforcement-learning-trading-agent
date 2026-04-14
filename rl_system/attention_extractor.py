"""
Attention-based Feature Extractor for RL Trading Agent

Использует Multi-Head Attention чтобы агент мог "фокусироваться" на важных 
моментах в истории цены, а не просто читать все подряд.

Архитектура:
1. Input: (batch, lookback * features) - плоский вектор
2. Reshape: (batch, lookback, features) - последовательность
3. Multi-Head Attention (4 heads, 2 layers) - "что важно?"
4. Residual connections + LayerNorm
5. Flatten + FC layers
6. Output: latent vector для policy/value heads
"""

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym


class AttentionFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor с Multi-Head Attention механизмом.
    
    Дает агенту способность "обращать внимание" на важные паттерны в истории,
    вместо равномерной обработки всех timesteps.
    """
    
    def __init__(
        self, 
        observation_space: gym.spaces.Box,
        features_dim: int = 256,
        n_attention_heads: int = 4,
        n_attention_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            observation_space: Gym observation space
            features_dim: Размер выходного latent vector
            n_attention_heads: Количество attention heads (должно делить n_features без остатка)
            n_attention_layers: Количество attention layers
            dropout: Dropout rate для регуляризации
        """
        super().__init__(observation_space, features_dim)
        
        # Position features всегда 5 (для обоих форматов)
        self.position_features = 5
        
        # Observation space может быть 1D или 2D
        if len(observation_space.shape) == 2:
            # 2D: (lookback, n_features)
            self.lookback, self.n_features = observation_space.shape
            print(f"✅ Detected 2D observation: shape={observation_space.shape}")
        else:
            # 1D: (lookback * n_features,) - плоский вектор
            # Предполагаем что последние 5 features - это position info
            input_size = observation_space.shape[0]
            remaining = input_size - self.position_features
            
            # Находим делители для определения lookback
            possible_lookbacks = [30, 60, 90, 120, 150, 180, 250, 500]
            self.lookback = None
            self.n_features = None
            
            for lb in possible_lookbacks:
                if remaining % lb == 0:
                    self.lookback = lb
                    self.n_features = remaining // lb
                    break
            
            if self.lookback is None:
                # Fallback: используем простую эвристику
                self.lookback = 120
                self.n_features = remaining // self.lookback
                print(f"⚠️  Warning: Could not determine exact lookback, using {self.lookback}")
        
        print(f"📊 Attention Extractor initialized:")
        print(f"   Lookback: {self.lookback}")
        print(f"   Features per timestep: {self.n_features}")
        print(f"   Attention heads: {n_attention_heads}")
        print(f"   Attention layers: {n_attention_layers}")
        
        # Проверяем что n_features делится на n_attention_heads
        if self.n_features % n_attention_heads != 0:
            # Добавим projection layer чтобы получить размерность кратную числу heads
            projected_dim = ((self.n_features // n_attention_heads) + 1) * n_attention_heads
            self.input_projection = nn.Linear(self.n_features, projected_dim)
            embed_dim = projected_dim
            print(f"   ⚙️  Added projection: {self.n_features} -> {projected_dim}")
        else:
            self.input_projection = None
            embed_dim = self.n_features
        
        # Multi-Head Attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=n_attention_heads,
                dropout=dropout,
                batch_first=True  # (batch, seq, feature) format
            )
            for _ in range(n_attention_layers)
        ])
        
        # Layer normalization после каждого attention layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim)
            for _ in range(n_attention_layers)
        ])
        
        # Dropout для регуляризации
        self.dropout = nn.Dropout(dropout)
        
        # Final FC layers для создания latent representation
        # Объединяем attention output + position features
        combined_size = embed_dim + self.position_features
        
        self.fc_layers = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, features_dim),
            nn.ReLU()
        )
        
        print(f"   Output features_dim: {features_dim}")
        print(f"✅ Attention extractor ready\n")
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass через attention mechanism.
        
        Args:
            observations: (batch, lookback, n_features) или (batch, lookback*n_features + position_features)
            
        Returns:
            latent: (batch, features_dim) - latent representation для policy/value
        """
        batch_size = observations.shape[0]
        
        # Определяем формат входа
        if len(observations.shape) == 3:
            # 2D input: (batch, lookback, n_features)
            sequence = observations
            # Position features будут нулями (или возьмем последние 5 features из последнего timestep)
            position_features = sequence[:, -1, -5:]  # Последние 5 features последнего timestep
        else:
            # 1D input: (batch, lookback*n_features + position_features)
            sequence_flat = observations[:, :-self.position_features]  # (batch, lookback*n_features)
            position_features = observations[:, -self.position_features:]  # (batch, 5)
            
            # Reshape в последовательность: (batch, lookback, n_features)
            sequence = sequence_flat.view(batch_size, self.lookback, self.n_features)
        
        # Если нужна проекция для совместимости с attention heads
        if self.input_projection is not None:
            sequence = self.input_projection(sequence)  # (batch, lookback, embed_dim)
        
        # Пропускаем через attention layers с residual connections
        x = sequence
        for attention, layer_norm in zip(self.attention_layers, self.layer_norms):
            # Self-attention
            attn_output, _ = attention(x, x, x)
            
            # Residual connection + LayerNorm (как в Transformer)
            x = layer_norm(x + self.dropout(attn_output))
        
        # Global average pooling по temporal dimension
        # Берем среднее по lookback чтобы получить (batch, embed_dim)
        pooled = x.mean(dim=1)  # (batch, embed_dim)
        
        # Объединяем с position features
        combined = torch.cat([pooled, position_features], dim=1)  # (batch, embed_dim + 5)
        
        # Final FC layers
        latent = self.fc_layers(combined)  # (batch, features_dim)
        
        return latent


class CNNAttentionFeatureExtractor(BaseFeaturesExtractor):
    """
    Гибридная архитектура: CNN для локальных паттернов + Attention для долгосрочных зависимостей.
    
    Использует CNN чтобы выделить краткосрочные паттерны (2-5 дней),
    затем Attention чтобы понять их важность в долгосрочной перспективе.
    """
    
    def __init__(
        self, 
        observation_space: gym.spaces.Box,
        features_dim: int = 256,
        n_attention_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__(observation_space, features_dim)
        
        # Position features всегда 5
        self.position_features = 5
        
        # Observation space может быть 1D или 2D
        if len(observation_space.shape) == 2:
            # 2D: (lookback, n_features)
            self.lookback, self.n_features = observation_space.shape
            print(f"✅ Detected 2D observation: shape={observation_space.shape}")
        else:
            # 1D: плоский вектор
            input_size = observation_space.shape[0]
            remaining = input_size - self.position_features
            
            possible_lookbacks = [30, 60, 90, 120, 150, 180, 250, 500]
            self.lookback = None
            self.n_features = None
            
            for lb in possible_lookbacks:
                if remaining % lb == 0:
                    self.lookback = lb
                    self.n_features = remaining // lb
                    break
            
            if self.lookback is None:
                self.lookback = 120
                self.n_features = remaining // self.lookback
        
        print(f"📊 CNN+Attention Extractor initialized:")
        print(f"   Lookback: {self.lookback}, Features: {self.n_features}")
        
        # CNN для локальных паттернов
        # Input: (batch, 1, lookback, n_features) - как изображение
        self.conv_layers = nn.Sequential(
            # Conv1: выделяем 2-3 дневные паттерны
            nn.Conv2d(1, 32, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            
            # Conv2: 5-7 дневные паттерны
            nn.Conv2d(32, 64, kernel_size=(5, 1), padding=(2, 0)),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )
        
        # После CNN размерность: (batch, 64, lookback, n_features)
        # Flatten по features dimension
        cnn_output_dim = 64 * self.n_features
        
        # Проецируем в размерность для attention
        embed_dim = 128
        self.cnn_projection = nn.Linear(cnn_output_dim, embed_dim)
        
        # Self-attention для понимания важности временных паттернов
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Final layers
        combined_size = embed_dim + self.position_features
        self.fc_layers = nn.Sequential(
            nn.Linear(combined_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )
        
        print(f"✅ CNN+Attention extractor ready\n")
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        
        # Определяем формат входа
        if len(observations.shape) == 3:
            # 2D input: (batch, lookback, n_features)
            sequence_2d = observations
            position_features = sequence_2d[:, -1, -5:]  # Последние 5 features последнего timestep
            # Reshape для CNN: (batch, 1, lookback, n_features)
            sequence = sequence_2d.unsqueeze(1)  # Add channel dimension
        else:
            # 1D input: (batch, lookback*n_features + position_features)
            sequence_flat = observations[:, :-self.position_features]
            position_features = observations[:, -self.position_features:]
            
            # Reshape для CNN: (batch, 1, lookback, n_features)
            sequence = sequence_flat.view(batch_size, 1, self.lookback, self.n_features)
        
        # CNN для локальных паттернов
        conv_out = self.conv_layers(sequence)  # (batch, 64, lookback, n_features)
        
        # Flatten по features dimension, оставляем temporal
        # (batch, 64, lookback, n_features) -> (batch, lookback, 64*n_features)
        batch, channels, lookback, n_feat = conv_out.shape
        conv_flat = conv_out.permute(0, 2, 1, 3).contiguous()  # (batch, lookback, 64, n_features)
        conv_flat = conv_flat.view(batch, lookback, channels * n_feat)  # (batch, lookback, 64*n_features)
        
        # Проецируем в embed_dim
        projected = self.cnn_projection(conv_flat)  # (batch, lookback, embed_dim)
        
        # Self-attention
        attn_output, _ = self.attention(projected, projected, projected)
        
        # Residual connection + LayerNorm
        attended = self.layer_norm(projected + self.dropout(attn_output))
        
        # Global average pooling
        pooled = attended.mean(dim=1)  # (batch, embed_dim)
        
        # Объединяем с position features
        combined = torch.cat([pooled, position_features], dim=1)
        
        # Final FC
        latent = self.fc_layers(combined)
        
        return latent


if __name__ == "__main__":
    # Тест архитектуры
    print("🧪 Testing Attention Extractors\n")
    
    # Симулируем observation space: lookback=120, n_features=99, position=5
    lookback = 120
    n_features = 99
    position_features = 5
    input_dim = lookback * n_features + position_features
    
    observation_space = gym.spaces.Box(
        low=-float('inf'), 
        high=float('inf'), 
        shape=(input_dim,),
        dtype=float
    )
    
    print(f"Test observation space: {observation_space.shape}")
    print(f"Expected: lookback={lookback}, n_features={n_features}, position={position_features}\n")
    
    # Test 1: Pure Attention
    print("=" * 60)
    print("TEST 1: AttentionFeatureExtractor")
    print("=" * 60)
    extractor1 = AttentionFeatureExtractor(
        observation_space=observation_space,
        features_dim=256,
        n_attention_heads=4,
        n_attention_layers=2
    )
    
    # Dummy input
    batch_size = 32
    dummy_obs = torch.randn(batch_size, input_dim)
    
    output1 = extractor1(dummy_obs)
    print(f"Output shape: {output1.shape}")
    print(f"Expected: ({batch_size}, 256)")
    assert output1.shape == (batch_size, 256), "Output shape mismatch!"
    print("✅ Test 1 passed!\n")
    
    # Test 2: CNN + Attention
    print("=" * 60)
    print("TEST 2: CNNAttentionFeatureExtractor")
    print("=" * 60)
    extractor2 = CNNAttentionFeatureExtractor(
        observation_space=observation_space,
        features_dim=256,
        n_attention_heads=4
    )
    
    output2 = extractor2(dummy_obs)
    print(f"Output shape: {output2.shape}")
    print(f"Expected: ({batch_size}, 256)")
    assert output2.shape == (batch_size, 256), "Output shape mismatch!"
    print("✅ Test 2 passed!\n")
    
    print("=" * 60)
    print("🎉 All tests passed! Attention extractors ready to use.")
    print("=" * 60)
