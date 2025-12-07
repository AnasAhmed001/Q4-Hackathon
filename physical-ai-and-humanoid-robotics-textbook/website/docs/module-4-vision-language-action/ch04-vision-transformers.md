---
title: Chapter 4 - Vision Transformers for Robotics
description: Explore vision transformers and their application in robotic vision systems, including object detection, scene understanding, and real-time inference optimization.
sidebar_position: 38
---

# Chapter 4 - Vision Transformers for Robotics

Vision Transformers (ViTs) have revolutionized computer vision by applying the transformer architecture, originally developed for natural language processing, to image understanding tasks. For humanoid robots, vision transformers offer powerful capabilities for object detection, scene understanding, and real-time perception that are crucial for navigation, manipulation, and interaction tasks. This chapter explores the application of vision transformers in robotics, focusing on practical implementation, optimization for real-time performance, and integration with robotic systems.

## 4.1 Introduction to Vision Transformers

Vision Transformers represent a paradigm shift from convolutional neural networks (CNNs) to attention-based models for visual recognition. Unlike CNNs that process images through local receptive fields, ViTs divide images into patches and process them using self-attention mechanisms, enabling global context understanding.

### 4.1.1 Key Advantages for Robotics

- **Global Context**: Better understanding of relationships between distant objects in a scene
- **Scalability**: Improved performance with larger models and datasets
- **Flexibility**: Adaptable to various vision tasks (classification, detection, segmentation)
- **Transfer Learning**: Strong performance with pre-trained models on robotic tasks

### 4.1.2 Vision Transformer Architecture

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # Convert images to patches
        x = self.proj(x)  # [B, embed_dim, n_patches ** 0.5, n_patches ** 0.5]
        x = x.flatten(2)  # [B, embed_dim, n_patches]
        x = x.transpose(1, 2)  # [B, n_patches, embed_dim]
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=768, n_heads=12, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        assert self.head_dim * n_heads == embed_dim, "embed_dim must be divisible by n_heads"

        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, n_heads=12, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000,
                 embed_dim=768, depth=12, n_heads=12, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + (img_size // patch_size) ** 2, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=B)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return self.head(x[:, 0])  # Return class token
```

## 4.2 Vision Transformers for Robotic Perception

### 4.2.1 Object Detection with Vision Transformers

For robotic applications, object detection is crucial. DETR (DEtection TRansformer) and its variants have shown excellent performance:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import detr
from torchvision.transforms import functional as TF
import numpy as np

class DETRForObjectDetection(nn.Module):
    def __init__(self, num_classes=91, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        # Backbone using a pre-trained CNN
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove classification head

        # Transformer
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers
        )

        # Prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for no-object class
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)  # 4 for bbox coordinates

        # Query embeddings
        self.query_embed = nn.Embedding(100, hidden_dim)  # 100 object queries

        # Projection from backbone features to hidden dimension
        self.input_proj = nn.Conv2d(2048, hidden_dim, kernel_size=1)

    def forward(self, images):
        # Extract features from backbone
        features = self.backbone(images)

        # Reshape features for transformer
        features = self.input_proj(features)
        bs, c, h, w = features.shape
        features = features.flatten(2).permute(2, 0, 1)  # [H*W, batch, channels]

        # Positional encoding
        pos_embed = self.build_2d_sincos_position_embedding(h, w, c)

        # Transformer
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        tgt = torch.zeros_like(query_embed)

        hs = self.transformer(
            features, tgt,
            query_pos=query_embed,
            pos=pos_embed
        )

        # Prediction heads
        outputs_class = self.class_embed(hs.transpose(0, 1))
        outputs_coord = self.bbox_embed(hs.transpose(0, 1)).sigmoid()

        return {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

    def build_2d_sincos_position_embedding(self, h, w, embed_dim):
        """Build 2D sine-cosine positional embedding"""
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)

        emb_w = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_w)
        emb_h = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_h)

        emb = torch.cat([emb_w, emb_h], dim=1)  # [H*W, C]
        return emb

    def get_1d_sincos_pos_embed_from_grid(self, embed_dim, pos):
        """Sine-cosine positional embedding from 1D grid"""
        assert embed_dim % 2 == 0
        omega = torch.arange(embed_dim // 2, dtype=torch.float32)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega

        pos = pos.reshape(-1)  # [H*W]
        out = torch.einsum('m,d->md', pos, omega)  # [H*W, embed_dim//2]

        emb_sin = torch.sin(out)  # [H*W, embed_dim//2]
        emb_cos = torch.cos(out)  # [H*W, embed_dim//2]

        emb = torch.cat([emb_sin, emb_cos], dim=1)  # [H*W, embed_dim]
        return emb

class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
```

### 4.2.2 Segmentation with Vision Transformers

Segmentation is crucial for humanoid robots to understand object boundaries and spatial relationships:

```python
class SegmentationViT(nn.Module):
    def __init__(self, vit_model, num_classes=21, patch_size=16):
        super().__init__()
        self.vit = vit_model
        self.num_classes = num_classes
        self.patch_size = patch_size

        # Segmentation head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(vit_model.embed_dim, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

        # Upsampling to original image size
        self.upsample = nn.Upsample(scale_factor=patch_size, mode='bilinear', align_corners=False)

    def forward(self, x):
        # Pass through ViT encoder
        B, C, H, W = x.shape
        x = self.vit.patch_embed(x)

        # Add class token and positional embedding
        cls_tokens = repeat(self.vit.cls_token, '() n d -> b n d', b=B)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.vit.pos_embed
        x = self.vit.pos_drop(x)

        # Transformer blocks
        for blk in self.vit.blocks:
            x = blk(x)

        x = self.vit.norm(x)

        # Remove class token and reshape for segmentation
        x = x[:, 1:, :]  # Remove class token
        x = x.permute(0, 2, 1).contiguous()  # [B, embed_dim, n_patches]

        # Reshape to 2D feature map
        patch_h = patch_w = int((x.shape[-1]) ** 0.5)
        x = x.view(B, -1, patch_h, patch_w)

        # Apply segmentation head
        x = self.segmentation_head(x)

        # Upsample to original size
        x = self.upsample(x)

        return x
```

## 4.3 Optimized Vision Transformers for Robotics

### 4.3.1 Mobile-Optimized Models

For humanoid robots with limited computational resources, optimized models are essential:

```python
class MobileVisionTransformer(nn.Module):
    """Lightweight Vision Transformer optimized for mobile/robotic platforms"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000,
                 embed_dim=192, depth=12, n_heads=3, mlp_ratio=4., dropout=0.1,
                 drop_path_rate=0.1, distilled=False):
        super().__init__()
        self.num_classes = num_classes
        self.distilled = distilled
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2 + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Transformer blocks with optimized attention
        self.blocks = nn.ModuleList([
            self._make_block(embed_dim, n_heads, mlp_ratio, dropout, dpr[i])
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Classifier head(s)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if distilled:
            self.head_dist = nn.Linear(embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _make_block(self, embed_dim, n_heads, mlp_ratio, dropout, drop_path):
        """Create an optimized transformer block"""
        return nn.Sequential(
            nn.LayerNorm(embed_dim),
            MultiHeadAttentionOptimized(embed_dim, n_heads, dropout),
            DropPath(drop_path) if drop_path > 0. else nn.Identity(),
            nn.LayerNorm(embed_dim),
            MlpOptimized(embed_dim, int(embed_dim * mlp_ratio), dropout),
            DropPath(drop_path) if drop_path > 0. else nn.Identity()
        )

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        if self.distilled:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if not self.training:
                # During inference, return the average of both classifier predictions
                x = (x + x_dist) / 2
        else:
            x = self.head(x)
        return x

class MultiHeadAttentionOptimized(nn.Module):
    """Optimized multi-head attention for mobile deployment"""
    def __init__(self, embed_dim, n_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Use grouped convolutions for efficiency
        self.qkv_conv = nn.Conv1d(embed_dim, embed_dim * 3, 1, groups=n_heads)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, C // self.n_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x

class MlpOptimized(nn.Module):
    """Optimized MLP for mobile deployment"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def trunc_normal_(tensor, mean=0., std=1.):
    """Truncated normal initialization"""
    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        lower = torch.erfinv(torch.as_tensor(2 * 0.025 - 1))
        upper = torch.erfinv(torch.as_tensor(2 * 0.975 - 1))

        # Fill tensor with uniform values in [0, 1]
        tensor.uniform_(2 * lower, 2 * upper)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=lower, max=upper)
        return tensor
```

### 4.3.2 Quantized Vision Transformers

For deployment on edge devices like the NVIDIA Jetson, quantization is crucial:

```python
import torch.quantization as tq

class QuantizedVisionTransformer(nn.Module):
    def __init__(self, vit_model):
        super().__init__()
        self.vit = vit_model

        # Prepare for quantization
        self.vit.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

    def prepare_quantization(self):
        """Prepare model for quantization-aware training"""
        torch.quantization.prepare_qat(self.vit, inplace=True)

    def convert_quantized(self):
        """Convert to fully quantized model"""
        self.eval()
        quantized_model = torch.quantization.convert(self.vit, inplace=False)
        return quantized_model

    def forward(self, x):
        return self.vit(x)

def create_quantized_model(model_path=None):
    """Create and optimize a quantized vision transformer"""

    # Create model
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=384,  # Smaller for mobile
        depth=12,
        n_heads=6,
        num_classes=1000
    )

    if model_path:
        model.load_state_dict(torch.load(model_path))

    # Quantize the model
    quantized_model = QuantizedVisionTransformer(model)

    # Prepare for quantization (training phase)
    quantized_model.prepare_quantization()

    # After training, convert to fully quantized model
    # final_model = quantized_model.convert_quantized()

    return quantized_model
```

## 4.4 Real-time Inference Optimization

### 4.4.1 TensorRT Integration

For NVIDIA Jetson platforms, TensorRT provides significant performance improvements:

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

class TensorRTVisionTransformer:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.engine = self.load_engine()
        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self.allocate_buffers()

    def load_engine(self):
        """Load TensorRT engine"""
        with open(self.engine_path, 'rb') as f:
            engine_data = f.read()
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        return runtime.deserialize_cuda_engine(engine_data)

    def allocate_buffers(self):
        """Allocate input/output buffers for TensorRT"""
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        for idx in range(self.engine.num_bindings):
            binding_name = self.engine.get_binding_name(idx)
            binding_shape = self.engine.get_binding_shape(idx)
            binding_dtype = trt.nptype(self.engine.get_binding_dtype(idx))

            size = trt.volume(binding_shape) * self.engine.max_batch_size * np.dtype(binding_dtype).itemsize

            host_mem = cuda.pagelocked_empty(size, binding_dtype)
            device_mem = cuda.mem_alloc(size)

            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(idx):
                self.inputs.append({'host': host_mem, 'device': device_mem, 'name': binding_name})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem, 'name': binding_name})

    def infer(self, input_data):
        """Perform inference using TensorRT"""
        # Copy input data to device
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)

        # Run inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # Copy output data to host
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()

        # Return output
        return self.outputs[0]['host'].reshape(self.engine.get_binding_shape(1)[1:])

    def build_engine(self, model_path, input_shape, output_shape, precision='fp16'):
        """Build TensorRT engine from PyTorch model"""
        # This would involve converting PyTorch model to ONNX, then to TensorRT
        # For brevity, we'll outline the process:

        # 1. Export PyTorch model to ONNX
        # 2. Create TensorRT builder and network
        # 3. Parse ONNX model to TensorRT network
        # 4. Optimize for target precision (FP16, INT8)
        # 5. Build engine
        # 6. Serialize engine to file

        pass  # Implementation would be extensive
```

### 4.4.2 Performance Monitoring

```python
import time
import psutil
import GPUtil
from collections import deque
import threading

class VisionTransformerPerformanceMonitor:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.inference_times = deque(maxlen=window_size)
        self.gpu_usages = deque(maxlen=window_size)
        self.memory_usages = deque(maxlen=window_size)
        self.fps_values = deque(maxlen=window_size)

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.monitor_thread.start()

    def record_inference(self, inference_time):
        """Record inference time for performance monitoring"""
        self.inference_times.append(inference_time)

        # Calculate FPS
        if inference_time > 0:
            fps = 1.0 / inference_time
            self.fps_values.append(fps)

    def get_performance_stats(self):
        """Get current performance statistics"""
        if not self.inference_times:
            return {}

        return {
            'avg_inference_time': np.mean(self.inference_times),
            'std_inference_time': np.std(self.inference_times),
            'min_inference_time': min(self.inference_times),
            'max_inference_time': max(self.inference_times),
            'avg_fps': np.mean(self.fps_values) if self.fps_values else 0,
            'current_gpu_usage': self._get_current_gpu_usage(),
            'current_memory_usage': psutil.virtual_memory().percent,
            'sample_count': len(self.inference_times)
        }

    def _get_current_gpu_usage(self):
        """Get current GPU usage"""
        gpus = GPUtil.getGPUs()
        return gpus[0].load if gpus else 0

    def _monitor_system(self):
        """Background monitoring thread"""
        while self.monitoring:
            gpu = GPUtil.getGPUs()
            if gpu:
                self.gpu_usages.append(gpu[0].load * 100)

            self.memory_usages.append(psutil.virtual_memory().percent)
            time.sleep(1.0)  # Update every second

    def should_optimize(self):
        """Check if optimization is needed based on performance"""
        stats = self.get_performance_stats()

        if not stats:
            return False

        # Optimization thresholds
        avg_inference_time = stats.get('avg_inference_time', float('inf'))
        avg_gpu_usage = stats.get('current_gpu_usage', 0)

        # Optimize if inference time is too high or GPU usage is too high
        return (avg_inference_time > 0.1 or  # More than 100ms per inference
                avg_gpu_usage > 90.0)         # GPU usage above 90%
```

## 4.5 Integration with ROS 2

### 4.5.1 ROS 2 Vision Transformer Node

```python
# vision_transformer_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose
from std_msgs.msg import Header
from cv_bridge import CvBridge
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image as PILImage

class VisionTransformerNode(Node):
    def __init__(self):
        super().__init__('vision_transformer_node')

        # Initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()
        self.model.to(self.device)
        self.model.eval()

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Initialize transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10
        )

        self.detection_pub = self.create_publisher(
            Detection2DArray,
            'vision_transformer/detections',
            10
        )

        # Performance monitoring
        self.perf_monitor = VisionTransformerPerformanceMonitor()

        # Class labels (COCO dataset for example)
        self.class_labels = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
            'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
            'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
            'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
            'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        self.get_logger().info('Vision Transformer Node initialized')

    def load_model(self):
        """Load the vision transformer model"""
        # For DETR-based model
        model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
        return model

    def image_callback(self, msg):
        """Process incoming image messages"""
        start_time = time.time()

        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

            # Convert to PIL Image
            pil_image = PILImage.fromarray(cv_image)

            # Apply transforms
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor)

                # Process outputs (this would depend on the specific model)
                detections = self.process_outputs(outputs, cv_image.shape[:2])

            # Publish detections
            self.publish_detections(detections, msg.header)

            # Record performance
            inference_time = time.time() - start_time
            self.perf_monitor.record_inference(inference_time)

            self.get_logger().debug(f'Processed image in {inference_time:.3f}s')

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def process_outputs(self, outputs, image_shape):
        """Process model outputs to create detections"""
        # This would depend on the specific model (DETR, ViT, etc.)
        # For DETR, we would process the predicted boxes and logits
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]  # Remove "no object" class
        boxes = outputs['pred_boxes'][0]

        # Convert boxes from normalized coordinates to pixel coordinates
        h, w = image_shape
        boxes = boxes * torch.tensor([w, h, w, h], dtype=torch.float32)

        # Get top predictions
        topk = min(10, len(probas))  # Limit to top 10 detections
        probas, indices = torch.topk(probas.max(-1)[0], topk)
        boxes = boxes[indices]

        detections = []
        for i in range(len(boxes)):
            box = boxes[i]
            conf = probas[i]

            # Get predicted class
            pred_class = probas.max(-1)[1][i]
            class_name = self.class_labels[pred_class] if pred_class < len(self.class_labels) else 'unknown'

            detection = {
                'class_name': class_name,
                'confidence': conf.item(),
                'bbox': [box[0].item(), box[1].item(), box[2].item(), box[3].item()]  # [x, y, w, h]
            }
            detections.append(detection)

        return detections

    def publish_detections(self, detections, header):
        """Publish detections as ROS messages"""
        detection_array = Detection2DArray()
        detection_array.header = header

        for det in detections:
            detection_msg = Detection2D()
            detection_msg.header = header
            detection_msg.results = []

            # Create hypothesis
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.id = det['class_name']
            hypothesis.score = det['confidence']

            detection_msg.results.append(hypothesis)

            # Set bounding box (convert from [x, y, w, h] to the format expected)
            # This would depend on the specific vision_msgs format
            detection_msg.bbox.center.x = det['bbox'][0] + det['bbox'][2] / 2  # center x
            detection_msg.bbox.center.y = det['bbox'][1] + det['bbox'][3] / 2  # center y
            detection_msg.bbox.size_x = det['bbox'][2]  # width
            detection_msg.bbox.size_y = det['bbox'][3]  # height

            detection_array.detections.append(detection_msg)

        self.detection_pub.publish(detection_array)

def main(args=None):
    rclpy.init(args=args)
    node = VisionTransformerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 4.6 Advanced Applications in Robotics

### 4.6.1 Scene Understanding for Navigation

```python
class SceneUnderstandingTransformer(nn.Module):
    def __init__(self, vit_model, num_classes=50):
        super().__init__()
        self.vit = vit_model
        self.num_classes = num_classes

        # Scene classification head
        self.scene_classifier = nn.Linear(vit_model.embed_dim, num_classes)

        # Spatial layout prediction head
        self.layout_predictor = nn.Sequential(
            nn.Linear(vit_model.embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)  # Predict 10 key spatial features
        )

        # Traversable region prediction head
        self.traversability_head = nn.Sequential(
            nn.Conv2d(vit_model.embed_dim, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1),  # Binary traversability mask
            nn.Sigmoid()
        )

    def forward(self, x):
        # Get features from ViT
        features = self.vit(x)

        # Scene classification
        scene_logits = self.scene_classifier(features)

        # Layout prediction
        layout_features = self.layout_predictor(features)

        # Traversability prediction (this would require reshaping features)
        # For simplicity, we'll use a different approach
        traversability = self.predict_traversability(x)

        return {
            'scene_classification': scene_logits,
            'layout_features': layout_features,
            'traversability_map': traversability
        }

    def predict_traversability(self, x):
        """Predict traversability map using a separate pathway"""
        # This would typically use a decoder network
        # For now, return a simple prediction
        batch_size = x.size(0)
        h, w = x.size(2), x.size(3)
        return torch.zeros(batch_size, 1, h, w).to(x.device)

class NavigationSceneAnalyzer:
    def __init__(self, scene_model):
        self.model = scene_model
        self.model.eval()

    def analyze_scene_for_navigation(self, image_tensor):
        """Analyze scene for navigation purposes"""
        with torch.no_grad():
            outputs = self.model(image_tensor)

        # Process outputs for navigation
        scene_type = torch.argmax(outputs['scene_classification'], dim=1)
        traversability_map = outputs['traversability_map']

        # Generate navigation-relevant information
        nav_info = {
            'scene_type': scene_type.item(),
            'traversable_regions': self.extract_traversable_regions(traversability_map),
            'obstacle_locations': self.detect_obstacles(image_tensor),
            'navigation_path_recommendations': self.suggest_navigation_paths(traversability_map)
        }

        return nav_info

    def extract_traversable_regions(self, traversability_map):
        """Extract regions that are traversable"""
        # Threshold to determine traversable regions
        traversable_mask = traversability_map > 0.5
        return traversable_mask

    def detect_obstacles(self, image):
        """Detect obstacles using the vision transformer"""
        # This would integrate with object detection capabilities
        # For now, return a placeholder
        return []

    def suggest_navigation_paths(self, traversability_map):
        """Suggest potential navigation paths based on traversability"""
        # This would implement path planning algorithms
        # For now, return a placeholder
        return []
```

### 4.6.2 Manipulation Object Recognition

```python
class ManipulationObjectDetector:
    def __init__(self, vit_model, manipulation_classes):
        self.model = vit_model
        self.manipulation_classes = manipulation_classes
        self.model.eval()

        # Manipulation-specific heads
        self.graspability_head = nn.Sequential(
            nn.Linear(vit_model.embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Probability of graspability
        )

        self.pose_estimator = nn.Sequential(
            nn.Linear(vit_model.embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 6)  # 3D position + 3D orientation
        )

    def detect_manipulation_targets(self, image_tensor):
        """Detect objects suitable for manipulation"""
        with torch.no_grad():
            features = self.model(image_tensor)

            # Get class predictions
            class_logits = self.model.head(features)
            class_probs = torch.softmax(class_logits, dim=1)

            # Get graspability scores
            graspability_scores = self.graspability_head(features)

            # Get pose estimates
            pose_estimates = self.pose_estimator(features)

        # Filter for manipulable objects
        manipulable_objects = []
        for i, (prob, grasp_score) in enumerate(zip(class_probs[0], graspability_scores[0])):
            class_idx = torch.argmax(prob)
            class_name = self.manipulation_classes[class_idx] if class_idx < len(self.manipulation_classes) else 'unknown'

            if prob[class_idx] > 0.5 and grasp_score > 0.7:  # Confidence thresholds
                obj_info = {
                    'class_name': class_name,
                    'confidence': prob[class_idx].item(),
                    'graspability': grasp_score.item(),
                    'pose_estimate': pose_estimates[0].cpu().numpy()
                }
                manipulable_objects.append(obj_info)

        return manipulable_objects
```

## 4.7 Quality Assurance and Validation

### 4.7.1 Testing Framework

```python
import unittest
from vision_transformer_node import VisionTransformerNode

class TestVisionTransformerNode(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # We'll mock the ROS node for testing
        pass

    def test_model_loading(self):
        """Test that the model loads correctly"""
        node = VisionTransformerNode()
        self.assertIsNotNone(node.model)
        self.assertTrue(hasattr(node.model, 'eval'))

    def test_image_processing(self):
        """Test image processing pipeline"""
        # Create a dummy image
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        node = VisionTransformerNode()

        # Test that transforms work correctly
        pil_image = Image.fromarray(dummy_image)
        transformed = node.transform(pil_image)

        self.assertEqual(transformed.shape, (3, 224, 224))  # Expected shape after transforms

    def test_performance_monitoring(self):
        """Test performance monitoring functionality"""
        monitor = VisionTransformerPerformanceMonitor(window_size=10)

        # Record some inference times
        for _ in range(10):
            monitor.record_inference(0.05)  # 50ms per inference

        stats = monitor.get_performance_stats()

        self.assertAlmostEqual(stats['avg_inference_time'], 0.05, places=3)
        self.assertAlmostEqual(stats['avg_fps'], 20.0, places=1)

class VisionTransformerValidator:
    def __init__(self, model, test_dataset):
        self.model = model
        self.test_dataset = test_dataset
        self.model.eval()

    def validate_model_performance(self):
        """Validate model performance on test dataset"""
        correct = 0
        total = 0
        inference_times = []

        with torch.no_grad():
            for images, targets in self.test_dataset:
                start_time = time.time()
                outputs = self.model(images)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)

                # Calculate accuracy (simplified)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        accuracy = 100 * correct / total
        avg_inference_time = sum(inference_times) / len(inference_times)

        return {
            'accuracy': accuracy,
            'avg_inference_time': avg_inference_time,
            'total_samples': total
        }

    def validate_real_time_constraints(self, target_fps=30):
        """Validate that model meets real-time constraints"""
        target_inference_time = 1.0 / target_fps

        # Test with batch size 1 (worst case for real-time)
        dummy_input = torch.randn(1, 3, 224, 224)

        inference_times = []
        for _ in range(100):  # Test 100 times
            start_time = time.time()
            with torch.no_grad():
                _ = self.model(dummy_input)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

        avg_time = sum(inference_times) / len(inference_times)
        meets_constraint = avg_time <= target_inference_time

        return {
            'meets_real_time': meets_constraint,
            'actual_fps': 1.0 / avg_time,
            'target_fps': target_fps,
            'avg_inference_time': avg_time
        }
```

## 4.8 Best Practices for Robotics Applications

### 4.8.1 Model Selection Guidelines

1. **Computational Constraints**: Choose model size based on target hardware
2. **Latency Requirements**: Consider inference speed for real-time applications
3. **Accuracy vs. Speed**: Balance accuracy with computational efficiency
4. **Robustness**: Test performance under various lighting and environmental conditions

### 4.8.2 Deployment Considerations

1. **Edge Optimization**: Use quantization and pruning for deployment
2. **Memory Management**: Monitor and optimize GPU/CPU memory usage
3. **Thermal Management**: Consider heat dissipation in mobile robots
4. **Power Consumption**: Optimize for battery-powered systems

## Summary

Vision Transformers provide powerful capabilities for robotic vision systems, offering superior performance in object detection, scene understanding, and real-time perception tasks. By implementing optimized architectures, leveraging hardware acceleration, and following best practices for deployment, humanoid robots can achieve sophisticated visual perception capabilities. The integration with ROS 2 enables seamless incorporation into robotic systems, while proper validation ensures reliable operation in real-world environments. In the next chapter, we will explore multimodal fusion techniques that combine vision, language, and other sensory modalities for enhanced robotic intelligence.