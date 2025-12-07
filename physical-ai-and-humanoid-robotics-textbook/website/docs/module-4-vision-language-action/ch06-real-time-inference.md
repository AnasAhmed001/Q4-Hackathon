---
title: Chapter 6 - Real-time Inference Optimization
description: Learn techniques for optimizing neural network inference for real-time robotic applications, including quantization, pruning, and hardware acceleration.
sidebar_position: 40
---

# Chapter 6 - Real-time Inference Optimization

Real-time inference optimization is critical for humanoid robots, where computational resources are limited, and responsiveness is essential for safety and effective interaction. This chapter explores various optimization techniques to ensure neural networks can run efficiently on robotic platforms while maintaining the accuracy required for complex tasks like vision-language-action integration.

## 6.1 Introduction to Real-time Inference Challenges

Real-time inference on humanoid robots faces several unique challenges:

- **Limited Computational Resources**: Embedded GPUs like Jetson Orin have constrained memory and processing power
- **Strict Latency Requirements**: Robot control systems often require responses within 10-30ms
- **Power Consumption**: Battery-powered robots need to optimize energy efficiency
- **Thermal Constraints**: Compact robot bodies limit cooling options
- **Multiple Concurrent Tasks**: Robots must run perception, planning, and control simultaneously

### 6.1.1 Performance Requirements for Robotics

For humanoid robots, typical performance requirements include:

- **Vision Processing**: 30 FPS for real-time perception
- **Language Understanding**: < 100ms response time for natural interaction
- **Action Planning**: < 50ms for responsive behavior
- **System Integration**: < 10ms for control loops

### 6.1.2 Hardware Considerations

Different robotic platforms have varying computational capabilities:

| Platform | GPU | Memory | Peak Power | Recommended Models |
|----------|-----|---------|------------|-------------------|
| Jetson Orin | 2048 CUDA cores | 8-64GB LPDDR5 | 60W | MobileViT, EfficientNet, TinyBERT |
| Jetson Xavier | 512 CUDA cores | 8-32GB LPDDR4x | 30W | EfficientNets, MobileNets |
| Desktop GPU | 1000s+ CUDA cores | 8-24GB GDDR6 | 200-400W | Full-size transformers, ResNets |

## 6.2 Model Quantization Techniques

### 6.2.1 Post-Training Quantization

Post-training quantization is the most common approach for optimizing pre-trained models:

```python
import torch
import torch.quantization as quant
import torch.nn as nn
from torch.quantization import get_default_qconfig, prepare, convert
import numpy as np

class PostTrainingQuantizer:
    def __init__(self, model, calib_data_loader, device='cuda'):
        self.model = model
        self.calib_loader = calib_data_loader
        self.device = device

    def quantize_model(self, backend='fbgemm', observer_type='histogram'):
        """
        Quantize model using post-training quantization
        """
        # Set model to evaluation mode
        self.model.eval()

        # Select quantization configuration
        if observer_type == 'histogram':
            qconfig = torch.quantization.get_default_qconfig(backend)
        else:
            qconfig = torch.quantization.default_qconfig

        # Set qconfig for the model
        self.model.qconfig = qconfig

        # Prepare model for quantization
        model_prepared = prepare(self.model, inplace=False)

        # Calibrate the model with sample data
        print("Calibrating model...")
        with torch.no_grad():
            for i, (inputs, _) in enumerate(self.calib_loader):
                if i >= 100:  # Use first 100 batches for calibration
                    break
                inputs = inputs.to(self.device)
                model_prepared(inputs)

        # Convert to quantized model
        quantized_model = convert(model_prepared, inplace=False)

        print("Model quantized successfully!")
        return quantized_model

    def evaluate_quantization_impact(self, test_loader, original_model):
        """
        Compare accuracy and performance of original vs quantized models
        """
        original_model.eval()
        self.model.eval()  # This is now the quantized model

        original_accuracies = []
        quantized_accuracies = []

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Original model
                start_time = time.time()
                orig_outputs = original_model(inputs)
                orig_time = time.time() - start_time
                orig_acc = self.calculate_accuracy(orig_outputs, targets)

                # Quantized model
                start_time = time.time()
                quant_outputs = self.model(inputs)
                quant_time = time.time() - start_time
                quant_acc = self.calculate_accuracy(quant_outputs, targets)

                original_accuracies.append(orig_acc)
                quantized_accuracies.append(quant_acc)

        print(f"Original Model - Accuracy: {np.mean(original_accuracies):.4f}, Time: {np.mean(orig_time):.4f}s")
        print(f"Quantized Model - Accuracy: {np.mean(quantized_accuracies):.4f}, Time: {np.mean(quant_time):.4f}s")
        print(f"Speedup: {np.mean(orig_time) / np.mean(quant_time):.2f}x")

    def calculate_accuracy(self, outputs, targets):
        """Calculate top-1 accuracy"""
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == targets).sum().item()
        total = targets.size(0)
        return correct / total

# Example usage
def quantize_vision_transformer(vit_model, calib_loader):
    """Quantize a Vision Transformer model"""
    quantizer = PostTrainingQuantizer(vit_model, calib_loader)

    # For Vision Transformers, we might need special handling
    # because of LayerNorm and other components
    vit_model.eval()

    # Set quantization configuration
    vit_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    # Fuse operations for better quantization
    vit_model = torch.quantization.fuse_modules(
        vit_model,
        [['patch_embed.proj', 'norm']],  # Example fusion
        inplace=True
    )

    # Prepare for quantization
    vit_model_prepared = prepare(vit_model, inplace=False)

    # Calibrate
    with torch.no_grad():
        for i, (inputs, _) in enumerate(calib_loader):
            if i >= 50:  # Use 50 batches for calibration
                break
            vit_model_prepared(inputs)

    # Convert to quantized model
    quantized_vit = convert(vit_model_prepared, inplace=False)

    return quantized_vit
```

### 6.2.2 Quantization-Aware Training

For better accuracy preservation, quantization-aware training simulates quantization during training:

```python
class QuantizationAwareTraining(nn.Module):
    def __init__(self, model, quantization_bits=8):
        super().__init__()
        self.model = model
        self.bits = quantization_bits

        # Quantization modules
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        # Observer modules for calibration
        self.activation_observer = torch.quantization.MinMaxObserver(
            dtype=torch.qint8,
            qscheme=torch.per_tensor_affine
        )
        self.weight_observer = torch.quantization.MinMaxObserver(
            dtype=torch.qint8,
            qscheme=torch.per_channel_affine
        )

    def forward(self, x):
        # Quantize input
        x = self.quant(x)

        # Forward through model
        x = self.model(x)

        # Dequantize output
        x = self.dequant(x)
        return x

    def prepare_qat(self):
        """Prepare model for quantization-aware training"""
        self.model.train()

        # Set qconfig for QAT
        self.model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

        # Prepare model for QAT
        torch.quantization.prepare_qat(self.model, inplace=True)

    def convert_quantized(self):
        """Convert to fully quantized model after training"""
        self.model.eval()
        quantized_model = torch.quantization.convert(self.model, inplace=True)
        return quantized_model

class QATOptimizer:
    def __init__(self, model, learning_rate=1e-5):
        self.model = QuantizationAwareTraining(model)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )
        self.criterion = nn.CrossEntropyLoss()

    def train_step(self, inputs, targets):
        """Single training step with QAT"""
        self.model.train()

        # Forward pass
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate_step(self, inputs, targets):
        """Evaluation step"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            accuracy = self.calculate_accuracy(outputs, targets)
        return loss.item(), accuracy

    def calculate_accuracy(self, outputs, targets):
        """Calculate accuracy"""
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == targets).sum().item()
        total = targets.size(0)
        return correct / total
```

### 6.2.3 Mixed Precision Training

Mixed precision training uses both FP16 and FP32 to optimize performance:

```python
from torch.cuda.amp import GradScaler, autocast

class MixedPrecisionTrainer:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.scaler = GradScaler()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )
        self.criterion = nn.CrossEntropyLoss()

    def train_step(self, inputs, targets):
        """Training step with mixed precision"""
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        self.optimizer.zero_grad()

        # Use autocast for mixed precision
        with autocast():
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

        # Scale loss and backward
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item()

    def evaluate_step(self, inputs, targets):
        """Evaluation step with mixed precision"""
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        with torch.no_grad():
            with autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

        accuracy = self.calculate_accuracy(outputs, targets)
        return loss.item(), accuracy

    def calculate_accuracy(self, outputs, targets):
        """Calculate accuracy"""
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == targets).sum().item()
        total = targets.size(0)
        return correct / total
```

## 6.3 Model Pruning Techniques

### 6.3.1 Structured Pruning

Structured pruning removes entire channels or layers, which is more hardware-friendly:

```python
import torch.nn.utils.prune as prune

class StructuredPruner:
    def __init__(self, model):
        self.model = model

    def prune_channels(self, proportion=0.2, method='ln_structured'):
        """
        Prune channels from convolutional layers
        """
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Apply structured pruning
                prune.ln_structured(
                    module,
                    name='weight',
                    amount=proportion,
                    n=2,  # L2 norm
                    dim=0  # Prune output channels
                )

    def prune_neurons(self, proportion=0.2):
        """
        Prune neurons from linear layers
        """
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(
                    module,
                    name='weight',
                    amount=proportion
                )

    def fine_tune_after_pruning(self, train_loader, epochs=5):
        """
        Fine-tune model after pruning to recover accuracy
        """
        self.model.train()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-5  # Lower learning rate for fine-tuning
        )
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            total_loss = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to('cuda'), targets.to('cuda')

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Fine-tuning Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    def remove_pruning_masks(self):
        """
        Remove pruning masks to make model permanent
        """
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight_orig'):
                prune.remove(module, 'weight')

    def calculate_sparsity(self):
        """
        Calculate overall sparsity of the model
        """
        total_params = 0
        zero_params = 0

        for name, param in self.model.named_parameters():
            total_params += param.numel()
            zero_params += torch.sum(param == 0).item()

        sparsity = zero_params / total_params if total_params > 0 else 0
        return sparsity

def create_pruned_vision_transformer(vit_model, sparsity_target=0.3):
    """
    Create a pruned version of Vision Transformer
    """
    pruner = StructuredPruner(vit_model)

    # Prune different types of layers with different strategies
    for name, module in vit_model.named_modules():
        if isinstance(module, nn.Linear):
            # Prune linear layers (more aggressive)
            prune.l1_unstructured(module, name='weight', amount=sparsity_target)
        elif isinstance(module, nn.Conv2d):
            # For Conv layers in ViT (like patch embedding)
            prune.ln_structured(module, name='weight', amount=sparsity_target, n=2, dim=0)

    # Remove pruning masks to make pruning permanent
    for name, module in vit_model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)) and hasattr(module, 'weight_orig'):
            prune.remove(module, 'weight')

    return vit_model
```

### 6.3.2 Unstructured Pruning

Unstructured pruning can achieve higher sparsity but requires specialized hardware support:

```python
class UnstructuredPruner:
    def __init__(self, model):
        self.model = model

    def magnitude_pruning(self, sparsity_level=0.5):
        """
        Prune weights based on magnitude
        """
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Calculate threshold for pruning
                flattened_weights = module.weight.data.view(-1)
                threshold = torch.quantile(torch.abs(flattened_weights), sparsity_level)

                # Create mask
                mask = torch.abs(module.weight.data) >= threshold

                # Apply mask
                module.weight.data *= mask

    def iterative_pruning(self, train_loader, initial_sparsity=0.1, final_sparsity=0.5,
                         pruning_iterations=5, fine_tune_epochs=2):
        """
        Iteratively prune and fine-tune the model
        """
        current_sparsity = initial_sparsity
        step_size = (final_sparsity - initial_sparsity) / pruning_iterations

        for iteration in range(pruning_iterations):
            # Calculate target sparsity for this iteration
            target_sparsity = initial_sparsity + (iteration + 1) * step_size

            # Prune to target sparsity
            self.magnitude_pruning(target_sparsity)

            # Fine-tune after pruning
            self.fine_tune(train_loader, epochs=fine_tune_epochs)

            print(f"Iteration {iteration+1}: Sparsity = {target_sparsity:.2f}, "
                  f"Current sparsity = {self.calculate_sparsity():.2f}")

    def fine_tune(self, train_loader, epochs=2):
        """
        Fine-tune model after pruning
        """
        self.model.train()
        optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=1e-5
        )
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            total_loss = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to('cuda'), targets.to('cuda')

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Fine-tuning Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    def calculate_sparsity(self):
        """
        Calculate current sparsity level
        """
        total_params = 0
        zero_params = 0

        for param in self.model.parameters():
            total_params += param.numel()
            zero_params += torch.sum(param == 0).item()

        return zero_params / total_params if total_params > 0 else 0
```

## 6.4 Knowledge Distillation

Knowledge distillation creates smaller, faster student models that maintain teacher model performance:

```python
class KnowledgeDistillationTrainer:
    def __init__(self, teacher_model, student_model, device='cuda'):
        self.teacher = teacher_model.to(device).eval()
        self.student = student_model.to(device)
        self.device = device

        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

        # Optimizer
        self.optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)

    def distillation_loss(self, student_outputs, teacher_outputs, targets,
                        alpha=0.7, temperature=4.0):
        """
        Calculate distillation loss combining hard and soft targets
        """
        # Soft target loss (KL divergence)
        soft_targets = F.softmax(teacher_outputs / temperature, dim=1)
        soft_predictions = F.log_softmax(student_outputs / temperature, dim=1)
        soft_loss = F.kl_div(soft_predictions, soft_targets, reduction='batchmean')

        # Hard target loss (cross-entropy)
        hard_loss = self.ce_loss(student_outputs, targets)

        # Combined loss
        total_loss = alpha * soft_loss + (1 - alpha) * hard_loss

        return total_loss

    def train_step(self, inputs, targets):
        """
        Single training step for knowledge distillation
        """
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        # Get teacher outputs (no gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher(inputs)

        # Get student outputs
        student_outputs = self.student(inputs)

        # Calculate distillation loss
        loss = self.distillation_loss(student_outputs, teacher_outputs, targets)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self, test_loader):
        """
        Evaluate student model performance
        """
        self.student.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.student(inputs)

                loss = self.ce_loss(outputs, targets)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        accuracy = correct / total
        avg_loss = total_loss / len(test_loader)

        return avg_loss, accuracy

def create_student_model(teacher_model, compression_ratio=4):
    """
    Create a smaller student model based on teacher architecture
    """
    # Example: Create a smaller Vision Transformer
    student_model = VisionTransformer(
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=teacher_model.num_classes,
        embed_dim=teacher_model.embed_dim // compression_ratio,  # Reduced embedding
        depth=teacher_model.depth // 2,  # Reduced depth
        n_heads=teacher_model.n_heads // 2,  # Reduced heads
        mlp_ratio=teacher_model.mlp_ratio
    )
    return student_model

class ProgressiveDistillation:
    """
    Progressive knowledge distillation with gradual complexity increase
    """
    def __init__(self, teacher, student, device='cuda'):
        self.teacher = teacher.to(device).eval()
        self.student = student.to(device)
        self.device = device

    def progressive_train(self, train_loaders, epochs_per_stage=10):
        """
        Train student progressively with increasing complexity
        train_loaders: List of data loaders with increasing complexity
        """
        for stage, loader in enumerate(train_loaders):
            print(f"Stage {stage + 1}: Training with increasing complexity")

            # Adjust training parameters based on stage
            if stage == 0:
                # Start with high temperature for soft targets
                temperature = 8.0
                alpha = 0.9
            else:
                # Reduce temperature and increase hard target weight
                temperature = max(2.0, 8.0 - stage)
                alpha = max(0.5, 0.9 - stage * 0.1)

            trainer = KnowledgeDistillationTrainer(self.teacher, self.student, self.device)

            for epoch in range(epochs_per_stage):
                total_loss = 0
                for inputs, targets in loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    # Get teacher outputs
                    with torch.no_grad():
                        teacher_outputs = self.teacher(inputs)

                    # Student outputs
                    student_outputs = self.student(inputs)

                    # Calculate loss with current temperature and alpha
                    loss = trainer.distillation_loss(
                        student_outputs, teacher_outputs, targets,
                        alpha=alpha, temperature=temperature
                    )

                    # Backward pass
                    trainer.optimizer.zero_grad()
                    loss.backward()
                    trainer.optimizer.step()

                    total_loss += loss.item()

                print(f"Stage {stage+1}, Epoch {epoch+1}: Loss = {total_loss/len(loader):.4f}")

        return self.student
```

## 6.5 Hardware Acceleration

### 6.5.1 TensorRT Optimization

TensorRT provides significant acceleration for NVIDIA GPUs:

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

class TensorRTOptimizer:
    def __init__(self, model_path=None):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

    def build_engine(self, onnx_model_path, batch_size=1, precision='fp16'):
        """
        Build TensorRT engine from ONNX model
        """
        builder = trt.Builder(self.logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self.logger)

        # Parse ONNX model
        with open(onnx_model_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        # Configure builder
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB

        # Set precision
        if precision == 'fp16':
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            else:
                print("FP16 not supported, using FP32")

        # Set dynamic shapes if needed
        profile = builder.create_optimization_profile()
        # Example for image input [batch, channels, height, width]
        profile.set_shape("input",
                         min=(1, 3, 224, 224),
                         opt=(batch_size, 3, 224, 224),
                         max=(batch_size, 3, 224, 224))
        config.add_optimization_profile(profile)

        # Build engine
        serialized_engine = builder.build_serialized_network(network, config)
        engine = self.runtime.deserialize_cuda_engine(serialized_engine)

        return engine

    def create_trt_inference_session(self, engine):
        """
        Create inference session from TensorRT engine
        """
        return TrtInferenceSession(engine)

class TrtInferenceSession:
    def __init__(self, engine):
        self.engine = engine
        self.context = self.engine.create_execution_context()

        # Allocate buffers
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
                self.inputs.append({
                    'host': host_mem,
                    'device': device_mem,
                    'name': binding_name,
                    'shape': binding_shape
                })
            else:
                self.outputs.append({
                    'host': host_mem,
                    'device': device_mem,
                    'name': binding_name,
                    'shape': binding_shape
                })

    def infer(self, input_data):
        """
        Perform inference using TensorRT engine
        """
        # Copy input data to device
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)

        # Run inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # Copy output data to host
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()

        # Return output
        output_shape = self.outputs[0]['shape']
        output_data = self.outputs[0]['host'].reshape(output_shape)

        return output_data

def optimize_model_with_tensorrt(pytorch_model, input_shape, output_path):
    """
    Full pipeline: PyTorch -> ONNX -> TensorRT
    """
    import torch.onnx as torch_onnx

    # Set model to evaluation mode
    pytorch_model.eval()

    # Create dummy input
    dummy_input = torch.randn(input_shape).cuda()

    # Export to ONNX
    onnx_path = output_path.replace('.engine', '.onnx')
    torch_onnx.export(
        pytorch_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    # Optimize with TensorRT
    optimizer = TensorRTOptimizer()
    engine = optimizer.build_engine(onnx_path, batch_size=input_shape[0])

    # Serialize engine
    with open(output_path, 'wb') as f:
        f.write(engine.serialize())

    print(f"TensorRT engine saved to {output_path}")
    return engine
```

### 6.5.2 NVIDIA Triton Inference Server

For production deployment, Triton Inference Server provides optimized serving:

```python
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
import numpy as np

class TritonInferenceClient:
    def __init__(self, server_url="localhost:8000"):
        self.client = httpclient.InferenceServerClient(url=server_url)

    def infer(self, model_name, input_data, model_version="1"):
        """
        Perform inference using Triton server
        """
        # Prepare input
        inputs = []
        for i, data in enumerate(input_data):
            input_name = f"input_{i}" if len(input_data) > 1 else "input"
            inputs.append(httpclient.InferInput(input_name, data.shape, "FP32"))
            inputs[-1].set_data_from_numpy(data)

        # Perform inference
        try:
            response = self.client.infer(
                model_name=model_name,
                inputs=inputs,
                model_version=model_version
            )

            # Get output
            output_names = [output.name for output in response._result.outputs]
            results = {}
            for name in output_names:
                results[name] = response.as_numpy(name)

            return results

        except InferenceServerException as e:
            print(f"Inference failed: {e}")
            return None

# Example Triton model configuration (config.pbtxt)
triton_config = """
name: "vision_transformer"
platform: "tensorrt_plan"
max_batch_size: 8
input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [-1, 3, 224, 224]
  }
]
output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [-1, 1000]
  }
]
optimization {
  execution_accelerators {
    gpu_execution_accelerator : [ {
      name : "tensorrt"
      parameters { key: "precision_mode" value: "FP16" }
    }]
  }
}
"""
```

## 6.6 Dynamic Batching and Scheduling

### 6.6.1 Dynamic Batch Size Adjustment

```python
import asyncio
import time
from collections import deque
from typing import List, Tuple

class DynamicBatchScheduler:
    def __init__(self, min_batch_size=1, max_batch_size=8, timeout=0.01):
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.timeout = timeout

        self.request_queue = asyncio.Queue()
        self.pending_requests = deque()
        self.model = None
        self.is_running = False

    async def add_request(self, input_data):
        """
        Add request to scheduler
        """
        future = asyncio.Future()
        request = (input_data, future)
        await self.request_queue.put(request)
        return await future

    async def process_batches(self, model):
        """
        Process requests in dynamic batches
        """
        self.model = model
        self.is_running = True

        while self.is_running:
            try:
                # Wait for first request
                first_request = await asyncio.wait_for(
                    self.request_queue.get(),
                    timeout=self.timeout
                )
                self.pending_requests.append(first_request)

                # Collect more requests within timeout
                start_time = time.time()
                while (time.time() - start_time < self.timeout and
                       len(self.pending_requests) < self.max_batch_size):
                    try:
                        request = await asyncio.wait_for(
                            self.request_queue.get(),
                            timeout=max(0, self.timeout - (time.time() - start_time))
                        )
                        self.pending_requests.append(request)
                    except asyncio.TimeoutError:
                        break

                # Process batch
                await self.process_batch()

            except asyncio.TimeoutError:
                # Process any accumulated requests even if below min batch size
                if self.pending_requests:
                    await self.process_batch()

    async def process_batch(self):
        """
        Process a batch of requests
        """
        if not self.pending_requests:
            return

        # Collect inputs and futures
        inputs = []
        futures = []

        for _ in range(len(self.pending_requests)):
            input_data, future = self.pending_requests.popleft()
            inputs.append(input_data)
            futures.append(future)

        # Pad batch if necessary
        while len(inputs) < self.max_batch_size:
            inputs.append(inputs[0])  # Duplicate first input

        # Stack inputs
        batch_input = torch.stack(inputs)

        # Run inference
        with torch.no_grad():
            outputs = self.model(batch_input)

        # Return results to futures
        for i, future in enumerate(futures):
            result = outputs[i:i+1]  # Extract individual output
            future.set_result(result)

    def stop(self):
        """
        Stop the scheduler
        """
        self.is_running = False

class AdaptiveInferenceManager:
    def __init__(self, model, target_latency=0.05, max_memory=1000):
        self.model = model
        self.target_latency = target_latency
        self.max_memory = max_memory

        self.latency_history = deque(maxlen=100)
        self.throughput_history = deque(maxlen=100)
        self.batch_sizes = deque(maxlen=100)

        self.current_batch_size = 1
        self.current_latency = 0

    def monitor_performance(self, batch_size, latency, throughput):
        """
        Monitor and adjust inference parameters
        """
        self.batch_sizes.append(batch_size)
        self.latency_history.append(latency)
        self.throughput_history.append(throughput)

        # Calculate recent averages
        recent_latency = np.mean(list(self.latency_history)[-10:])
        recent_throughput = np.mean(list(self.throughput_history)[-10:])

        # Adjust batch size based on performance
        if recent_latency > self.target_latency * 1.2:
            # Too slow, reduce batch size
            self.current_batch_size = max(1, self.current_batch_size - 1)
        elif recent_latency < self.target_latency * 0.8 and len(self.batch_sizes) > 50:
            # Fast enough, try larger batch
            self.current_batch_size = min(8, self.current_batch_size + 1)

    def get_optimal_batch_size(self, incoming_requests):
        """
        Determine optimal batch size based on current conditions
        """
        if len(self.latency_history) < 10:
            return min(self.current_batch_size, len(incoming_requests))

        avg_latency = np.mean(list(self.latency_history)[-10:])

        if avg_latency > self.target_latency:
            # Prioritize latency, reduce batch size
            return max(1, min(self.current_batch_size - 2, len(incoming_requests)))
        else:
            # Can handle more, increase batch size
            return min(self.current_batch_size + 1, len(incoming_requests), 8)

    async def adaptive_inference(self, inputs):
        """
        Perform inference with adaptive batching
        """
        optimal_batch_size = self.get_optimal_batch_size(len(inputs))

        results = []
        start_time = time.time()

        # Process in optimal-sized batches
        for i in range(0, len(inputs), optimal_batch_size):
            batch_inputs = inputs[i:i+optimal_batch_size]

            # Pad if necessary
            padded_inputs = batch_inputs + [batch_inputs[0]] * (optimal_batch_size - len(batch_inputs))

            batch_tensor = torch.stack(padded_inputs)

            with torch.no_grad():
                batch_outputs = self.model(batch_tensor)

            # Take only the actual outputs
            actual_outputs = batch_outputs[:len(batch_inputs)]
            results.extend(actual_outputs)

        end_time = time.time()
        latency = end_time - start_time

        # Monitor performance
        throughput = len(inputs) / latency
        self.monitor_performance(optimal_batch_size, latency, throughput)

        return results
```

## 6.7 Performance Monitoring and Profiling

### 6.7.1 Real-time Performance Monitoring

```python
import psutil
import GPUtil
import time
from collections import deque
import threading
import json

class PerformanceMonitor:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.inference_times = deque(maxlen=window_size)
        self.gpu_loads = deque(maxlen=window_size)
        self.gpu_memory = deque(maxlen=window_size)
        self.cpu_loads = deque(maxlen=window_size)
        self.memory_usage = deque(maxlen=window_size)
        self.throughput_values = deque(maxlen=window_size)

        self.lock = threading.Lock()
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.monitor_thread.start()

    def record_inference(self, inference_time, batch_size=1):
        """
        Record inference performance metrics
        """
        with self.lock:
            self.inference_times.append(inference_time)

            # Calculate throughput
            if inference_time > 0:
                throughput = batch_size / inference_time
                self.throughput_values.append(throughput)

    def get_performance_summary(self):
        """
        Get current performance summary
        """
        with self.lock:
            if not self.inference_times:
                return {}

            summary = {
                'inference_stats': {
                    'avg_time': np.mean(self.inference_times),
                    'min_time': min(self.inference_times),
                    'max_time': max(self.inference_times),
                    'std_time': np.std(self.inference_times),
                    'count': len(self.inference_times)
                },
                'throughput_stats': {
                    'avg_throughput': np.mean(self.throughput_values) if self.throughput_values else 0,
                    'current_throughput': self.throughput_values[-1] if self.throughput_values else 0
                },
                'system_stats': {
                    'avg_cpu_load': np.mean(self.cpu_loads) if self.cpu_loads else 0,
                    'avg_gpu_load': np.mean(self.gpu_loads) if self.gpu_loads else 0,
                    'avg_gpu_memory': np.mean(self.gpu_memory) if self.gpu_memory else 0,
                    'avg_memory_usage': np.mean(self.memory_usage) if self.memory_usage else 0
                }
            }

            return summary

    def _monitor_system(self):
        """
        Background system monitoring
        """
        while self.is_monitoring:
            with self.lock:
                # CPU usage
                cpu_percent = psutil.cpu_percent()
                self.cpu_loads.append(cpu_percent)

                # Memory usage
                memory_percent = psutil.virtual_memory().percent
                self.memory_usage.append(memory_percent)

                # GPU usage
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    self.gpu_loads.append(gpu.load * 100)
                    self.gpu_memory.append(gpu.memoryUtil * 100)
                else:
                    self.gpu_loads.append(0)
                    self.gpu_memory.append(0)

            time.sleep(0.1)  # Monitor every 100ms

    def should_optimize(self):
        """
        Check if optimization is needed based on performance
        """
        summary = self.get_performance_summary()

        if not summary:
            return False

        # Check various thresholds
        avg_time = summary['inference_stats'].get('avg_time', float('inf'))
        avg_gpu_load = summary['system_stats'].get('avg_gpu_load', 0)
        avg_cpu_load = summary['system_stats'].get('avg_cpu_load', 0)

        # Optimization needed if any threshold is exceeded
        return (avg_time > 0.1 or  # More than 100ms per inference
                avg_gpu_load > 90 or  # GPU heavily loaded
                avg_cpu_load > 90)    # CPU heavily loaded

    def save_performance_log(self, filename):
        """
        Save performance metrics to file
        """
        summary = self.get_performance_summary()
        summary['timestamp'] = time.time()

        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)

class OptimizedInferenceWrapper:
    def __init__(self, model, performance_monitor):
        self.model = model
        self.monitor = performance_monitor
        self.device = next(model.parameters()).device

        # Optimization flags
        self.use_half_precision = False
        self.current_batch_size = 1

    def __call__(self, inputs):
        """
        Optimized inference with performance monitoring
        """
        start_time = time.time()

        # Apply optimizations based on performance
        if self.monitor.should_optimize():
            self._apply_optimizations()

        # Convert to appropriate precision
        if self.use_half_precision:
            inputs = inputs.half()

        # Perform inference
        with torch.no_grad():
            outputs = self.model(inputs)

        end_time = time.time()
        inference_time = end_time - start_time

        # Record performance
        self.monitor.record_inference(inference_time, inputs.size(0))

        return outputs

    def _apply_optimizations(self):
        """
        Apply optimizations based on performance monitoring
        """
        summary = self.monitor.get_performance_summary()

        if summary:
            avg_time = summary['inference_stats'].get('avg_time', 0)
            avg_gpu_load = summary['system_stats'].get('avg_gpu_load', 0)

            # If inference is too slow, try half precision
            if avg_time > 0.05 and not self.use_half_precision:
                self.use_half_precision = True
                print("Switching to half precision for better performance")

            # If GPU is overloaded, reduce batch size
            if avg_gpu_load > 95 and self.current_batch_size > 1:
                self.current_batch_size = max(1, self.current_batch_size - 1)
                print(f"Reducing batch size to {self.current_batch_size}")
```

## 6.8 Implementation Best Practices

### 6.8.1 Optimization Pipeline

```python
class OptimizationPipeline:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.performance_monitor = PerformanceMonitor()

    def optimize_for_robotic_platform(self, target_latency=0.03, max_power_draw=50):
        """
        Full optimization pipeline for robotic platform
        """
        print("Starting optimization pipeline...")

        # Step 1: Quantization
        print("Step 1: Applying quantization...")
        quantized_model = self._apply_quantization()

        # Step 2: Pruning
        print("Step 2: Applying pruning...")
        pruned_model = self._apply_pruning(quantized_model)

        # Step 3: Knowledge distillation (if needed)
        print("Step 3: Applying knowledge distillation...")
        distilled_model = self._apply_distillation(pruned_model)

        # Step 4: TensorRT optimization
        print("Step 4: Applying TensorRT optimization...")
        trt_engine = self._apply_tensorrt(distilled_model)

        # Step 5: Performance validation
        print("Step 5: Validating performance...")
        self._validate_performance(trt_engine, target_latency)

        print("Optimization pipeline completed!")
        return trt_engine

    def _apply_quantization(self):
        """
        Apply quantization to model
        """
        self.model.eval()
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

        quantized_model = torch.quantization.prepare(self.model, inplace=False)

        # Calibrate with sample data
        # ... calibration code ...

        quantized_model = torch.quantization.convert(quantized_model, inplace=False)
        return quantized_model

    def _apply_pruning(self, model, target_sparsity=0.3):
        """
        Apply pruning to model
        """
        pruner = StructuredPruner(model)

        # Prune model
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=target_sparsity)

        # Remove pruning masks
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and hasattr(module, 'weight_orig'):
                prune.remove(module, 'weight')

        return model

    def _apply_distillation(self, model):
        """
        Apply knowledge distillation if model is still too large
        """
        # Check if distillation is needed
        model_size = self._get_model_size(model)

        if model_size > 50:  # If model is larger than 50MB
            # Create smaller student model
            student_model = self._create_lightweight_model(model)

            # Train student model using teacher
            trainer = KnowledgeDistillationTrainer(model, student_model, self.device)
            # ... training code ...

            return student_model

        return model

    def _apply_tensorrt(self, model):
        """
        Apply TensorRT optimization
        """
        # Export to ONNX
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)

        onnx_path = "optimized_model.onnx"
        torch.onnx.export(
            model, dummy_input, onnx_path,
            input_names=['input'], output_names=['output'],
            opset_version=11
        )

        # Build TensorRT engine
        optimizer = TensorRTOptimizer()
        engine = optimizer.build_engine(onnx_path, precision='fp16')

        return engine

    def _validate_performance(self, engine, target_latency):
        """
        Validate that optimized model meets requirements
        """
        session = TrtInferenceSession(engine)

        # Test with sample inputs
        test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

        latencies = []
        for _ in range(100):  # Test 100 times
            start = time.time()
            _ = session.infer(test_input)
            latency = time.time() - start
            latencies.append(latency)

        avg_latency = np.mean(latencies)

        if avg_latency > target_latency:
            print(f"Warning: Average latency {avg_latency:.4f}s exceeds target {target_latency:.4f}s")
        else:
            print(f"Performance validation passed: {avg_latency:.4f}s average latency")

        return avg_latency <= target_latency

    def _get_model_size(self, model):
        """
        Get model size in MB
        """
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_mb = (param_size + buffer_size) / 1024**2
        return size_mb

    def _create_lightweight_model(self, original_model):
        """
        Create a lightweight version of the model
        """
        # This would create a smaller architecture
        # Implementation depends on the specific model type
        pass
```

## 6.9 Real-world Deployment Considerations

### 6.9.1 Thermal Management

```python
class ThermalManager:
    def __init__(self, max_temperature=80.0, throttle_threshold=70.0):
        self.max_temp = max_temperature
        self.throttle_temp = throttle_threshold
        self.current_temp = 0
        self.throttling_active = False

    def get_gpu_temperature(self):
        """
        Get current GPU temperature
        """
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', 'dmon', '-s', 't', '-d', '1', '-c', '1'],
                                  capture_output=True, text=True)
            # Parse temperature from nvidia-smi output
            # This is a simplified example
            return 65.0  # Placeholder
        except:
            return 45.0  # Default temperature

    def should_reduce_performance(self):
        """
        Check if performance should be reduced due to thermal constraints
        """
        self.current_temp = self.get_gpu_temperature()

        if self.current_temp > self.throttle_temp:
            self.throttling_active = True
            return True
        elif self.current_temp < self.throttle_temp - 5:
            self.throttling_active = False

        return self.throttling_active

    def get_throttle_factor(self):
        """
        Get factor by which to reduce performance
        """
        if not self.throttling_active:
            return 1.0

        # Calculate throttle factor based on temperature
        excess_temp = max(0, self.current_temp - self.throttle_temp)
        throttle_factor = max(0.1, 1.0 - (excess_temp / (self.max_temp - self.throttle_temp)))

        return throttle_factor
```

## Summary

Real-time inference optimization is essential for deploying sophisticated AI models on humanoid robots with limited computational resources. The key techniques include:

1. **Quantization**: Reducing precision from FP32 to INT8 or FP16 for significant speedup
2. **Pruning**: Removing redundant weights to reduce model size and computation
3. **Knowledge Distillation**: Creating smaller, faster student models that maintain performance
4. **Hardware Acceleration**: Leveraging TensorRT and other accelerators for optimal performance
5. **Dynamic Batching**: Adjusting batch sizes based on workload and performance requirements
6. **Performance Monitoring**: Continuously monitoring and adapting to maintain optimal performance

The optimization process should be systematic, starting with less invasive techniques like quantization and progressing to more aggressive methods like pruning and distillation only when necessary. Performance monitoring is crucial to ensure that optimizations don't compromise the robot's safety or effectiveness.

In the next chapter, we will explore how to integrate all these optimization techniques into a complete end-to-end Vision-Language-Action system for humanoid robots.