---
title: Chapter 1 - OpenAI Whisper Integration
description: Learn how to integrate OpenAI Whisper for speech recognition in humanoid robots, including real-time processing and ROS 2 integration.
sidebar_position: 35
---

# Chapter 1 - OpenAI Whisper Integration

OpenAI Whisper is a state-of-the-art speech recognition model that provides robust automatic speech recognition (ASR) capabilities. For humanoid robots, integrating Whisper enables natural voice interaction, allowing robots to understand spoken commands and engage in conversational interfaces. This chapter covers the integration of Whisper into humanoid robot systems, focusing on real-time processing, ROS 2 integration, and optimization for robotic applications.

## 1.1 Introduction to OpenAI Whisper

Whisper is a general-purpose speech recognition model trained on 680,000 hours of multilingual and multitask supervised data. It demonstrates strong performance across various domains and languages, making it ideal for humanoid robot applications where robust speech recognition is crucial.

### 1.1.1 Key Features of Whisper

- **Multilingual Support**: Supports multiple languages out of the box
- **Robustness**: Handles various accents, background noise, and audio quality
- **Versatility**: Can perform speech recognition, translation, and language identification
- **Open Source**: Available with different model sizes for various computational requirements

### 1.1.2 Whisper Model Sizes

Whisper comes in five model sizes, each with different performance and computational requirements:

| Model | Parameters | English-only | Multilingual | Required VRAM |
|-------|------------|--------------|--------------|---------------|
| tiny  | 39 M       | 1 GB         | 1 GB         | 1 GB          |
| base  | 74 M       | 1 GB         | 1 GB         | 1 GB          |
| small | 244 M      | 2 GB         | 2 GB         | 2 GB          |
| medium| 769 M      | 5 GB         | 5 GB         | 5 GB          |
| large | 1550 M     | 10 GB        | 10 GB        | 10 GB         |

For humanoid robots, the `base` or `small` models typically provide a good balance between accuracy and computational efficiency.

## 1.2 Setting up Whisper for Robotics

### 1.2.1 Installation and Dependencies

First, install the required dependencies for Whisper integration:

```bash
# Install OpenAI Whisper
pip install openai-whisper

# Install additional dependencies for audio processing
pip install pyaudio soundfile librosa

# For ROS 2 integration
pip install rclpy std_msgs sensor_msgs
```

### 1.2.2 Basic Whisper Usage

```python
import whisper
import numpy as np
import soundfile as sf

# Load the model (choose appropriate size for your hardware)
model = whisper.load_model("base")  # or "small" for better accuracy

# Load audio file
audio, sr = sf.read("audio_file.wav")

# Convert to 16kHz (required for Whisper)
if sr != 16000:
    import librosa
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

# Transcribe audio
result = model.transcribe(audio)
print(result["text"])
```

## 1.3 Real-time Audio Processing for Robotics

### 1.3.1 Audio Stream Processing

For humanoid robots, real-time audio processing is crucial. Here's how to implement continuous audio capture and processing:

```python
import pyaudio
import numpy as np
import threading
import queue
import time
from dataclasses import dataclass
from typing import Optional, Callable

@dataclass
class AudioChunk:
    data: np.ndarray
    timestamp: float
    sample_rate: int = 16000

class AudioStreamProcessor:
    def __init__(self,
                 chunk_size: int = 1024,
                 sample_rate: int = 16000,
                 channels: int = 1,
                 callback: Optional[Callable[[AudioChunk], None]] = None):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.channels = channels
        self.callback = callback
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.stream = None
        self.audio = pyaudio.PyAudio()

    def start_recording(self):
        """Start audio recording in a separate thread"""
        self.is_recording = True

        # Open audio stream
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )

        # Start recording thread
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.recording_thread.start()

    def stop_recording(self):
        """Stop audio recording"""
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

    def _record_audio(self):
        """Internal method to record audio chunks"""
        while self.is_recording:
            try:
                # Read audio data
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                audio_array = np.frombuffer(data, dtype=np.float32)

                # Create audio chunk
                chunk = AudioChunk(
                    data=audio_array,
                    timestamp=time.time()
                )

                # Add to queue for processing
                self.audio_queue.put(chunk)

                # Call callback if provided
                if self.callback:
                    self.callback(chunk)

            except Exception as e:
                print(f"Error recording audio: {e}")
                break

    def get_audio_chunk(self) -> Optional[AudioChunk]:
        """Get the next available audio chunk"""
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None

    def get_audio_chunks(self) -> list[AudioChunk]:
        """Get all available audio chunks"""
        chunks = []
        try:
            while True:
                chunk = self.audio_queue.get_nowait()
                chunks.append(chunk)
        except queue.Empty:
            pass
        return chunks
```

### 1.3.2 Voice Activity Detection

To improve efficiency and reduce unnecessary processing, implement voice activity detection (VAD):

```python
import webrtcvad
import collections

class VoiceActivityDetector:
    def __init__(self, sample_rate: int = 16000, frame_duration: int = 30):
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(2)  # Aggressive VAD mode

        self.sample_rate = sample_rate
        self.frame_duration = frame_duration  # in ms
        self.frame_size = int(sample_rate * frame_duration / 1000)

        # Buffer for audio chunks
        self.audio_buffer = collections.deque(maxlen=100)  # 3 seconds of audio
        self.is_speaking = False
        self.speech_start_time = None
        self.speech_end_time = None

    def is_voice_active(self, audio_chunk: AudioChunk) -> bool:
        """Detect if voice is active in the audio chunk"""
        # Convert to bytes for VAD
        audio_bytes = (audio_chunk.data * 32767).astype(np.int16).tobytes()

        # Split into frames
        frames = [audio_bytes[i:i+2*self.frame_size] for i in range(0, len(audio_bytes), 2*self.frame_size)]

        voice_frames = 0
        total_frames = len(frames)

        for frame in frames:
            if len(frame) == 2 * self.frame_size:  # Ensure frame is complete
                if self.vad.is_speech(frame, self.sample_rate):
                    voice_frames += 1

        # Consider voice active if more than 30% of frames have voice
        voice_ratio = voice_frames / max(total_frames, 1)
        return voice_ratio > 0.3

    def process_audio_chunk(self, audio_chunk: AudioChunk) -> tuple[bool, Optional[np.ndarray]]:
        """
        Process audio chunk and return (is_speech_activated, accumulated_audio)
        """
        is_active = self.is_voice_active(audio_chunk)

        # Add to buffer
        self.audio_buffer.append(audio_chunk.data)

        if is_active and not self.is_speaking:
            # Speech started
            self.is_speaking = True
            self.speech_start_time = audio_chunk.timestamp
            return True, None

        elif not is_active and self.is_speaking:
            # Speech ended - return accumulated audio
            self.is_speaking = False
            self.speech_end_time = audio_chunk.timestamp

            # Return accumulated audio for processing
            accumulated_audio = np.concatenate(list(self.audio_buffer))
            self.audio_buffer.clear()
            return True, accumulated_audio

        elif self.is_speaking:
            # Continue accumulating speech
            return False, None
        else:
            # No speech, return None
            return False, None
```

## 1.4 Whisper Integration with ROS 2

### 1.4.1 ROS 2 Node for Whisper Processing

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import AudioData
from audio_common_msgs.msg import AudioData as AudioDataMsg
import whisper
import numpy as np
import threading
import queue
import time
from typing import Optional

class WhisperROSNode(Node):
    def __init__(self):
        super().__init__('whisper_ros_node')

        # Initialize Whisper model
        self.model_size = self.declare_parameter('model_size', 'base').value
        self.model = whisper.load_model(self.model_size)

        # Parameters
        self.silence_threshold = self.declare_parameter('silence_threshold', 0.5).value
        self.min_speech_duration = self.declare_parameter('min_speech_duration', 1.0).value
        self.max_speech_duration = self.declare_parameter('max_speech_duration', 10.0).value

        # Publishers
        self.transcription_pub = self.create_publisher(String, 'whisper/transcription', 10)
        self.speech_detected_pub = self.create_publisher(Bool, 'whisper/speech_detected', 10)

        # Subscribers
        self.audio_sub = self.create_subscription(
            AudioDataMsg,
            'audio/input',
            self.audio_callback,
            10
        )

        # Internal state
        self.audio_buffer = []
        self.recording_start_time = None
        self.is_recording = False
        self.transcription_queue = queue.Queue()

        # Start transcription processing thread
        self.transcription_thread = threading.Thread(target=self.process_transcriptions, daemon=True)
        self.transcription_thread.start()

        self.get_logger().info(f'Whisper ROS Node initialized with {self.model_size} model')

    def audio_callback(self, msg: AudioDataMsg):
        """Process incoming audio data"""
        try:
            # Convert audio data to numpy array
            audio_data = np.frombuffer(msg.data, dtype=np.int16).astype(np.float32) / 32768.0

            # Perform voice activity detection and speech accumulation
            should_process, speech_audio = self.process_speech_detection(audio_data)

            if should_process and speech_audio is not None:
                # Queue for transcription
                self.transcription_queue.put(speech_audio)

        except Exception as e:
            self.get_logger().error(f'Error processing audio: {e}')

    def process_speech_detection(self, audio_chunk: np.ndarray) -> tuple[bool, Optional[np.ndarray]]:
        """Process speech detection and accumulation"""
        # Calculate energy of the chunk
        energy = np.mean(audio_chunk ** 2)

        if energy > self.silence_threshold:
            # Voice detected
            if not self.is_recording:
                # Start new recording
                self.is_recording = True
                self.recording_start_time = time.time()
                self.audio_buffer = [audio_chunk]
                self.speech_detected_pub.publish(Bool(data=True))
            else:
                # Continue recording
                self.audio_buffer.append(audio_chunk)

                # Check if max duration exceeded
                current_duration = time.time() - self.recording_start_time
                if current_duration > self.max_speech_duration:
                    # Max duration reached, return accumulated audio
                    accumulated = np.concatenate(self.audio_buffer)
                    self.is_recording = False
                    self.audio_buffer = []
                    return True, accumulated
        else:
            # Silence detected
            if self.is_recording:
                # Check if minimum speech duration was met
                current_duration = time.time() - self.recording_start_time
                if current_duration >= self.min_speech_duration:
                    # Return accumulated audio for processing
                    accumulated = np.concatenate(self.audio_buffer)
                    self.is_recording = False
                    self.audio_buffer = []
                    return True, accumulated
                else:
                    # Below minimum duration, reset
                    self.is_recording = False
                    self.audio_buffer = []

        return False, None

    def process_transcriptions(self):
        """Process transcriptions in a separate thread"""
        while rclpy.ok():
            try:
                # Get audio for transcription
                audio_data = self.transcription_queue.get(timeout=1.0)

                # Perform transcription
                result = self.model.transcribe(audio_data)
                transcription = result["text"].strip()

                if transcription:  # Only publish non-empty transcriptions
                    # Publish transcription
                    transcription_msg = String()
                    transcription_msg.data = transcription
                    self.transcription_pub.publish(transcription_msg)

                    self.get_logger().info(f'Transcribed: {transcription}')

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'Error in transcription processing: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = WhisperROSNode()

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

### 1.4.2 Launch File for Whisper Node

```xml
<!-- whisper_integration.launch.py -->
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Launch arguments
    model_size_arg = DeclareLaunchArgument(
        'model_size',
        default_value='base',
        description='Whisper model size (tiny, base, small, medium, large)'
    )

    silence_threshold_arg = DeclareLaunchArgument(
        'silence_threshold',
        default_value='0.01',
        description='Threshold for voice activity detection'
    )

    min_speech_duration_arg = DeclareLaunchArgument(
        'min_speech_duration',
        default_value='1.0',
        description='Minimum speech duration in seconds'
    )

    max_speech_duration_arg = DeclareLaunchArgument(
        'max_speech_duration',
        default_value='10.0',
        description='Maximum speech duration in seconds'
    )

    # Whisper node
    whisper_node = Node(
        package='whisper_integration',
        executable='whisper_ros_node',
        name='whisper_node',
        parameters=[
            {'model_size': LaunchConfiguration('model_size')},
            {'silence_threshold': LaunchConfiguration('silence_threshold')},
            {'min_speech_duration': LaunchConfiguration('min_speech_duration')},
            {'max_speech_duration': LaunchConfiguration('max_speech_duration')}
        ],
        remappings=[
            ('audio/input', '/microphone/audio_raw'),
            ('whisper/transcription', '/speech_recognition/text'),
            ('whisper/speech_detected', '/speech_recognition/speech_detected')
        ]
    )

    return LaunchDescription([
        model_size_arg,
        silence_threshold_arg,
        min_speech_duration_arg,
        max_speech_duration_arg,
        whisper_node
    ])
```

## 1.5 Optimization for Real-time Performance

### 1.5.1 Model Optimization Techniques

For humanoid robots with limited computational resources, several optimization techniques can be applied:

```python
import torch
import whisper
from whisper import load_model
from whisper.audio import log_mel_spectrogram, pad_or_trim
from whisper.decoding import DecodingOptions

class OptimizedWhisperProcessor:
    def __init__(self, model_size: str = "base", device: str = "cuda"):
        self.device = device

        # Load model and move to device
        self.model = load_model(model_size).to(device)

        # Set to evaluation mode
        self.model.eval()

        # Use torch.jit for optimization (if available)
        self.use_jit = torch.jit.is_scripting() or torch.jit.is_tracing()

        # Warm up the model
        self._warm_up()

    def _warm_up(self):
        """Warm up the model with dummy input"""
        dummy_audio = torch.zeros(16000 * 5)  # 5 seconds of silence
        try:
            self.transcribe(dummy_audio)
        except:
            pass  # Ignore errors during warmup

    def transcribe(self, audio: np.ndarray, language: str = "en",
                   task: str = "transcribe") -> str:
        """Optimized transcription function"""
        # Convert to tensor and move to device
        if not isinstance(audio, torch.Tensor):
            audio = torch.from_numpy(audio)

        audio = audio.to(self.device)

        # Compute log-Mel spectrogram
        mel = log_mel_spectrogram(audio)

        # Pad or trim to required length
        mel = pad_or_trim(mel, N=whisper.audio.N_FRAMES)

        # Perform transcription
        with torch.no_grad():
            options = DecodingOptions(
                language=language,
                task=task,
                fp16=(self.device == "cuda")  # Use fp16 on GPU for speed
            )

            result = self.model.decode(mel, options)

        return result.text

    def batch_transcribe(self, audio_batch: list[np.ndarray]) -> list[str]:
        """Batch transcription for multiple audio segments"""
        results = []

        for audio in audio_batch:
            result = self.transcribe(audio)
            results.append(result)

        return results

class WhisperPipeline:
    def __init__(self, model_size: str = "base", device: str = "cuda"):
        self.processor = OptimizedWhisperProcessor(model_size, device)
        self.audio_buffer = []
        self.transcription_history = []
        self.max_history = 100  # Keep last 100 transcriptions

    def add_audio(self, audio_chunk: np.ndarray) -> Optional[str]:
        """Add audio chunk and return transcription if speech detected"""
        # Add to buffer for processing
        self.audio_buffer.append(audio_chunk)

        # Check if we have enough audio for processing
        total_samples = sum(len(chunk) for chunk in self.audio_buffer)

        if total_samples >= 16000 * 2:  # At least 2 seconds of audio
            # Combine audio chunks
            combined_audio = np.concatenate(self.audio_buffer)

            # Clear buffer
            self.audio_buffer = []

            # Perform transcription
            transcription = self.processor.transcribe(combined_audio)

            # Add to history
            self.transcription_history.append({
                'text': transcription,
                'timestamp': time.time()
            })

            # Maintain history size
            if len(self.transcription_history) > self.max_history:
                self.transcription_history = self.transcription_history[-self.max_history:]

            return transcription

        return None
```

### 1.5.2 Memory and Performance Monitoring

```python
import psutil
import GPUtil
import time
from dataclasses import dataclass

@dataclass
class PerformanceMetrics:
    cpu_percent: float
    memory_percent: float
    gpu_percent: float
    gpu_memory: float
    transcription_time: float
    buffer_size: int

class WhisperPerformanceMonitor:
    def __init__(self):
        self.metrics_history = []
        self.max_history = 100

    def record_metrics(self, transcription_time: float, buffer_size: int) -> PerformanceMetrics:
        """Record performance metrics"""
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent

        gpus = GPUtil.getGPUs()
        gpu_percent = gpus[0].load if gpus else 0
        gpu_memory = gpus[0].memoryUtil if gpus else 0

        metrics = PerformanceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            gpu_percent=gpu_percent,
            gpu_memory=gpu_memory,
            transcription_time=transcription_time,
            buffer_size=buffer_size
        )

        # Add to history
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history:
            self.metrics_history = self.metrics_history[-self.max_history:]

        return metrics

    def get_performance_summary(self) -> dict:
        """Get performance summary statistics"""
        if not self.metrics_history:
            return {}

        # Calculate averages
        avg_cpu = sum(m.cpu_percent for m in self.metrics_history) / len(self.metrics_history)
        avg_memory = sum(m.memory_percent for m in self.metrics_history) / len(self.metrics_history)
        avg_gpu = sum(m.gpu_percent for m in self.metrics_history) / len(self.metrics_history)
        avg_transcription_time = sum(m.transcription_time for m in self.metrics_history) / len(self.metrics_history)

        return {
            'avg_cpu_percent': avg_cpu,
            'avg_memory_percent': avg_memory,
            'avg_gpu_percent': avg_gpu,
            'avg_transcription_time': avg_transcription_time,
            'total_samples': len(self.metrics_history)
        }

    def should_optimize(self) -> bool:
        """Check if system optimization is needed"""
        summary = self.get_performance_summary()

        # Optimization thresholds
        cpu_threshold = 80.0
        memory_threshold = 85.0
        gpu_threshold = 85.0
        time_threshold = 2.0  # seconds

        return (summary.get('avg_cpu_percent', 0) > cpu_threshold or
                summary.get('avg_memory_percent', 0) > memory_threshold or
                summary.get('avg_gpu_percent', 0) > gpu_threshold or
                summary.get('avg_transcription_time', 0) > time_threshold)
```

## 1.6 Advanced Whisper Features for Robotics

### 1.6.1 Language Detection and Multilingual Support

```python
class MultilingualWhisperProcessor:
    def __init__(self, model_size: str = "small"):  # Use 'small' or larger for multilingual
        self.model = whisper.load_model(model_size)
        self.supported_languages = [
            'en', 'zh', 'de', 'es', 'ru', 'ko', 'fr', 'ja', 'pt', 'tr',
            'pl', 'ca', 'nl', 'ar', 'sv', 'it', 'id', 'hi', 'fi', 'vi',
            'he', 'uk', 'el', 'ms', 'cs', 'ro', 'da', 'hu', 'ta', 'no',
            'th', 'ur', 'hr', 'bg', 'lt', 'la', 'mi', 'ml', 'cy', 'sk',
            'te', 'fa', 'lv', 'bn', 'sr', 'az', 'sl', 'kn', 'et', 'mk',
            'br', 'eu', 'is', 'hy', 'ne', 'mn', 'bs', 'kk', 'sq', 'sw',
            'gl', 'mr', 'pa', 'si', 'km', 'sn', 'yo', 'so', 'af', 'oc',
            'ka', 'be', 'tg', 'sd', 'gu', 'am', 'yi', 'lo', 'uz', 'fo',
            'ht', 'ps', 'tk', 'nn', 'mt', 'sa', 'lb', 'my', 'bo', 'tl',
            'mg', 'as', 'tt', 'haw', 'ln', 'ha', 'ba', 'jw', 'su'
        ]

    def detect_language(self, audio: np.ndarray) -> str:
        """Detect the language of the audio"""
        # Transcribe with language detection
        result = self.model.transcribe(audio, task="detect_language")
        detected_language = result["language"]

        return detected_language

    def transcribe_multilingual(self, audio: np.ndarray, target_language: str = "en") -> str:
        """Transcribe audio with translation if needed"""
        # Detect source language
        source_language = self.detect_language(audio)

        if source_language == target_language:
            # Same language, just transcribe
            result = self.model.transcribe(audio, language=source_language)
        else:
            # Different language, translate
            result = self.model.transcribe(
                audio,
                language=source_language,
                task="translate" if target_language == "en" else "transcribe"
            )

        return result["text"]
```

### 1.6.2 Context-Aware Transcription

```python
class ContextAwareWhisper:
    def __init__(self, model_size: str = "base"):
        self.model = whisper.load_model(model_size)
        self.context_keywords = set()
        self.robot_commands = [
            "move forward", "turn left", "turn right", "stop", "help",
            "come here", "go there", "pick up", "put down", "follow me",
            "wait", "start", "stop", "hello", "goodbye", "yes", "no"
        ]

    def add_context_keywords(self, keywords: list[str]):
        """Add context-specific keywords for better recognition"""
        self.context_keywords.update(keywords)

    def transcribe_with_context(self, audio: np.ndarray, context: str = "") -> str:
        """Transcribe audio with context information"""
        # For now, we'll use the standard transcription
        # In practice, you might use the initial_prompt parameter
        # or fine-tune the model with context-specific data

        if self.context_keywords:
            # Create a prompt that includes context keywords
            initial_prompt = " ".join(list(self.context_keywords)[:50])  # Limit prompt length
            result = self.model.transcribe(audio, initial_prompt=initial_prompt)
        else:
            result = self.model.transcribe(audio)

        return result["text"]

    def validate_robot_command(self, transcription: str) -> tuple[bool, str]:
        """Validate if transcription contains a valid robot command"""
        transcription_lower = transcription.lower()

        for command in self.robot_commands:
            if command in transcription_lower:
                return True, command

        return False, transcription
```

## 1.7 Error Handling and Robustness

### 1.7.1 Robust Whisper Integration

```python
import asyncio
import aiohttp
from typing import Optional, Union

class RobustWhisperClient:
    def __init__(self,
                 model_size: str = "base",
                 max_retries: int = 3,
                 timeout: float = 30.0):
        self.model_size = model_size
        self.max_retries = max_retries
        self.timeout = timeout
        self.model = None
        self.load_model_with_fallback()

    def load_model_with_fallback(self):
        """Load model with fallback to smaller models if needed"""
        model_sizes = [self.model_size]
        if self.model_size in ["large", "medium", "small"]:
            model_sizes.extend(["base", "tiny"])  # Fallback to smaller models

        for size in model_sizes:
            try:
                self.model = whisper.load_model(size)
                print(f"Successfully loaded Whisper model: {size}")
                break
            except Exception as e:
                print(f"Failed to load Whisper model {size}: {e}")
                if size == "tiny":  # Last fallback failed
                    raise RuntimeError("Could not load any Whisper model")

    async def transcribe_async(self, audio: Union[np.ndarray, str]) -> Optional[str]:
        """Async transcription with retry logic"""
        for attempt in range(self.max_retries):
            try:
                if isinstance(audio, str):  # File path
                    result = self.model.transcribe(audio)
                else:  # Audio array
                    result = self.model.transcribe(audio)

                return result["text"]

            except Exception as e:
                print(f"Transcription attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:  # Last attempt
                    return None
                await asyncio.sleep(1.0)  # Wait before retry

        return None

    def transcribe_with_timeout(self, audio: np.ndarray, timeout_seconds: int = 10) -> Optional[str]:
        """Transcribe with timeout protection"""
        def transcribe_func():
            return self.model.transcribe(audio)["text"]

        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(transcribe_func)
            try:
                result = future.result(timeout=timeout_seconds)
                return result
            except concurrent.futures.TimeoutError:
                print("Transcription timed out")
                return None
```

## 1.8 Best Practices for Robotics Integration

### 1.8.1 Performance Considerations

1. **Model Selection**: Choose the appropriate model size based on your hardware capabilities
2. **Batch Processing**: Process multiple audio segments when possible
3. **Caching**: Cache frequently used transcriptions or model states
4. **Resource Management**: Monitor and manage GPU/CPU usage
5. **Error Handling**: Implement robust error handling and fallback mechanisms

### 1.8.2 Integration Guidelines

1. **Real-time Constraints**: Ensure transcription latency meets robotic system requirements
2. **Audio Quality**: Use high-quality microphones and pre-processing when possible
3. **Context Awareness**: Incorporate contextual information for better accuracy
4. **Privacy**: Consider privacy implications when processing speech data
5. **Testing**: Test with diverse audio conditions and speakers

## Summary

OpenAI Whisper provides powerful speech recognition capabilities that are essential for natural human-robot interaction in humanoid robots. By properly integrating Whisper with ROS 2, optimizing for real-time performance, and implementing robust error handling, you can create sophisticated voice interfaces for your humanoid robot. The key to successful integration lies in balancing accuracy, performance, and resource constraints while maintaining a responsive and reliable user experience. In the next chapter, we will explore LLM prompt engineering techniques specifically tailored for robotics applications.