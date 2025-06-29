"""
Training Configuration Management for Food-101 Model Training
Handles different training presets, model configurations, and hyperparameter management
"""

import json
try:
    import yaml
except ImportError:
    yaml = None
    print("Warning: PyYAML not installed. YAML config files will not be supported.")
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Union
import tensorflow as tf
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for model architecture"""
    model_type: str = "EfficientNetB0"
    input_size: tuple = (224, 224)
    num_classes: int = 101
    dropout_rate: float = 0.3
    dense_units: Optional[List[int]] = None
    use_batch_norm: bool = True
    activation: str = "relu"
    
    def __post_init__(self):
        if self.dense_units is None:
            self.dense_units = [512, 256]

@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    lr_reduction_patience: int = 5
    lr_reduction_factor: float = 0.3
    min_learning_rate: float = 1e-7
    optimizer: str = "adam"
    loss_function: str = "categorical_crossentropy"
    metrics: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["accuracy", "top_5_accuracy"]

@dataclass
class AugmentationConfig:
    """Configuration for data augmentation"""
    rotation_range: float = 30.0
    width_shift_range: float = 0.3
    height_shift_range: float = 0.3
    horizontal_flip: bool = True
    zoom_range: float = 0.3
    shear_range: float = 0.2
    brightness_range: tuple = (0.7, 1.3)
    channel_shift_range: float = 0.2
    fill_mode: str = "nearest"
    rescale: float = 1.0/255.0
    
@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning"""
    enabled: bool = True
    unfreeze_at_layer: int = 100
    fine_tune_learning_rate: float = 0.0001
    fine_tune_epochs: int = 10
    gradual_unfreezing: bool = False
    unfreeze_steps: Optional[List[int]] = None
    
    def __post_init__(self):
        if self.unfreeze_steps is None:
            self.unfreeze_steps = [50, 75, 100]

@dataclass
class CallbackConfig:
    """Configuration for training callbacks"""
    early_stopping: bool = True
    model_checkpoint: bool = True
    reduce_lr_on_plateau: bool = True
    tensorboard: bool = True
    csv_logger: bool = True
    custom_callbacks: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.custom_callbacks is None:
            self.custom_callbacks = []

class Food101TrainingConfig:
    """
    Comprehensive training configuration manager for Food-101 models
    """
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.model_config = ModelConfig()
        self.training_config = TrainingConfig()
        self.augmentation_config = AugmentationConfig()
        self.fine_tuning_config = FineTuningConfig()
        self.callback_config = CallbackConfig()
        
        # Predefined training presets
        self.presets = self._create_presets()
        
        # Model architecture specifications
        self.model_specs = self._create_model_specs()
        
        # Load config if file provided
        if config_file and Path(config_file).exists():
            self.load_config(config_file)
    
    def _create_presets(self) -> Dict[str, Dict[str, Any]]:
        """Create predefined training presets"""
        return {
            "quick": {
                "name": "⚡ Quick Training",
                "description": "Fast training for testing and prototyping (15-20 min)",
                "estimated_time": "15-20 minutes",
                "target_accuracy": "75-80%",
                "model": {
                    "model_type": "MobileNetV2",
                    "dropout_rate": 0.2,
                    "dense_units": [256, 128]
                },
                "training": {
                    "epochs": 15,
                    "batch_size": 32,
                    "learning_rate": 0.001,
                    "early_stopping_patience": 5
                },
                "augmentation": {
                    "rotation_range": 20.0,
                    "zoom_range": 0.2,
                    "brightness_range": (0.8, 1.2)
                },
                "fine_tuning": {
                    "enabled": True,
                    "fine_tune_epochs": 5,
                    "unfreeze_at_layer": 50
                }
            },
            "balanced": {
                "name": "⚖️ Balanced Training",
                "description": "Good balance between speed and accuracy (45-60 min)",
                "estimated_time": "45-60 minutes",
                "target_accuracy": "80-85%",
                "model": {
                    "model_type": "EfficientNetB0",
                    "dropout_rate": 0.3,
                    "dense_units": [512, 256]
                },
                "training": {
                    "epochs": 30,
                    "batch_size": 16,
                    "learning_rate": 0.0001,
                    "early_stopping_patience": 10
                },
                "augmentation": {
                    "rotation_range": 30.0,
                    "zoom_range": 0.3,
                    "brightness_range": (0.7, 1.3)
                },
                "fine_tuning": {
                    "enabled": True,
                    "fine_tune_epochs": 10,
                    "unfreeze_at_layer": 100
                }
            },
            "high_accuracy": {
                "name": "🎯 High Accuracy Training",
                "description": "Maximum accuracy with longer training time (2-3 hours)",
                "estimated_time": "2-3 hours",
                "target_accuracy": "85-90%",
                "model": {
                    "model_type": "EfficientNetB1",
                    "dropout_rate": 0.4,
                    "dense_units": [1024, 512, 256]
                },
                "training": {
                    "epochs": 50,
                    "batch_size": 8,
                    "learning_rate": 0.00001,
                    "early_stopping_patience": 15
                },
                "augmentation": {
                    "rotation_range": 35.0,
                    "zoom_range": 0.4,
                    "brightness_range": (0.6, 1.4),
                    "channel_shift_range": 0.3
                },
                "fine_tuning": {
                    "enabled": True,
                    "unfreeze_at_layer": 150,
                    "fine_tune_epochs": 20,
                    "gradual_unfreezing": True
                }
            },
            "lightweight": {
                "name": "🪶 Lightweight Model",
                "description": "Optimized for deployment and inference speed (30-40 min)",
                "estimated_time": "30-40 minutes",
                "target_accuracy": "75-80%",
                "model": {
                    "model_type": "MobileNetV2",
                    "dropout_rate": 0.2,
                    "dense_units": [128, 64]
                },
                "training": {
                    "epochs": 25,
                    "batch_size": 64,
                    "learning_rate": 0.001,
                    "early_stopping_patience": 8
                },
                "augmentation": {
                    "rotation_range": 25.0,
                    "zoom_range": 0.25,
                    "brightness_range": (0.8, 1.2)
                },
                "fine_tuning": {
                    "enabled": False
                }
            },
            "research": {
                "name": "🔬 Research Configuration",
                "description": "Advanced configuration for research purposes (4-6 hours)",
                "estimated_time": "4-6 hours",
                "target_accuracy": "90%+",
                "model": {
                    "model_type": "ResNet50",
                    "dropout_rate": 0.5,
                    "dense_units": [2048, 1024, 512, 256]
                },
                "training": {
                    "epochs": 100,
                    "batch_size": 16,
                    "learning_rate": 0.0001,
                    "early_stopping_patience": 20
                },
                "augmentation": {
                    "rotation_range": 40.0,
                    "zoom_range": 0.5,
                    "brightness_range": (0.5, 1.5),
                    "channel_shift_range": 0.4
                },
                "fine_tuning": {
                    "enabled": True,
                    "unfreeze_at_layer": 120,
                    "fine_tune_epochs": 30,
                    "gradual_unfreezing": True
                }
            },
            "production": {
                "name": "🚀 Production Ready",
                "description": "Optimized for production deployment (1-2 hours)",
                "estimated_time": "1-2 hours",
                "target_accuracy": "83-87%",
                "model": {
                    "model_type": "EfficientNetB0",
                    "dropout_rate": 0.35,
                    "dense_units": [512, 256, 128]
                },
                "training": {
                    "epochs": 40,
                    "batch_size": 24,
                    "learning_rate": 0.0001,
                    "early_stopping_patience": 12
                },
                "augmentation": {
                    "rotation_range": 30.0,
                    "zoom_range": 0.3,
                    "brightness_range": (0.7, 1.3)
                },
                "fine_tuning": {
                    "enabled": True,
                    "unfreeze_at_layer": 100,
                    "fine_tune_epochs": 15
                }
            }
        }
    
    def _create_model_specs(self) -> Dict[str, Dict[str, Any]]:
        """Create model architecture specifications"""
        return {
            "EfficientNetB0": {
                "params": "5.3M",
                "size": "21 MB",
                "input_size": (224, 224),
                "best_batch_size": [16, 32],
                "best_lr": [0.0001, 0.001],
                "description": "Balanced efficiency and accuracy",
                "pros": ["Fast training", "Good accuracy", "Small size"],
                "cons": ["May need more epochs for complex data"]
            },
            "EfficientNetB1": {
                "params": "7.8M",
                "size": "31 MB",
                "input_size": (240, 240),
                "best_batch_size": [8, 16],
                "best_lr": [0.00005, 0.0001],
                "description": "Higher accuracy with moderate size increase",
                "pros": ["Better accuracy", "Good generalization"],
                "cons": ["Slower training", "Larger memory usage"]
            },
            "EfficientNetB2": {
                "params": "9.2M",
                "size": "36 MB",
                "input_size": (260, 260),
                "best_batch_size": [8, 16],
                "best_lr": [0.00005, 0.0001],
                "description": "Further improved accuracy",
                "pros": ["High accuracy", "Good for complex datasets"],
                "cons": ["Slower training", "More memory intensive"]
            },
            "MobileNetV2": {
                "params": "3.5M",
                "size": "14 MB",
                "input_size": (224, 224),
                "best_batch_size": [32, 64],
                "best_lr": [0.001, 0.01],
                "description": "Lightweight and fast",
                "pros": ["Very fast", "Small size", "Mobile-friendly"],
                "cons": ["Lower accuracy", "Less detailed features"]
            },
            "ResNet50": {
                "params": "25.6M",
                "size": "98 MB",
                "input_size": (224, 224),
                "best_batch_size": [8, 16],
                "best_lr": [0.0001, 0.001],
                "description": "Deep residual network with proven performance",
                "pros": ["Proven architecture", "Good for complex tasks"],
                "cons": ["Large size", "Slower training", "More parameters"]
            },
            "InceptionV3": {
                "params": "23.9M",
                "size": "92 MB",
                "input_size": (299, 299),
                "best_batch_size": [8, 16],
                "best_lr": [0.0001, 0.001],
                "description": "Inception modules for multi-scale feature extraction",
                "pros": ["Multi-scale features", "Good accuracy"],
                "cons": ["Large input size", "Complex architecture"]
            },
            "DenseNet121": {
                "params": "8.1M",
                "size": "33 MB",
                "input_size": (224, 224),
                "best_batch_size": [16, 32],
                "best_lr": [0.0001, 0.001],
                "description": "Dense connections for feature reuse",
                "pros": ["Feature reuse", "Good gradient flow"],
                "cons": ["Memory intensive during training"]
            }
        }
    
    def apply_preset(self, preset_name: str) -> bool:
        """Apply a predefined training preset"""
        if preset_name not in self.presets:
            logger.error(f"Preset '{preset_name}' not found")
            return False
        
        preset = self.presets[preset_name]
        
        # Apply model config
        if "model" in preset:
            for key, value in preset["model"].items():
                if hasattr(self.model_config, key):
                    setattr(self.model_config, key, value)
        
        # Apply training config
        if "training" in preset:
            for key, value in preset["training"].items():
                if hasattr(self.training_config, key):
                    setattr(self.training_config, key, value)
        
        # Apply augmentation config
        if "augmentation" in preset:
            for key, value in preset["augmentation"].items():
                if hasattr(self.augmentation_config, key):
                    setattr(self.augmentation_config, key, value)
        
        # Apply fine-tuning config
        if "fine_tuning" in preset:
            for key, value in preset["fine_tuning"].items():
                if hasattr(self.fine_tuning_config, key):
                    setattr(self.fine_tuning_config, key, value)
        
        logger.info(f"Applied preset: {preset_name}")
        return True
    
    def get_callbacks(self, checkpoint_path: str = "models/checkpoint.h5") -> List[tf.keras.callbacks.Callback]:
        """Get training callbacks based on configuration"""
        callbacks = []
        
        if self.callback_config.early_stopping:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=self.training_config.early_stopping_patience,
                    restore_best_weights=True,
                    verbose=1,
                    mode='max'
                )
            )
        
        if self.callback_config.model_checkpoint:
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    checkpoint_path,
                    monitor='val_accuracy',
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=1,
                    mode='max'
                )
            )
        
        if self.callback_config.reduce_lr_on_plateau:
            # Create the callback without min_lr to avoid type issues
            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=self.training_config.lr_reduction_factor,
                    patience=self.training_config.lr_reduction_patience,
                    verbose=1,
                    mode='min'
                )
            )
        
        if self.callback_config.tensorboard:
            from datetime import datetime
            log_dir = f"logs/tensorboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            callbacks.append(
                tf.keras.callbacks.TensorBoard(
                    log_dir=log_dir,
                    histogram_freq=1,
                    write_graph=True,
                    write_images=True,
                    update_freq='epoch'
                )
            )
        
        if self.callback_config.csv_logger:
            from datetime import datetime
            log_file = f"logs/training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            callbacks.append(
                tf.keras.callbacks.CSVLogger(
                    log_file,
                    separator=',',
                    append=False
                )
            )
        
        return callbacks
    
    def get_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """Get optimizer based on configuration"""
        if self.training_config.optimizer.lower() == "adam":
            return tf.keras.optimizers.Adam(learning_rate=self.training_config.learning_rate)
        elif self.training_config.optimizer.lower() == "sgd":
            return tf.keras.optimizers.SGD(
                learning_rate=self.training_config.learning_rate,
                momentum=0.9,
                nesterov=True
            )
        elif self.training_config.optimizer.lower() == "rmsprop":
            return tf.keras.optimizers.RMSprop(learning_rate=self.training_config.learning_rate)
        else:
            logger.warning(f"Unknown optimizer: {self.training_config.optimizer}, using Adam")
            return tf.keras.optimizers.Adam(learning_rate=self.training_config.learning_rate)
    
    def get_data_generators(self, train_dir: str, val_dir: Optional[str] = None) -> tuple:
        """Get data generators based on augmentation configuration"""
        # Training data generator with augmentation
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=self.augmentation_config.rescale,
            rotation_range=int(self.augmentation_config.rotation_range),
            width_shift_range=self.augmentation_config.width_shift_range,
            height_shift_range=self.augmentation_config.height_shift_range,
            horizontal_flip=self.augmentation_config.horizontal_flip,
            zoom_range=self.augmentation_config.zoom_range,
            shear_range=self.augmentation_config.shear_range,
            brightness_range=self.augmentation_config.brightness_range,
            channel_shift_range=self.augmentation_config.channel_shift_range,
            fill_mode=self.augmentation_config.fill_mode,
            validation_split=self.training_config.validation_split if val_dir is None else 0.0
        )
        
        # Validation data generator (only rescaling)
        if val_dir is not None:
            val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=self.augmentation_config.rescale
            )
        else:
            val_datagen = train_datagen
        
        return train_datagen, val_datagen
    
    def save_config(self, filepath: str):
        """Save current configuration to file"""
        config_dict = {
            "model": asdict(self.model_config),
            "training": asdict(self.training_config),
            "augmentation": asdict(self.augmentation_config),
            "fine_tuning": asdict(self.fine_tuning_config),
            "callbacks": asdict(self.callback_config),
            "metadata": {
                "created_by": "DevanshSrajput",
                "created_at": "2025-06-29 09:17:52",
                "version": "2.0.0"
            }
        }
        
        filepath_obj = Path(filepath)
        filepath_obj.parent.mkdir(parents=True, exist_ok=True)
        
        if filepath_obj.suffix.lower() == '.yaml' or filepath_obj.suffix.lower() == '.yml':
            if yaml is not None:
                with open(filepath_obj, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            else:
                logger.warning("PyYAML not available, saving as JSON instead")
                with open(filepath_obj.with_suffix('.json'), 'w') as f:
                    json.dump(config_dict, f, indent=2)
        else:
            with open(filepath_obj, 'w') as f:
                json.dump(config_dict, f, indent=2)
        
        logger.info(f"Configuration saved to {filepath_obj}")
    
    def load_config(self, filepath: str):
        """Load configuration from file"""
        filepath_obj = Path(filepath)
        
        if not filepath_obj.exists():
            logger.error(f"Config file not found: {filepath_obj}")
            return
        
        try:
            if filepath_obj.suffix.lower() in ['.yaml', '.yml']:
                if yaml is not None:
                    with open(filepath_obj, 'r') as f:
                        config_dict = yaml.safe_load(f)
                else:
                    logger.error("PyYAML not available, cannot load YAML files")
                    return
            else:
                with open(filepath_obj, 'r') as f:
                    config_dict = json.load(f)
            
            # Load configurations
            if "model" in config_dict:
                self.model_config = ModelConfig(**config_dict["model"])
            
            if "training" in config_dict:
                self.training_config = TrainingConfig(**config_dict["training"])
            
            if "augmentation" in config_dict:
                self.augmentation_config = AugmentationConfig(**config_dict["augmentation"])
            
            if "fine_tuning" in config_dict:
                self.fine_tuning_config = FineTuningConfig(**config_dict["fine_tuning"])
            
            if "callbacks" in config_dict:
                self.callback_config = CallbackConfig(**config_dict["callbacks"])
            
            logger.info(f"Configuration loaded from {filepath_obj}")
            
        except Exception as e:
            logger.error(f"Error loading config: {e}")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration"""
        return {
            "model": {
                "type": self.model_config.model_type,
                "parameters": self.model_specs.get(self.model_config.model_type, {}).get("params", "Unknown"),
                "input_size": self.model_config.input_size,
                "num_classes": self.model_config.num_classes
            },
            "training": {
                "epochs": self.training_config.epochs,
                "batch_size": self.training_config.batch_size,
                "learning_rate": self.training_config.learning_rate,
                "optimizer": self.training_config.optimizer
            },
            "fine_tuning": {
                "enabled": self.fine_tuning_config.enabled,
                "epochs": self.fine_tuning_config.fine_tune_epochs if self.fine_tuning_config.enabled else 0
            },
            "estimated_time": self._estimate_training_time(),
            "target_accuracy": self._estimate_target_accuracy()
        }
    
    def _estimate_training_time(self) -> str:
        """Estimate training time based on configuration"""
        base_time_per_epoch = {
            "MobileNetV2": 1.5,
            "EfficientNetB0": 2.0,
            "EfficientNetB1": 2.5,
            "EfficientNetB2": 3.0,
            "ResNet50": 3.5,
            "InceptionV3": 4.0,
            "DenseNet121": 2.8
        }
        
        time_per_epoch = base_time_per_epoch.get(self.model_config.model_type, 2.5)
        
        # Adjust for batch size
        if self.training_config.batch_size < 16:
            time_per_epoch *= 1.5
        elif self.training_config.batch_size > 32:
            time_per_epoch *= 0.8
        
        total_epochs = self.training_config.epochs
        if self.fine_tuning_config.enabled:
            total_epochs += self.fine_tuning_config.fine_tune_epochs
        
        total_minutes = total_epochs * time_per_epoch
        
        if total_minutes < 60:
            return f"{total_minutes:.0f} minutes"
        else:
            hours = total_minutes / 60
            return f"{hours:.1f} hours"
    
    def _estimate_target_accuracy(self) -> str:
        """Estimate target accuracy based on configuration"""
        accuracy_ranges = {
            "MobileNetV2": (0.75, 0.82),
            "EfficientNetB0": (0.80, 0.87),
            "EfficientNetB1": (0.83, 0.90),
            "EfficientNetB2": (0.85, 0.92),
            "ResNet50": (0.82, 0.89),
            "InceptionV3": (0.81, 0.88),
            "DenseNet121": (0.82, 0.89)
        }
        
        base_range = accuracy_ranges.get(self.model_config.model_type, (0.75, 0.85))
        
        # Adjust for training configuration
        if self.training_config.epochs >= 50:
            base_range = (base_range[0] + 0.02, base_range[1] + 0.03)
        elif self.training_config.epochs <= 15:
            base_range = (base_range[0] - 0.03, base_range[1] - 0.02)
        
        if self.fine_tuning_config.enabled:
            base_range = (base_range[0] + 0.02, base_range[1] + 0.02)
        
        return f"{base_range[0]:.0%} - {base_range[1]:.0%}"
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of warnings/errors"""
        warnings = []
        
        # Model validation
        if self.model_config.model_type not in self.model_specs:
            warnings.append(f"Unknown model type: {self.model_config.model_type}")
        
        # Training validation
        if self.training_config.batch_size < 4:
            warnings.append("Very small batch size may cause training instability")
        elif self.training_config.batch_size > 64:
            warnings.append("Large batch size may require more memory")
        
        if self.training_config.learning_rate > 0.01:
            warnings.append("High learning rate may cause training instability")
        elif self.training_config.learning_rate < 1e-6:
            warnings.append("Very low learning rate may result in slow convergence")
        
        # Fine-tuning validation
        if self.fine_tuning_config.enabled and self.fine_tuning_config.fine_tune_learning_rate >= self.training_config.learning_rate:
            warnings.append("Fine-tuning learning rate should be lower than initial learning rate")
        
        return warnings

    def setup_fast_training_config(self):
        """Setup optimized training configuration using your existing presets"""
        from training_config import Food101TrainingConfig
        
        # Use your existing 'quick' preset for development
        config_manager = Food101TrainingConfig()
        config_manager.apply_preset('quick')  # This uses MobileNetV2 with optimized settings
        
        # Update your trainer config directly
        config_manager.model_config.model_type = 'MobileNetV2'  # Fastest from your presets
        config_manager.training_config.batch_size = 64  # Larger batches for efficiency
        config_manager.training_config.learning_rate = 0.001
        config_manager.training_config.epochs = 15  # Reduced for quick iterations
        config_manager.model_config.input_size = (224, 224)
        
        return config_manager

# Utility functions
def create_config_from_preset(preset_name: str, config_file: Optional[str] = None) -> Food101TrainingConfig:
    """Create configuration from preset"""
    config = Food101TrainingConfig(config_file)
    config.apply_preset(preset_name)
    return config

def compare_presets(preset_names: List[str]) -> Dict[str, Dict[str, Any]]:
    """Compare multiple presets"""
    comparison = {}
    
    config = Food101TrainingConfig()
    
    for preset_name in preset_names:
        if preset_name in config.presets:
            preset = config.presets[preset_name]
            comparison[preset_name] = {
                "name": preset["name"],
                "description": preset["description"],
                "estimated_time": preset["estimated_time"],
                "target_accuracy": preset["target_accuracy"],
                "model_type": preset["model"]["model_type"],
                "epochs": preset["training"]["epochs"],
                "batch_size": preset["training"]["batch_size"]
            }
    
    return comparison

# Example usage and testing
if __name__ == "__main__":
    # Create default configuration
    config = Food101TrainingConfig()
    
    print("Available presets:")
    for preset_name, preset_info in config.presets.items():
        print(f"  {preset_name}: {preset_info['name']}")
        print(f"    {preset_info['description']}")
        print(f"    Time: {preset_info['estimated_time']}")
        print(f"    Target: {preset_info['target_accuracy']}")
        print()
    
    # Test preset application
    config.apply_preset("balanced")
    print("Applied 'balanced' preset")
    
    # Get configuration summary
    summary = config.get_config_summary()
    print("Configuration Summary:")
    for section, details in summary.items():
        print(f"  {section}: {details}")
    
    # Validate configuration
    warnings = config.validate_config()
    if warnings:
        print("Configuration warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("Configuration is valid!")
    
    # Save configuration
    config.save_config("config/balanced_config.yaml")
    print("Configuration saved to config/balanced_config.yaml")