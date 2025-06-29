"""
Model Trainer for Food-101 Dataset
Handles model creation, training, fine-tuning, and evaluation
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from datetime import datetime
import logging
from typing import Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def configure_tensorflow_early():
    """Configure TensorFlow before any operations - call this first!"""
    try:
        import os
        
        # Set environment variables (always safe)
        os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
        os.environ['TF_NUM_INTRAOP_THREADS'] = str(os.cpu_count())
        os.environ['TF_NUM_INTEROP_THREADS'] = '2'
        
        # Try to set TensorFlow config (may fail if already initialized)
        try:
            tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count())
            tf.config.threading.set_inter_op_parallelism_threads(2)
            logger.info("Early TensorFlow threading configuration successful")
        except RuntimeError:
            logger.info("TensorFlow threading configured via environment variables")
            
        # GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Early GPU memory growth enabled for {len(gpus)} GPU(s)")
            except RuntimeError:
                logger.info("GPU configuration handled at runtime")
                
    except Exception as e:
        logger.warning(f"Early TensorFlow configuration warning: {e}")

class Food101ModelTrainer:
    """
    Comprehensive model trainer for Food-101 dataset with advanced features
    """
    
    def __init__(self, data_dir: str = "organized_food101", model_dir: str = "models"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.model = None
        self.training_history = None
        self.class_names = None
        self.num_classes = None
        
        # Add missing attributes for async training
        self.is_training = False
        self.training_logs = []
        
        # Training configuration
        self.config = {
            'img_size': (224, 224),
            'batch_size': 32,
            'learning_rate': 0.001,
            'epochs': 30,
            'validation_split': 0.2,
            'model_type': 'EfficientNetB0'
        }
        
        # Initialize TensorFlow configuration
        self._initialize_tensorflow_config()
        
        # Always use float32 for CPU optimization
        tf.keras.mixed_precision.set_global_policy('float32')
        
    def _initialize_tensorflow_config(self):
        """Initialize TensorFlow configuration at startup"""
        try:
            # CPU threading configuration
            import os
            
            # Set environment variables first (these should work even after TF init)
            os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
            os.environ['TF_NUM_INTRAOP_THREADS'] = str(os.cpu_count())
            os.environ['TF_NUM_INTEROP_THREADS'] = '2'
            
            # Try to set TensorFlow threading config (may fail if already initialized)
            try:
                tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count())
                tf.config.threading.set_inter_op_parallelism_threads(2)
                logger.info("TensorFlow threading configuration set successfully")
            except RuntimeError:
                logger.info("TensorFlow already initialized - threading config set via environment variables")
            
            # GPU configuration
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
                except RuntimeError:
                    logger.info("GPU configuration already set or TensorFlow already initialized")
            
        except Exception as e:
            logger.warning(f"TensorFlow configuration warning: {e}")
    
    def set_config(self, **kwargs):
        """Update training configuration"""
        self.config.update(kwargs)
        logger.info(f"Updated config: {kwargs}")
    
    def create_model(self, model_type: str = "EfficientNetB0", num_classes: int = 101) -> tf.keras.Model:
        """
        Create transfer learning model with various architectures
        
        Args:
            model_type: Type of base model ('EfficientNetB0', 'MobileNetV2', 'ResNet50', 'InceptionV3')
            num_classes: Number of output classes
            
        Returns:
            Compiled Keras model
        """
        logger.info(f"Creating {model_type} model with {num_classes} classes")
        
        # Base model selection
        base_models = {
            'EfficientNetB0': tf.keras.applications.EfficientNetB0,
            'EfficientNetB1': tf.keras.applications.EfficientNetB1,
            'EfficientNetB2': tf.keras.applications.EfficientNetB2,
            'MobileNetV2': tf.keras.applications.MobileNetV2,
            'ResNet50': tf.keras.applications.ResNet50,
            'ResNet101': tf.keras.applications.ResNet101,
            'InceptionV3': tf.keras.applications.InceptionV3,
            'DenseNet121': tf.keras.applications.DenseNet121,
            'VGG16': tf.keras.applications.VGG16
        }
        
        if model_type not in base_models:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Create base model
        base_model = base_models[model_type](
            weights='imagenet',
            include_top=False,
            input_shape=(*self.config['img_size'], 3)
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Data augmentation layers
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomHeight(0.2),
            tf.keras.layers.RandomWidth(0.2),
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomContrast(0.2),
            tf.keras.layers.RandomBrightness(0.2)
        ], name="data_augmentation")
        
        # Preprocessing layer
        if model_type.startswith('EfficientNet'):
            preprocess_layer = tf.keras.applications.efficientnet.preprocess_input
        elif model_type.startswith('MobileNet'):
            preprocess_layer = tf.keras.applications.mobilenet_v2.preprocess_input
        elif model_type.startswith('ResNet'):
            preprocess_layer = tf.keras.applications.resnet.preprocess_input
        elif model_type.startswith('Inception'):
            preprocess_layer = tf.keras.applications.inception_v3.preprocess_input
        elif model_type.startswith('DenseNet'):
            preprocess_layer = tf.keras.applications.densenet.preprocess_input
        elif model_type.startswith('VGG'):
            preprocess_layer = tf.keras.applications.vgg16.preprocess_input
        else:
            preprocess_layer = lambda x: x  # No preprocessing for unknown models
        
        # Build complete model
        inputs = tf.keras.Input(shape=(*self.config['img_size'], 3))
        x = data_augmentation(inputs)
        x = preprocess_layer(x)
        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Classification head with multiple layers
        x = tf.keras.layers.Dense(512, activation='relu', name='dense_1')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        x = tf.keras.layers.Dense(256, activation='relu', name='dense_2')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='predictions')(x)
        
        self.model = tf.keras.Model(inputs, outputs, name=f"{model_type}_Food101")
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_5_accuracy']
        )
        
        # Store model info
        self.num_classes = num_classes
        
        logger.info(f"Model created with {self.model.count_params():,} parameters")
        return self.model
    
    def create_optimized_model(self, model_type: str = "MobileNetV2", num_classes: int = 101) -> tf.keras.Model:
        """Create CPU/GPU optimized model"""
        
        # Check if GPU is available
        gpus = tf.config.experimental.list_physical_devices('GPU')
        has_gpu = len(gpus) > 0
        
        # Only enable mixed precision if GPU is available
        if has_gpu:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            logger.info("Mixed precision enabled for GPU")
        else:
            tf.keras.mixed_precision.set_global_policy('float32')
            logger.info("Using float32 for CPU training")
        
        # Use MobileNetV2 for CPU (fastest option)
        if model_type in ['EfficientNetB0', 'EfficientNetB1'] and not has_gpu:
            model_type = 'MobileNetV2'  # Force MobileNet for CPU
            logger.info("Switched to MobileNetV2 for CPU optimization")
        
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.config['img_size'], 3),
            alpha=0.75 if not has_gpu else 1.0  # Reduce model size for CPU
        )
        
        base_model.trainable = False
        
        # CPU-optimized architecture
        inputs = tf.keras.Input(shape=(*self.config['img_size'], 3))
        x = base_model(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        # Smaller dense layer for CPU
        x = tf.keras.layers.Dense(128, activation='relu')(x)  # Reduced from 256
        x = tf.keras.layers.Dropout(0.2)(x)
        
        outputs = tf.keras.layers.Dense(
            num_classes, 
            activation='softmax',
            dtype='float32'  # Always use float32 output
        )(x)
        
        model = tf.keras.Model(inputs, outputs)
        
        # CPU-optimized optimizer
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        # Don't wrap with LossScaleOptimizer for CPU
        if has_gpu:
            try:
                # Use the modern way without LossScaleOptimizer
                # Mixed precision policy handles this automatically
                pass
            except Exception as e:
                logger.warning(f"Could not create LossScaleOptimizer: {e}")
                # Fall back to regular optimizer
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"Model created for {'GPU' if has_gpu else 'CPU'} training")
        return model
    
    def create_cpu_optimized_model(self, model_type: str = "MobileNetV2", num_classes: int = 101) -> tf.keras.Model:
        """Create CPU-optimized model for ultra-fast training"""
        logger.info(f"Creating CPU-optimized {model_type} model")
        
        # Ensure float32 policy for CPU
        tf.keras.mixed_precision.set_global_policy('float32')
        
        # Use MobileNetV2 with reduced complexity for CPU
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.config['img_size'], 3),
            alpha=0.5  # Significantly reduced for CPU speed
        )
        
        base_model.trainable = False
        
        # Very lightweight architecture for CPU
        inputs = tf.keras.Input(shape=(*self.config['img_size'], 3))
        x = base_model(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        # Single small dense layer for CPU speed
        x = tf.keras.layers.Dense(64, activation='relu')(x)  # Very small layer
        x = tf.keras.layers.Dropout(0.1)(x)
        
        outputs = tf.keras.layers.Dense(
            num_classes, 
            activation='softmax',
            dtype='float32'
        )(x)
        
        model = tf.keras.Model(inputs, outputs)
        
        # CPU-optimized optimizer (no mixed precision)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"CPU-optimized model created with {model.count_params():,} parameters")
        return model

    def prepare_data_generators(self) -> Tuple[tf.keras.utils.Sequence, tf.keras.utils.Sequence, tf.keras.utils.Sequence]:
        """
        Prepare data generators for training, validation, and testing
        
        Returns:
            Tuple of (train_generator, validation_generator, test_generator)
        """
        logger.info("Preparing data generators")
        
        train_dir = self.data_dir / "train"
        val_dir = self.data_dir / "val"
        test_dir = self.data_dir / "test"
        
        # Check if organized structure exists
        if not all([train_dir.exists(), val_dir.exists(), test_dir.exists()]):
            raise FileNotFoundError("Organized dataset not found. Run food101_downloader.py first.")
        
        # Advanced data augmentation for training
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.3,
            height_shift_range=0.3,
            horizontal_flip=True,
            zoom_range=0.3,
            shear_range=0.2,
            brightness_range=[0.7, 1.3],
            channel_shift_range=0.2,
            fill_mode='nearest'
        )
        
        # Only rescaling for validation and test
        val_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.config['img_size'],
            batch_size=self.config['batch_size'],
            class_mode='categorical',
            shuffle=True,
            seed=42
        )
        
        validation_generator = val_test_datagen.flow_from_directory(
            val_dir,
            target_size=self.config['img_size'],
            batch_size=self.config['batch_size'],
            class_mode='categorical',
            shuffle=False
        )
        
        test_generator = val_test_datagen.flow_from_directory(
            test_dir,
            target_size=self.config['img_size'],
            batch_size=self.config['batch_size'],
            class_mode='categorical',
            shuffle=False
        )
        
        # Store class names
        self.class_names = list(train_generator.class_indices.keys())
        self.num_classes = len(self.class_names)
        
        logger.info(f"Data generators created:")
        logger.info(f"  - Training samples: {train_generator.samples}")
        logger.info(f"  - Validation samples: {validation_generator.samples}")
        logger.info(f"  - Test samples: {test_generator.samples}")
        logger.info(f"  - Classes: {self.num_classes}")
        
        return train_generator, validation_generator, test_generator
    
    def prepare_optimized_data_pipeline(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Create highly optimized tf.data pipelines for maximum performance
        """
        logger.info("Creating optimized data pipeline")
        
        train_dir = self.data_dir / "train"
        val_dir = self.data_dir / "val"
        test_dir = self.data_dir / "test"
        
        # Get file paths and labels efficiently
        def get_dataset_info(directory):
            image_paths = []
            labels = []
            class_names = sorted([d.name for d in directory.iterdir() if d.is_dir()])
            
            for i, class_name in enumerate(class_names):
                class_dir = directory / class_name
                for img_path in class_dir.glob("*.jpg"):
                    image_paths.append(str(img_path))
                    labels.append(i)
            
            return image_paths, labels, class_names
        
        train_paths, train_labels, self.class_names = get_dataset_info(train_dir)
        val_paths, val_labels, _ = get_dataset_info(val_dir)
        test_paths, test_labels, _ = get_dataset_info(test_dir)
        
        self.num_classes = len(self.class_names)
        
        # Critical performance settings
        AUTOTUNE = tf.data.AUTOTUNE
        BATCH_SIZE = self.config['batch_size']
        
        def create_optimized_dataset(paths, labels, is_training=False):
            dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
            
            if is_training:
                dataset = dataset.shuffle(buffer_size=min(10000, len(paths)), seed=42)
            
            # Parallel image loading - this is the key optimization
            dataset = dataset.map(
                lambda path, label: self._load_and_preprocess_image(path, label, is_training),
                num_parallel_calls=AUTOTUNE
            )
            
            dataset = dataset.batch(BATCH_SIZE)
            
            if is_training:
                dataset = dataset.repeat()
            
            # Critical: prefetch for pipeline optimization
            dataset = dataset.prefetch(AUTOTUNE)
            
            return dataset
        
        train_dataset = create_optimized_dataset(train_paths, train_labels, is_training=True)
        val_dataset = create_optimized_dataset(val_paths, val_labels, is_training=False)
        test_dataset = create_optimized_dataset(test_paths, test_labels, is_training=False)
        
        # Calculate steps for training
        self.steps_per_epoch = len(train_paths) // BATCH_SIZE
        self.validation_steps = len(val_paths) // BATCH_SIZE
        
        return train_dataset, val_dataset, test_dataset

    @tf.function
    def _load_and_preprocess_image(self, image_path, label, is_training):
        """Optimized image loading with tf.function compilation"""
        # Load and decode image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        
        # Efficient resize
        image = tf.image.resize(image, self.config['img_size'])
        
        # Normalize
        image = tf.math.truediv(tf.cast(image, tf.float32), 255.0)
        
        # Apply augmentation only for training
        if is_training:
            image = self._apply_efficient_augmentation(image)
        
        # One-hot encode labels (labels come as integers from get_dataset_info)
        label = tf.one_hot(label, self.num_classes)
        
        return image, label

    @tf.function
    def _apply_efficient_augmentation(self, image):
        """Efficient data augmentation using only core tf.image operations"""
        
        # Random horizontal flip
        if tf.random.uniform([]) > 0.5:
            image = tf.image.flip_left_right(image)
        
        # Random brightness and contrast
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        
        # Random rotation using tf.image.rot90 (90-degree increments only for speed)
        if tf.random.uniform([]) > 0.7:
            # Random 90-degree rotations (0, 90, 180, 270 degrees)
            k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
            image = tf.image.rot90(image, k)
        
        # Additional augmentations using core TensorFlow
        # Random saturation
        if tf.random.uniform([]) > 0.6:
            image = tf.image.random_saturation(image, 0.7, 1.3)
        
        # Random hue (slight color shift)
        if tf.random.uniform([]) > 0.8:
            image = tf.image.random_hue(image, 0.1)
        
        # Ensure values stay in [0, 1] range
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image
    
    def get_callbacks(self, checkpoint_path: str, early_stopping_patience: int = 10) -> list:
        """
        Get training callbacks
        
        Args:
            checkpoint_path: Path to save model checkpoints
            early_stopping_patience: Patience for early stopping
            
        Returns:
            List of callbacks
        """
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=max(5, early_stopping_patience // 2),
                min_lr=1,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=f"logs/tensorboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                histogram_freq=1,
                write_graph=True,
                write_images=True
            ),
            tf.keras.callbacks.CSVLogger(
                f"logs/training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
        ]
        
        return callbacks
    
    def train_model(self, 
                   model_type: str = "EfficientNetB0",
                   epochs: int = 30,
                   fine_tune: bool = True,
                   fine_tune_at: int = 100,
                   save_model: bool = True) -> Dict[str, Any]:
        """
        Train the model with transfer learning and optional fine-tuning
        
        Args:
            model_type: Type of base model to use
            epochs: Number of training epochs
            fine_tune: Whether to fine-tune the base model
            fine_tune_at: Layer from which to start fine-tuning
            save_model: Whether to save the final model
            
        Returns:
            Training history and metrics
        """
        logger.info(f"Starting training with {model_type}")
        
        # Prepare data first to set num_classes
        train_gen, val_gen, test_gen = self.prepare_data_generators()
        
        # Create model with known num_classes
        model = self.create_model(model_type, self.num_classes or 101)
        
        # Training configuration
        checkpoint_path = self.model_dir / f"food101_{model_type}_best.h5"
        callbacks = self.get_callbacks(str(checkpoint_path))
        
        # Phase 1: Transfer Learning
        logger.info("Phase 1: Transfer Learning")
        initial_epochs = min(epochs, 20) if fine_tune else epochs
        
        history_1 = model.fit(
            train_gen,
            epochs=initial_epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose="auto"
        )
        
        # Phase 2: Fine-tuning (if enabled)
        if fine_tune and epochs > 20:
            logger.info("Phase 2: Fine-tuning")
            
            # Find base model more safely
            base_model = None
            for layer in model.layers:
                if hasattr(layer, 'layers') and len(layer.layers) > 10:  # Likely a base model
                    base_model = layer
                    break
            
            if base_model is None:
                logger.warning("Could not find base model for fine-tuning, skipping fine-tuning phase")
            else:
                # Unfreeze base model
                base_model.trainable = True
                
                # Fine-tune from specific layer (with bounds checking)
                total_layers = len(base_model.layers)
                fine_tune_from = min(fine_tune_at, total_layers)
                
                for i, layer in enumerate(base_model.layers):
                    layer.trainable = i >= fine_tune_from
                
                # Lower learning rate for fine-tuning
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate'] / 10),
                    loss='categorical_crossentropy',
                    metrics=['accuracy', 'top_5_accuracy']
                )
                
                # Continue training
                remaining_epochs = epochs - initial_epochs
                history_2 = model.fit(
                    train_gen,
                    epochs=remaining_epochs,
                    validation_data=val_gen,
                    callbacks=callbacks,
                    verbose="auto",
                    initial_epoch=initial_epochs
                )
                
                # Safely combine histories
                try:
                    if (history_1 and history_2 and 
                        hasattr(history_1, 'history') and hasattr(history_2, 'history') and
                        history_1.history is not None and history_2.history is not None):
                        
                        if isinstance(history_1.history, dict) and isinstance(history_2.history, dict):
                            # Convert keys to list to avoid "Never" iteration issue
                            history_keys = list(history_1.history.keys()) if history_1.history else []
                            for key in history_keys:
                                if key in history_2.history:
                                    # Ensure both are lists before concatenating
                                    hist1_values = list(history_1.history[key]) if history_1.history[key] else []
                                    hist2_values = list(history_2.history[key]) if history_2.history[key] else []
                                    history_1.history[key] = hist1_values + hist2_values
                except Exception as e:
                    logger.warning(f"Could not combine training histories: {e}")
        
        self.training_history = history_1
        
        # Evaluate on test set
        logger.info("Evaluating on test set")
        test_results = model.evaluate(test_gen, verbose="auto")
        
        # Safely access history values
        try:
            if history_1 and hasattr(history_1, 'history') and history_1.history is not None:
                final_train_acc = history_1.history.get('accuracy', [0.0])[-1] if history_1.history.get('accuracy') else 0.0
                final_val_acc = history_1.history.get('val_accuracy', [0.0])[-1] if history_1.history.get('val_accuracy') else 0.0
                final_train_loss = history_1.history.get('loss', [0.0])[-1] if history_1.history.get('loss') else 0.0
                final_val_loss = history_1.history.get('val_loss', [0.0])[-1] if history_1.history.get('val_loss') else 0.0
            else:
                final_train_acc = final_val_acc = final_train_loss = final_val_loss = 0.0
        except (KeyError, IndexError, TypeError) as e:
            logger.warning(f"Could not access history values: {e}")
            final_train_acc = final_val_acc = final_train_loss = final_val_loss = 0.0
        
        # Create results dictionary
        results = {
            'model_type': model_type,
            'epochs_trained': epochs,
            'final_train_accuracy': final_train_acc,
            'final_val_accuracy': final_val_acc,
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'test_loss': test_results[0] if len(test_results) > 0 else 0.0,
            'test_accuracy': test_results[1] if len(test_results) > 1 else 0.0,
            'test_top5_accuracy': test_results[2] if len(test_results) > 2 else None,
            'num_classes': self.num_classes,
            'training_date': datetime.now().isoformat(),
            'model_params': model.count_params()
        }
        
        # Store the trained model
        self.model = model
        
        # Save model and metadata
        if save_model:
            self.save_model_and_metadata(model_type, results)
        
        logger.info(f"Training completed. Test accuracy: {results['test_accuracy']:.4f}")
        return results
    
    def setup_fast_training_config(self):
        """Configure settings for fast training"""
        # Reduce image size for faster processing
        self.config['img_size'] = (160, 160)  # Smaller than original 224x224
        
        # Increase batch size if GPU memory allows
        self.config['batch_size'] = 64  # Adjust based on available memory
        
        # Use a smaller learning rate for mixed precision training
        self.config['learning_rate'] = 0.0005
        
        # Reduce epochs for faster iterations during development
        self.config['epochs'] = 15
        
        # Use simpler model variant
        self.config['model_type'] = 'MobileNetV2'
        
        logger.info("Fast training configuration set up")
    
    def enable_quick_dev_mode(self):
        """Ultra-fast mode for rapid development and testing"""
        logger.info("Enabling quick development mode")
        
        self.config.update({
            'img_size': (128, 128),  # Smaller images = faster training
            'batch_size': 128,       # Larger batches
            'epochs': 5,             # Very few epochs
            'model_type': 'MobileNetV2',
            'learning_rate': 0.01    # Higher LR for faster convergence
        })
        
        # Use subset of data for quick testing
        self.use_data_subset = True
        self.subset_size = 100  # Only 100 samples per class
        
        logger.info("Quick dev mode enabled - expect 10-20x faster training")
    
    def train_optimized_model(self, 
                         model_type: str = "MobileNetV2",
                         epochs: int = 15,
                         **kwargs) -> Dict[str, Any]:
        """Optimized training with performance monitoring"""
        
        # Setup fast config
        self.setup_fast_training_config()
        
        # Enable GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info("GPU memory growth enabled")
            except RuntimeError as e:
                logger.warning(f"GPU setup failed: {e}")
        
        # Use optimized data pipeline
        train_ds, val_ds, test_ds = self.prepare_optimized_data_pipeline()
        
        # Create optimized model
        model = self.create_optimized_model(model_type, self.num_classes or 101)
        
        # Streamlined callbacks (remove expensive ones during development)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,  # Reduced patience
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1
            )
            # Remove TensorBoard and CSV logger for speed during development
        ]
        
        # Train with steps_per_epoch for better control
        logger.info(f"Starting optimized training: {self.steps_per_epoch} steps/epoch")
        
        history = model.fit(
            train_ds,
            steps_per_epoch=self.steps_per_epoch,
            epochs=epochs,
            validation_data=val_ds,
            validation_steps=self.validation_steps,
            callbacks=callbacks,
            verbose="auto"
        )
        
        # Store results
        self.model = model
        self.training_history = history
        
        # Quick evaluation
        test_results = model.evaluate(test_ds, steps=self.validation_steps, verbose="auto")
        
        results = {
            'model_type': model_type,
            'epochs_trained': epochs,
            'test_accuracy': test_results[1] if len(test_results) > 1 else 0.0,
            'training_time': 'optimized',
            'optimization': 'enabled'
        }
        
        logger.info(f"Optimized training completed. Test accuracy: {results['test_accuracy']:.4f}")
        return results
    
    def save_model_and_metadata(self, model_type: str, results: Dict[str, Any]):
        """Save trained model and metadata"""
        if self.model is None:
            logger.error("No model to save")
            return
            
        # Save model
        model_path = self.model_dir / f"food101_{model_type}_final.h5"
        self.model.save(str(model_path))
        logger.info(f"Model saved to {model_path}")
        
        # Save class names
        class_names_path = self.model_dir / "food101_classes.json"
        if self.class_names is not None:
            with open(class_names_path, 'w') as f:
                json.dump(self.class_names, f, indent=2)
        
        # Save training metadata
        metadata_path = self.model_dir / f"training_metadata_{model_type}.json"
        results['class_names'] = self.class_names if self.class_names is not None else []
        with open(metadata_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save training history
        if self.training_history is not None and hasattr(self.training_history, 'history'):
            history_path = self.model_dir / f"training_history_{model_type}.json"
            try:
                if self.training_history.history and isinstance(self.training_history.history, dict):
                    history_items = list(self.training_history.history.items()) if self.training_history.history else []
                    history_dict = {key: [float(val) for val in values] 
                                   for key, values in history_items
                                   if values is not None}
                    with open(history_path, 'w') as f:
                        json.dump(history_dict, f, indent=2)
                    logger.info("Training history saved")
            except Exception as e:
                logger.warning(f"Could not save training history: {e}")
        
        logger.info("Metadata saved")
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        if self.training_history is None:
            logger.warning("No training history found")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy plot
        axes[0, 0].plot(self.training_history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.training_history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss plot
        axes[0, 1].plot(self.training_history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.training_history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Top-5 Accuracy (if available)
        if 'top_5_accuracy' in self.training_history.history:
            axes[1, 0].plot(self.training_history.history['top_5_accuracy'], label='Training Top-5 Accuracy')
            axes[1, 0].plot(self.training_history.history['val_top_5_accuracy'], label='Validation Top-5 Accuracy')
            axes[1, 0].set_title('Model Top-5 Accuracy')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Top-5 Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Learning rate (if available)
        if hasattr(self.training_history, 'lr'):
            axes[1, 1].plot(self.training_history.history.get('lr', []))
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, 'Learning Rate\nNot Available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training plots saved to {save_path}")
        
        plt.show()
    
    def evaluate_model(self, model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Comprehensive model evaluation
        
        Args:
            model_path: Path to saved model (if None, uses current model)
            
        Returns:
            Evaluation metrics and visualizations
        """
        if model_path:
            self.model = tf.keras.models.load_model(model_path)
            logger.info(f"Loaded model from {model_path}")
        
        if self.model is None:
            raise ValueError("No model available for evaluation")
        
        # Prepare test data
        _, _, test_gen = self.prepare_data_generators()
        
        # Generate predictions
        logger.info("Generating predictions...")
        predictions = self.model.predict(test_gen, verbose="auto")
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Get true labels - use a safer approach
        try:
            if hasattr(test_gen, 'classes'):
                true_classes = getattr(test_gen, 'classes')
            else:
                # Fallback: generate labels from the generator
                true_classes = []
                for i in range(len(test_gen)):
                    batch_x, batch_y = test_gen[i]
                    true_classes.extend(np.argmax(batch_y, axis=1))
                true_classes = np.array(true_classes)
        except Exception as e:
            logger.error(f"Could not extract true classes: {e}")
            return {"error": "Could not extract true classes for evaluation"}
        
        # Calculate metrics
        test_loss, test_accuracy = self.model.evaluate(test_gen, verbose="auto")[:2]
        
        # Classification report
        class_report = classification_report(
            true_classes, predicted_classes,
            target_names=self.class_names,
            output_dict=True
        )
        
        # Top-k accuracy
        try:
            # Create one-hot encoded labels for top-k calculation
            true_labels_one_hot = tf.keras.utils.to_categorical(true_classes, num_classes=self.num_classes)
            top_5_acc = tf.keras.metrics.top_k_categorical_accuracy(
                true_labels_one_hot, predictions, k=5
            ).numpy().mean()
        except Exception as e:
            logger.warning(f"Could not calculate top-5 accuracy: {e}")
            top_5_acc = 0.0
        
        results = {
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'top_5_accuracy': top_5_acc,
            'classification_report': class_report,
            'num_samples': len(true_classes),
            'evaluation_date': datetime.now().isoformat()
        }
        
        # Save confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        self._plot_confusion_matrix(cm, save_path=str(self.model_dir / "confusion_matrix.png"))
        
        logger.info(f"Evaluation completed. Test accuracy: {test_accuracy:.4f}")
        return results
    
    def _plot_confusion_matrix(self, cm: np.ndarray, save_path: Optional[str] = None):
        """Plot confusion matrix"""
        if self.class_names is None or len(self.class_names) == 0:
            logger.warning("No class names available for confusion matrix")
            return
            
        plt.figure(figsize=(20, 16))
        
        # For large number of classes, show only a subset
        if len(self.class_names) > 20:
            # Show top 20 classes by frequency
            class_counts = np.sum(cm, axis=1)
            top_indices = np.argsort(class_counts)[-20:]
            cm_subset = cm[np.ix_(top_indices, top_indices)]
            class_names_subset = [self.class_names[i] for i in top_indices]
        else:
            cm_subset = cm
            class_names_subset = self.class_names
        
        sns.heatmap(cm_subset, 
                   annot=True if len(class_names_subset) <= 10 else False,
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=class_names_subset,
                   yticklabels=class_names_subset)
        
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()

    def diagnose_gpu_setup(self):
        """Comprehensive GPU diagnostics for RTX 3050"""
        print("=== RTX 3050 GPU Diagnostics ===")
        
        # Check TensorFlow version
        print(f"TensorFlow Version: {tf.__version__}")
        
        # Check if TensorFlow was built with CUDA
        print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
        
        # List physical devices
        print("Physical devices:")
        for device in tf.config.list_physical_devices():
            print(f"  - {device}")
        
        # Check GPU devices specifically
        gpus = tf.config.list_physical_devices('GPU')
        print(f"GPU devices found: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu}")
        
        # Check logical devices
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"Logical GPU devices: {len(logical_gpus)}")
        
        # Test GPU availability
        print(f"GPU Available: {tf.test.is_gpu_available()}")
        
        # GPU memory info
        if gpus:
            try:
                gpu_details = tf.config.experimental.get_device_details(gpus[0])
                print(f"GPU Details: {gpu_details}")
            except Exception as e:
                print(f"Could not get GPU details: {e}")
        
        print("=== End Diagnostics ===\n")
        return len(gpus) > 0

    def initialize_rtx_3050(self):
        """Initialize RTX 3050 GPU for optimal performance"""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        
        if gpus:
            try:
                # RTX 3050 specific optimizations
                for gpu in gpus:
                    # Enable memory growth to prevent allocation issues
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                        # Set memory limit for RTX 3050 (8GB VRAM, leave some headroom)
                        tf.config.experimental.set_virtual_device_configuration(
                            gpu,
                            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7168)]
                        )
                    except RuntimeError as e:
                        logger.warning(f"Could not configure GPU memory (already initialized): {e}")
                
                # Enable XLA compilation for speed
                try:
                    tf.config.optimizer.set_jit(True)
                except RuntimeError as e:
                    logger.warning(f"Could not enable XLA (already initialized): {e}")
                
                # Enable mixed precision (RTX 3050 supports it well)
                try:
                    policy = tf.keras.mixed_precision.Policy('mixed_float16')
                    tf.keras.mixed_precision.set_global_policy(policy)
                except RuntimeError as e:
                    logger.warning(f"Could not set mixed precision policy (already initialized): {e}")
                
                logger.info(f"RTX 3050 initialized successfully: {len(gpus)} GPU(s) available")
                logger.info("Mixed precision and XLA enabled for maximum performance")
                return True
                
            except RuntimeError as e:
                logger.error(f"RTX 3050 initialization failed: {e}")
                return False
        else:
            logger.warning("No GPU devices found!")
            return False

    def setup_rtx_3050_ultra_fast_config(self):
        """Ultra-fast configuration specifically for RTX 3050 - 10 minute target"""
        logger.info("Setting up RTX 3050 ultra-fast configuration (10-minute target)")
        
        # Ultra-fast settings for RTX 3050
        self.config.update({
            'img_size': (160, 160),      # Smaller images for speed
            'batch_size': 96,            # Optimized for RTX 3050 8GB VRAM
            'learning_rate': 0.002,      # Higher LR for faster convergence
            'epochs': 8,                 # Maximum 8 epochs for 10-minute target
            'model_type': 'MobileNetV2'  # Fastest architecture
        })
        
        # Use subset of data for ultra-fast training
        self.use_data_subset = True
        self.subset_size = 200  # 200 samples per class (enough for good accuracy)
        
        logger.info("RTX 3050 ultra-fast mode enabled - targeting 8-10 minutes training time")

    def prepare_subset_data_pipeline(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Prepare optimized data pipeline for ultra-fast training"""
        logger.info("🔄 Setting up ultra-fast data pipeline")
        
        train_dir = self.data_dir / "train"
        val_dir = self.data_dir / "val"
        test_dir = self.data_dir / "test"
        
        # Get class names
        self.class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        self.num_classes = len(self.class_names)
        
        logger.info(f"Found {self.num_classes} classes: {self.class_names[:5]}...")
        
        # Use smaller subset for ultra-fast training
        if self.use_data_subset:
            subset_size = getattr(self, 'subset_size', 200)
            logger.info(f"Using subset: {subset_size} samples per class")
    
        # Create datasets from directory with proper parameters
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            validation_split=None,  # Don't split here, use separate directories
            image_size=self.config['img_size'],
            batch_size=self.config['batch_size'],
            shuffle=True,
            seed=123,
            label_mode='categorical'  # Use categorical labels for categorical_crossentropy
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            validation_split=None,
            image_size=self.config['img_size'],
            batch_size=self.config['batch_size'],
            shuffle=False,
            seed=123,
            label_mode='categorical'  # Use categorical labels for categorical_crossentropy
        )

        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            validation_split=None,
            image_size=self.config['img_size'],
            batch_size=self.config['batch_size'],
            shuffle=False,
            seed=123,
            label_mode='categorical'  # Use categorical labels for categorical_crossentropy
        )

        # Ensure datasets are tf.data.Dataset objects
        if not isinstance(train_ds, tf.data.Dataset):
            raise TypeError(f"Expected tf.data.Dataset, got {type(train_ds)}")
        if not isinstance(val_ds, tf.data.Dataset):
            raise TypeError(f"Expected tf.data.Dataset, got {type(val_ds)}")
        if not isinstance(test_ds, tf.data.Dataset):
            raise TypeError(f"Expected tf.data.Dataset, got {type(test_ds)}")

        # Apply subset if needed
        if self.use_data_subset:
            subset_size = getattr(self, 'subset_size', 200)
            # Calculate steps needed for subset
            total_samples = subset_size * self.num_classes
            subset_steps = total_samples // self.config['batch_size']
            
            # Take only the subset, then repeat to ensure complete epochs
            train_ds = train_ds.take(subset_steps).repeat()
            val_ds = val_ds.take(subset_steps // 4).repeat()  # Smaller validation set
            test_ds = test_ds.take(subset_steps // 4)  # Don't repeat test set
            
            # Update steps per epoch
            self.steps_per_epoch = subset_steps
            self.validation_steps = max(1, subset_steps // 4)
            
            logger.info(f"Subset configuration: {subset_steps} steps/epoch, {self.validation_steps} val steps")
        else:
            # Calculate steps for full dataset
            train_count = len(list(train_dir.glob('*/*.jpg')))
            val_count = len(list(val_dir.glob('*/*.jpg')))
            
            self.steps_per_epoch = train_count // self.config['batch_size']
            self.validation_steps = val_count // self.config['batch_size']
    
        # Optimize performance without cache to avoid the warning
        AUTOTUNE = tf.data.AUTOTUNE
        
        def preprocess_data(image, label):
            # Normalize to [0,1] range
            image = tf.math.divide(tf.cast(image, tf.float32), 255.0)
            # Labels are already one-hot encoded when using label_mode='categorical'
            # Ensure labels are float32 for compatibility with mixed precision
            label = tf.cast(label, tf.float32)
            return image, label
        
        # Apply preprocessing and optimization
        train_ds = (train_ds
                    .map(preprocess_data, num_parallel_calls=AUTOTUNE)
                    .prefetch(AUTOTUNE))
    
        val_ds = (val_ds
                  .map(preprocess_data, num_parallel_calls=AUTOTUNE)
                  .prefetch(AUTOTUNE))
    
        test_ds = (test_ds
                   .map(preprocess_data, num_parallel_calls=AUTOTUNE)
                   .prefetch(AUTOTUNE))
    
        logger.info(f"✅ Data pipeline ready: {self.steps_per_epoch} steps/epoch, {self.validation_steps} validation steps")
        
        return train_ds, val_ds, test_ds

    @tf.function
    def _load_and_preprocess_image_fast(self, image_path, label, is_training):
        """Ultra-fast image loading optimized for RTX 3050"""
        # Load and decode image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        
        # Efficient resize
        image = tf.image.resize(image, self.config['img_size'], method='bilinear')
        
        # Fast normalization
        image = tf.math.truediv(tf.cast(image, tf.float32), 255.0)
        
        # Minimal augmentation for speed
        if is_training:
            # Only horizontal flip for speed
            if tf.random.uniform([]) > 0.5:
                image = tf.image.flip_left_right(image)
            
            # Quick brightness adjustment
            image = tf.image.random_brightness(image, 0.1)
        
        # Labels are already one-hot encoded when using label_mode='categorical'
        # No need to apply tf.one_hot again
        
        return image, label

    def create_rtx_3050_model(self, model_type: str = "MobileNetV2", num_classes: int = 101) -> tf.keras.Model:
        """Create ultra-optimized model for RTX 3050"""
        logger.info(f"Creating RTX 3050 optimized {model_type} model")
        
        # Use MobileNetV2 with reduced alpha for speed
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.config['img_size'], 3),
            alpha=0.75  # Reduced model complexity for speed
        )
        
        base_model.trainable = False
        
        # Ultra-lightweight architecture for speed
        inputs = tf.keras.Input(shape=(*self.config['img_size'], 3))
        x = base_model(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        # Single dense layer for speed
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        
        outputs = tf.keras.layers.Dense(
            num_classes, 
            activation='softmax',
            dtype='float32'  # Always use float32 for loss computation compatibility
        )(x)
        
        model = tf.keras.Model(inputs, outputs)
        
        # Optimized optimizer for RTX 3050
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        # Wrap with mixed precision optimizer
        try:
            # Use modern mixed precision - no need for LossScaleOptimizer
            pass
        except Exception as e:
            logger.warning(f"Could not create LossScaleOptimizer: {e}")
            # Use regular optimizer
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy'],
            jit_compile=True  # Enable XLA compilation
        )
        
        logger.info(f"RTX 3050 model created with {model.count_params():,} parameters")
        return model

    def train_rtx_3050_ultra_fast(self, model_type: str = "MobileNetV2", epochs: int = 8) -> Dict[str, Any]:
        """Ultra-fast training specifically optimized for RTX 3050 - 10 minute target"""
        
        # Diagnose GPU first
        gpu_available = self.diagnose_gpu_setup()
        
        if not gpu_available:
            logger.error("No GPU detected! RTX 3050 should be available.")
            logger.info("Falling back to CPU ultra-fast mode...")
            return self.train_cpu_ultra_fast(epochs=3)
        
        # Initialize RTX 3050
        if not self.initialize_rtx_3050():
            logger.error("RTX 3050 initialization failed!")
            return self.train_cpu_fallback(epochs=3)
        
        logger.info("🎮 RTX 3050 detected and initialized - starting ultra-fast training")
        
        # Setup ultra-fast configuration
        self.setup_rtx_3050_ultra_fast_config()
        
        # Use subset data pipeline for speed
        train_ds, val_ds, test_ds = self.prepare_subset_data_pipeline()
        
        # Create RTX 3050 optimized model
        model = self.create_rtx_3050_model(model_type, self.num_classes or 101)
        
        # Minimal callbacks for maximum speed
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=3,  # Increased patience to allow more training
                restore_best_weights=True,
                verbose=1,
                mode='max'
            )
        ]
        
        # Start training
        logger.info(f"🚀 Starting RTX 3050 ultra-fast training: {self.steps_per_epoch} steps/epoch for {epochs} epochs")
        start_time = datetime.now()
        
        # Ensure training completes by using proper parameters
        history = model.fit(
            train_ds,
            steps_per_epoch=self.steps_per_epoch,
            epochs=epochs,
            validation_data=val_ds,
            validation_steps=self.validation_steps,
            callbacks=callbacks,
            verbose="auto",
            workers=4,
            use_multiprocessing=True
        )
        
        end_time = datetime.now()
        training_duration = (end_time - start_time).total_seconds() / 60
        
        # Store results
        self.model = model
        self.training_history = history
        
        # Quick evaluation - use a fixed number of steps to avoid dataset truncation
        logger.info("🔍 Evaluating model...")
        test_results = model.evaluate(
            test_ds, 
            steps=min(50, self.validation_steps),  # Use more steps for better evaluation
            verbose="auto",
            return_dict=True
        )
        
        # Save model using the new format
        try:
            os.makedirs('models', exist_ok=True)
            model_name = f'food101_rtx3050_ultrafast_{epochs}epochs.keras'
            model_path = f'models/{model_name}'
            
            logger.info(f"💾 Saving model to {model_path}...")
            
            # Reset to float32 policy before saving to avoid mixed precision issues
            original_policy = tf.keras.mixed_precision.global_policy()
            tf.keras.mixed_precision.set_global_policy('float32')
            
            # Save in the new Keras format
            model.save(model_path, save_format='keras')
            
            # Restore original policy
            tf.keras.mixed_precision.set_global_policy(original_policy)
            
            logger.info(f"✅ Model saved successfully as {model_name}")
            
            # Also save class names
            if self.class_names:
                class_names_path = 'models/food101_classes.json'
                with open(class_names_path, 'w') as f:
                    json.dump(self.class_names, f, indent=2)
                logger.info("📋 Class names saved")
            
        except Exception as e:
            logger.warning(f"⚠️ Model saving failed: {e} - continuing without saving")
        
        # Extract accuracy from results
        test_accuracy = test_results.get('accuracy', 0.0) if isinstance(test_results, dict) else (test_results[1] if len(test_results) > 1 else 0.0)
        
        results = {
            'model_type': model_type,
            'epochs_trained': epochs,
            'test_accuracy': test_accuracy,
            'training_duration_minutes': round(training_duration, 2),
            'hardware': 'RTX 3050',
            'optimization': 'ultra_fast',
            'data_subset_size': getattr(self, 'subset_size', 200),
            'target_achieved': training_duration <= 10,
            'steps_per_epoch': self.steps_per_epoch,
            'validation_steps': self.validation_steps
        }
        
        logger.info(f"✅ RTX 3050 ultra-fast training completed!")
        logger.info(f"📊 Results: {results['test_accuracy']:.3f} accuracy in {results['training_duration_minutes']:.1f} minutes")
        logger.info(f"🎯 Target {'ACHIEVED' if results['target_achieved'] else 'MISSED'} (≤10 minutes)")
        logger.info(f"📈 Epochs completed: {results['epochs_trained']}/{epochs}")
        
        return results

    def train_cpu_fallback(self, epochs: int = 3) -> Dict[str, Any]:
        """Emergency CPU fallback with ultra-fast settings"""
        logger.info("💻 Running CPU fallback mode")
        
        # CPU ultra-fast settings
        self.config.update({
            'img_size': (96, 96),
            'batch_size': 32,
            'epochs': epochs,
            'model_type': 'MobileNetV2'
        })
        
        self.use_data_subset = True
        self.subset_size = 50  # Very small subset for CPU
        
        # Use subset pipeline
        train_ds, val_ds, test_ds = self.prepare_subset_data_pipeline()
        
        # Create CPU model (no mixed precision)
        tf.keras.mixed_precision.set_global_policy('float32')
        model = self.create_optimized_model("MobileNetV2", self.num_classes or 101)
        
        # Train quickly
        history = model.fit(
            train_ds,
            steps_per_epoch=self.steps_per_epoch,
            epochs=epochs,
            validation_data=val_ds,
            validation_steps=self.validation_steps,
            verbose="auto"
        )
        
        test_results = model.evaluate(test_ds, steps=5, verbose="auto")
        
        return {
            'model_type': 'MobileNetV2',
            'epochs_trained': epochs,
            'test_accuracy': test_results[1] if len(test_results) > 1 else 0.0,
             'hardware': 'CPU_FALLBACK',
            'optimization': 'emergency_fallback'
        }

    def train_cpu_ultra_fast(self, model_type: str = "MobileNetV2", epochs: int = 5) -> Dict[str, Any]:
        """Ultra-fast CPU training - 10 minute target with CPU optimizations"""
        logger.info("💻 Starting CPU ultra-fast training (10-minute target)")
        
        # CPU ultra-fast configuration
        self.config.update({
            'img_size': (112, 112),     # Very small for CPU speed
            'batch_size': 64,           # Optimal for CPU
            'learning_rate': 0.003,     # Higher LR for faster convergence
            'epochs': epochs,           # 5 epochs max
            'model_type': 'MobileNetV2'
        })
        
        # Use very small subset for ultra-fast training
        self.use_data_subset = True
        self.subset_size = 100  # Only 100 samples per class (10,100 total)
        
        logger.info("CPU configuration: 112x112 images, 100 samples/class, 5 epochs")
        
        # CPU optimization environment variables (safe to set multiple times)
        os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
        os.environ['TF_NUM_INTRAOP_THREADS'] = str(os.cpu_count())
        os.environ['TF_NUM_INTEROP_THREADS'] = '2'
        
        # Use subset data pipeline
        train_ds, val_ds, test_ds = self.prepare_subset_data_pipeline()
        
        # Create CPU-optimized model
        model = self.create_cpu_optimized_model("MobileNetV2", self.num_classes or 101)
        
        # Minimal callbacks for speed
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=3,  # Increased patience
                restore_best_weights=True,
                verbose=1,
                mode='max'
            )
        ]
        
        # Start training with timing
        start_time = datetime.now()
        logger.info(f"🚀 Starting CPU ultra-fast training: {self.steps_per_epoch} steps/epoch for {epochs} epochs")
        
        history = model.fit(
            train_ds,
            steps_per_epoch=self.steps_per_epoch,
            epochs=epochs,
            validation_data=val_ds,
            validation_steps=self.validation_steps,
            callbacks=callbacks,
            verbose="auto",
            workers=1,  # Single worker for CPU
            use_multiprocessing=False
        )
        
        end_time = datetime.now()
        training_duration = (end_time - start_time).total_seconds() / 60
        
        # Store results
        self.model = model
        self.training_history = history
        
        # Quick evaluation with proper steps
        logger.info("🔍 Evaluating model...")
        test_results = model.evaluate(
            test_ds, 
            steps=min(20, self.validation_steps), 
            verbose="auto",
            return_dict=True
        )
        
        # Save model using the new format
        try:
            os.makedirs('models', exist_ok=True)
            model_name = f'food101_cpu_ultrafast_{epochs}epochs.keras'
            model_path = f'models/{model_name}'
            
            logger.info(f"💾 Saving model to {model_path}...")
            
            # Save in the new Keras format
            model.save(model_path, save_format='keras')
            
            logger.info(f"✅ Model saved successfully as {model_name}")
            
            # Also save class names
            if self.class_names:
                class_names_path = 'models/food101_classes.json'
                with open(class_names_path, 'w') as f:
                    json.dump(self.class_names, f, indent=2)
                logger.info("📋 Class names saved")
            
        except Exception as e:
            logger.warning(f"⚠️ Model saving failed: {e} - continuing without saving")
    
        # Extract accuracy from results
        test_accuracy = test_results.get('accuracy', 0.0) if isinstance(test_results, dict) else (test_results[1] if len(test_results) > 1 else 0.0)
        
        results = {
            'model_type': model_type,
            'epochs_trained': len(history.history['loss']) if history and hasattr(history, 'history') and history.history else epochs,  # Actual epochs completed
            'test_accuracy': test_accuracy,
            'training_duration_minutes': round(training_duration, 2),
            'hardware': 'CPU_OPTIMIZED',
            'optimization': 'ultra_fast_cpu',
            'data_subset_size': self.subset_size,
            'target_achieved': training_duration <= 10,
            'steps_per_epoch': self.steps_per_epoch,
            'validation_steps': self.validation_steps
        }
        
        logger.info(f"✅ CPU ultra-fast training completed!")
        logger.info(f"📊 Results: {results['test_accuracy']:.3f} accuracy in {results['training_duration_minutes']:.1f} minutes")
        logger.info(f"🎯 Target {'ACHIEVED' if results['target_achieved'] else 'CLOSE'} (≤10 minutes)")
        logger.info(f"📈 Epochs completed: {results['epochs_trained']}/{epochs}")
        
        return results

    def train_model_async(self, data_dir, epochs, batch_size, model_type="MobileNetV2"):
        """Ultra-fast training process optimized for both GPU and CPU"""
        try:
            self.is_training = True
            self.training_logs.append("🚀 Starting ultra-fast training...")
            
            # Update data directory
            self.data_dir = Path(data_dir)
            
            # Check for GPU first
            gpus = tf.config.list_physical_devices('GPU')
            
            if gpus:
                self.training_logs.append("🎮 GPU detected - attempting RTX 3050 ultra-fast training...")
                try:
                    results = self.train_rtx_3050_ultra_fast(
                        model_type=model_type,
                        epochs=min(epochs, 8)  # Cap at 8 epochs for 10-minute target
                    )
                except Exception as gpu_error:
                    self.training_logs.append(f"⚠️ GPU training failed: {str(gpu_error)}")
                    self.training_logs.append("💻 Falling back to CPU ultra-fast mode...")
                    results = self.train_cpu_ultra_fast(
                        model_type=model_type,
                        epochs=min(epochs, 5)
                    )
            else:
                self.training_logs.append("💻 No GPU detected - using CPU ultra-fast training...")
                results = self.train_cpu_ultra_fast(
                    model_type=model_type,
                    epochs=min(epochs, 5)  # 5 epochs for CPU
                )
            
            # Log results
            self.training_logs.append("✅ Ultra-fast training completed!")
            self.training_logs.append(f"📊 Test Accuracy: {results['test_accuracy']:.3f}")
            self.training_logs.append(f"⏱️ Training Time: {results.get('training_duration_minutes', 'N/A')} minutes")
            self.training_logs.append(f"🎯 10-min Target: {'ACHIEVED' if results.get('target_achieved', False) else 'CLOSE'}")
            self.training_logs.append(f"🖥️ Hardware: {results.get('hardware', 'Unknown')}")
            
            # Performance summary
            if results.get('target_achieved'):
                self.training_logs.append("🎉 Training completed within 10-minute target!")
                self.training_logs.append("🔄 Model is now available for food recognition!")
            else:
                self.training_logs.append("📈 Training optimized for maximum speed")
                self.training_logs.append("✨ New model ready for testing!")
                
        except Exception as e:
            self.training_logs.append(f"❌ Training failed: {str(e)}")
            logger.error(f"Training error: {e}")
        finally:
            self.is_training = False

# ...existing code...

def main():
    """Main function for standalone execution"""
    import argparse
    
    # Configure TensorFlow early before any operations
    configure_tensorflow_early()
    
    parser = argparse.ArgumentParser(description="Food-101 Model Trainer")
    parser.add_argument("--data-dir", default="organized_food101", help="Path to organized dataset")
    parser.add_argument("--model-dir", default="models", help="Directory to save models")
    parser.add_argument("--model-type", default="EfficientNetB0", 
                       choices=['EfficientNetB0', 'EfficientNetB1', 'MobileNetV2', 'ResNet50', 'InceptionV3'],
                       help="Type of model to train")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--fine-tune", action="store_true", help="Enable fine-tuning")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model after training")
    parser.add_argument("--plot-history", action="store_true", help="Plot training history")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = Food101ModelTrainer(args.data_dir, args.model_dir)
    
    # Set configuration
    trainer.set_config(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        model_type=args.model_type
    )
    
    # Train model
    print(f"🚀 Starting training with {args.model_type}")
    results = trainer.train_model(
        model_type=args.model_type,
        epochs=args.epochs,
        fine_tune=args.fine_tune
    )
    
    print(f"✅ Training completed:")
    print(f"   - Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"   - Test Top-5 Accuracy: {results.get('test_top5_accuracy', 'N/A')}")
    
    # Plot training history
    if args.plot_history:
        trainer.plot_training_history(
            save_path=f"{args.model_dir}/training_history_{args.model_type}.png"
        )
    
    # Evaluate model
    if args.evaluate:
        print("📊 Running comprehensive evaluation...")
        eval_results = trainer.evaluate_model()
        print(f"   - Final Test Accuracy: {eval_results['test_accuracy']:.4f}")
        print(f"   - Top-5 Accuracy: {eval_results['top_5_accuracy']:.4f}")


if __name__ == "__main__":
    main()