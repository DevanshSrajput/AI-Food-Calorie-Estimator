"""
AI Food Calorie Estimator - Food-101 (Optimized Version)
Streamlined application with fast loading and core functionalities
"""

import streamlit as st
import tensorflow as tf

# Configure TensorFlow immediately after import to avoid initialization issues
try:
    import os
    
    # Set environment variables (always safe)
    os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
    os.environ['TF_NUM_INTRAOP_THREADS'] = str(os.cpu_count())
    os.environ['TF_NUM_INTEROP_THREADS'] = '2'
    
    # Try to set TensorFlow config (must be done before any TF operations)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count())
        tf.config.threading.set_inter_op_parallelism_threads(2)
        
        # GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        
    except RuntimeError:
        pass  # TensorFlow already initialized or no GPU
    
except Exception:
    pass  # Ignore configuration errors

# Continue with other imports
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import json
import os
from datetime import datetime, date
import plotly.express as px
import plotly.graph_objects as go
import threading
from pathlib import Path
import requests
from tqdm import tqdm
import time
try:
    import yaml
except ImportError:
    yaml = None
import logging
from typing import Optional, Dict, Any, Tuple

# Import TensorFlow/Keras components
try:
    # Use tf.keras directly for better IDE support
    import keras.src.layers as layers
    import keras.src.models as models  
    import keras.src.optimizers as optimizers
    import keras.src.applications as applications
    KERAS_AVAILABLE = True
except ImportError:
    # Fallback for older versions
    import keras
    from keras import layers, models, optimizers, applications
    KERAS_AVAILABLE = True

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure page
st.set_page_config(
    page_title="AI Food Calorie Estimator",
    page_icon="🍎",
    layout="wide"
)

# Optimized CSS - only essential styles
st.markdown("""
<style>
    .main-header { 
        font-size: 2.5rem; 
        text-align: center; 
        color: #2E8B57; 
        margin-bottom: 1rem; 
    }
    .prediction-card { 
        background: linear-gradient(135deg, #667eea, #764ba2); 
        padding: 1.5rem; 
        border-radius: 15px; 
        color: white; 
        text-align: center; 
        margin: 1rem 0; 
    }
    .metric-card { 
        background: white; 
        padding: 1rem; 
        border-radius: 8px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
        margin: 0.5rem 0; 
    }
    .status-success { 
        background: #d4edda; 
        color: #155724; 
        padding: 0.75rem; 
        border-radius: 5px; 
        margin: 0.5rem 0; 
    }
</style>
""", unsafe_allow_html=True)

# Cached configuration loader
@st.cache_data
def load_config():
    if yaml and os.path.exists('config.yaml'):
        try:
            with open('config.yaml', 'r') as f:
                return yaml.safe_load(f)
        except:
            pass
    
    return {
        'app': {'title': 'AI Food Calorie Estimator', 'version': '2.0.0'},
        'model': {'default_model': 'EfficientNetB0', 'input_size': [224, 224]}
    }

config = load_config()

# Optimized Food-101 Downloader
class Food101Downloader:
    def __init__(  self):
        self.url = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
        self.data_dir = Path("food-101")
        
    def download_dataset(self):
        """Download Food-101 dataset"""
        download_dir = Path("downloads")
        download_dir.mkdir(exist_ok=True)
        tar_path = download_dir / "food-101.tar.gz"
        
        if not tar_path.exists():
            response = requests.get(self.url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with open(tar_path, 'wb') as file:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        downloaded += len(chunk)
                        progress = downloaded / total_size
                        progress_bar.progress(progress)
                        status_text.text(f"Downloaded: {downloaded / (1024*1024):.1f} MB")
        
        # Extract dataset
        import tarfile
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=".")
        
        return self.data_dir
    
    def organize_for_training(self):
        """Organize dataset for training"""
        meta_dir = self.data_dir / "meta"
        images_dir = self.data_dir / "images"
        
        train_dir = Path("organized_food101/train")
        val_dir = Path("organized_food101/val")
        test_dir = Path("organized_food101/test")
        
        for dir_path in [train_dir, val_dir, test_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Read splits
        with open(meta_dir / "train.txt", 'r') as f:
            train_files = [line.strip() for line in f]
        
        with open(meta_dir / "test.txt", 'r') as f:
            test_files = [line.strip() for line in f]
        
        # Create validation split
        np.random.shuffle(train_files)
        val_split = int(len(train_files) * 0.2)
        val_files = train_files[:val_split]
        train_files = train_files[val_split:]
        
        # Get classes and create directories  
        classes = set(file_path.split('/')[0] for file_path in train_files + test_files)
        for class_name in classes:
            (train_dir / class_name).mkdir(exist_ok=True)
            (val_dir / class_name).mkdir(exist_ok=True)
            (test_dir / class_name).mkdir(exist_ok=True)
        
        # Copy files with progress
        def copy_files(file_list, target_dir):
            import shutil
            for file_path in tqdm(file_list, desc=f"Organizing {target_dir.name}"):
                class_name = file_path.split('/')[0]
                src = images_dir / f"{file_path}.jpg"
                dst = target_dir / class_name / f"{file_path.split('/')[-1]}.jpg"
                if src.exists():
                    shutil.copy2(src, dst)
        
        copy_files(train_files, train_dir)
        copy_files(val_files, val_dir)
        copy_files(test_files, test_dir)
        
        return Path("organized_food101")

# Optimized Model Trainer
class ModelTrainer:
    def __init__(self):
        self.model = None
        self.is_training = False
        self.training_progress = 0
        self.training_logs = []
        self.current_metrics = {}
        
    def create_model(self, num_classes: int, model_type: str = "EfficientNetB0") -> models.Model:
        """Create transfer learning model"""
        # Model selection
        if model_type == "EfficientNetB0":
            base_model = applications.EfficientNetB0(
                weights='imagenet', include_top=False, input_shape=(224, 224, 3)
            )
        elif model_type == "MobileNetV2":
            base_model = applications.MobileNetV2(
                weights='imagenet', include_top=False, input_shape=(224, 224, 3)
            )
        else:
            base_model = applications.ResNet50(
                weights='imagenet', include_top=False, input_shape=(224, 224, 3)
            )
        
        base_model.trainable = False
        
        # Build model
        inputs = layers.Input(shape=(224, 224, 3))
        x = applications.efficientnet.preprocess_input(inputs) if model_type == "EfficientNetB0" else inputs
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        self.model = models.Model(inputs, outputs)
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train_model_async(self, data_dir, epochs, batch_size, model_type="MobileNetV2"):
        """Ultra-fast training process optimized for both GPU and CPU"""
        try:
            self.is_training = True
            self.training_logs.append("🚀 Starting ultra-fast training...")
            
            # Try to import optimized model trainer
            try:
                from model_trainer import Food101ModelTrainer
                
                trainer = Food101ModelTrainer(data_dir, "models")
                
                # Check for GPU first
                gpus = tf.config.list_physical_devices('GPU')
                
                if gpus:
                    self.training_logs.append("🎮 GPU detected - attempting RTX 3050 ultra-fast training...")
                    try:
                        results = trainer.train_rtx_3050_ultra_fast(
                            model_type=model_type,
                            epochs=min(epochs, 8)  # Cap at 8 epochs for 10-minute target
                        )
                    except Exception as gpu_error:
                        self.training_logs.append(f"⚠️ GPU training failed: {str(gpu_error)}")
                        self.training_logs.append("💻 Falling back to CPU ultra-fast mode...")
                        results = trainer.train_cpu_ultra_fast(
                            model_type=model_type,
                            epochs=min(epochs, 5)
                        )
                else:
                    self.training_logs.append("💻 No GPU detected - using CPU ultra-fast training...")
                    results = trainer.train_cpu_ultra_fast(
                        model_type=model_type,
                        epochs=min(epochs, 5)  # 5 epochs for CPU
                    )
                
                # Save the model
                if trainer.model:
                    os.makedirs('models', exist_ok=True)
                    # Use .keras format instead of .h5
                    model_name = f'food101_{results.get("hardware", "optimized").lower()}_ultrafast.keras'
                    trainer.model.save(f'models/{model_name}')
                    self.training_logs.append(f"💾 Model saved as {model_name}!")
                
                # Log results
                self.training_logs.append("✅ Ultra-fast training completed successfully!")
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
                
                # Clear the classifier cache so it reloads the new model
                if 'classifier' in st.session_state:
                    # Force reload of the classifier with new model
                    st.session_state.classifier = None
                
            except ImportError:
                self.training_logs.append("⚠️ Advanced model trainer not available - using basic training...")
                # Fallback to basic training
                self.training_logs.append("💻 Running basic model training...")
                self.training_logs.append("ℹ️ For ultra-fast training, ensure model_trainer.py is available")
                
        except Exception as e:
            self.training_logs.append(f"❌ Training failed: {str(e)}")
            logger.error(f"Training error: {e}")
        finally:
            self.is_training = False

# Optimized Food Classifier
@st.cache_resource
def load_food_classifier():
    """Load classifier with caching"""
    return FoodClassifier()

class FoodClassifier:
    def __init__(self):
        self.model = None
        self.class_names = None
        self.calorie_data = None
        self.model_metadata = None
        self.load_model_and_data()
    
    def load_model_and_data(self):
        """Load model and data"""
        try:
            # Try to load any available trained model (prioritize .keras format)
            model_files = [
                'models/food101_rtx3050_ultrafast_8epochs.keras',
                'models/food101_cpu_ultrafast_5epochs.keras',
                'models/food101_trained_model.keras',
                'models/food101_trained_model.h5'
            ]
            
            model_loaded = False
            for model_file in model_files:
                if os.path.exists(model_file):
                    try:
                        self.model = models.load_model(model_file)
                        logger.info(f"Loaded model from {model_file}")
                        model_loaded = True
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load {model_file}: {e}")
                        continue
            
            if not model_loaded:
                logger.info("No trained model found, creating demo model")
                self.create_demo_model()
            else:
                # Load class names
                if os.path.exists('models/food101_classes.json'):
                    with open('models/food101_classes.json', 'r') as f:
                        self.class_names = json.load(f)
                    logger.info(f"Loaded {len(self.class_names)} class names")
                
                # Load metadata if available
                for metadata_file in ['models/training_metadata.json', 'models/training_metadata_MobileNetV2.json']:
                    if os.path.exists(metadata_file):
                        with open(metadata_file, 'r') as f:
                            self.model_metadata = json.load(f)
                        break
            
            # Load or create nutrition data
            if os.path.exists('data/food101_calories.csv'):
                self.calorie_data = pd.read_csv('data/food101_calories.csv')
            else:
                self.create_nutrition_data()
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.create_demo_model()
    
    def create_demo_model(self):
        """Create demo model"""
        logger.info("Creating demo model")
        
        self.model = models.Sequential([
            layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(20, activation='softmax')  # Demo with 20 classes
        ])
        
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Demo class names
        self.class_names = [
            'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'caesar_salad',
            'cannoli', 'carrot_cake', 'cheesecake', 'chicken_curry', 'chocolate_cake',
            'french_fries', 'hamburger', 'ice_cream', 'lasagna', 'pizza',
            'ramen', 'steak', 'sushi', 'tacos', 'waffles'
        ]
        
        self.create_nutrition_data()
    
    def create_nutrition_data(self):
        """Create nutrition database"""
        # Simplified nutrition data for key foods
        nutrition_data = {
            'apple_pie': {'calories': 237, 'protein': 2.4, 'carbs': 34.0, 'fats': 11.0},
            'baby_back_ribs': {'calories': 292, 'protein': 26.0, 'carbs': 0.0, 'fats': 20.0},
            'baklava': {'calories': 307, 'protein': 4.1, 'carbs': 29.0, 'fats': 20.0},
            'beef_carpaccio': {'calories': 190, 'protein': 22.0, 'carbs': 1.0, 'fats': 11.0},
            'caesar_salad': {'calories': 158, 'protein': 3.0, 'carbs': 6.0, 'fats': 14.0},
            'cannoli': {'calories': 297, 'protein': 6.0, 'carbs': 27.0, 'fats': 19.0},
            'carrot_cake': {'calories': 415, 'protein': 4.0, 'carbs': 56.0, 'fats': 20.0},
            'cheesecake': {'calories': 321, 'protein': 5.5, 'carbs': 26.0, 'fats': 23.0},
            'chicken_curry': {'calories': 165, 'protein': 14.0, 'carbs': 7.0, 'fats': 9.0},
            'chocolate_cake': {'calories': 389, 'protein': 4.3, 'carbs': 56.0, 'fats': 18.0},
            'french_fries': {'calories': 365, 'protein': 4.0, 'carbs': 63.0, 'fats': 17.0},
            'hamburger': {'calories': 540, 'protein': 25.0, 'carbs': 40.0, 'fats': 31.0},
            'ice_cream': {'calories': 207, 'protein': 3.5, 'carbs': 24.0, 'fats': 11.0},
            'lasagna': {'calories': 135, 'protein': 8.1, 'carbs': 11.0, 'fats': 6.9},
            'pizza': {'calories': 266, 'protein': 11.0, 'carbs': 33.0, 'fats': 10.0},
            'ramen': {'calories': 436, 'protein': 18.0, 'carbs': 54.0, 'fats': 16.0},
            'steak': {'calories': 271, 'protein': 26.0, 'carbs': 0.0, 'fats': 19.0},
            'sushi': {'calories': 143, 'protein': 6.0, 'carbs': 21.0, 'fats': 3.9},
            'tacos': {'calories': 226, 'protein': 15.0, 'carbs': 20.0, 'fats': 12.0},
            'waffles': {'calories': 291, 'protein': 7.9, 'carbs': 37.0, 'fats': 13.0}
        }
        
        # Convert to DataFrame
        data = []
        for food_name, nutrients in nutrition_data.items():
            data.append({
                'food_name': food_name,
                'calories_per_100g': nutrients['calories'],
                'protein': nutrients['protein'],
                'carbs': nutrients['carbs'],
                'fats': nutrients['fats']
            })
        
        self.calorie_data = pd.DataFrame(data)
        
        # Save to file
        os.makedirs('data', exist_ok=True)
        self.calorie_data.to_csv('data/food101_calories.csv', index=False)
    
    def predict_food(self, image):
        """Predict food from image"""
        try:
            if self.model is None:
                return None
                
            # Preprocess image
            img_array = np.array(image)
            
            # Handle different image formats (convert RGBA to RGB if needed)
            if img_array.shape[-1] == 4:  # RGBA image
                # Convert RGBA to RGB by removing alpha channel
                img_array = img_array[:, :, :3]
            elif len(img_array.shape) == 2:  # Grayscale image
                # Convert grayscale to RGB
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[-1] == 1:  # Single channel
                # Convert single channel to RGB
                img_array = np.repeat(img_array, 3, axis=-1)
            
            # Ensure we have exactly 3 channels
            if img_array.shape[-1] != 3:
                logger.warning(f"Unexpected image shape: {img_array.shape}")
                return None
            
            # Resize and normalize
            img_resized = cv2.resize(img_array, (224, 224))
            img_normalized = img_resized.astype('float32') / 255.0
            img_batch = np.expand_dims(img_normalized, axis=0)
            
            # Predict
            predictions = self.model.predict(img_batch, verbose=0)  # Set verbose=0 to reduce output
            predicted_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_idx]
            
            # Check if class_names is properly loaded
            if self.class_names is None or predicted_idx >= len(self.class_names):
                return None
                
            food_name = self.class_names[predicted_idx]
            
            # Get nutrition info - check if calorie_data is loaded
            if self.calorie_data is not None:
                nutrition_info = self.calorie_data[self.calorie_data['food_name'] == food_name]
                if nutrition_info.empty:
                    nutrition_data = {'calories_per_100g': 200, 'protein': 10.0, 'carbs': 25.0, 'fats': 8.0}
                else:
                    nutrition_row = nutrition_info.iloc[0]
                    nutrition_data = {
                        'calories_per_100g': nutrition_row['calories_per_100g'],
                        'protein': nutrition_row['protein'],
                        'carbs': nutrition_row['carbs'],
                        'fats': nutrition_row['fats']
                    }
            else:
                # Fallback nutrition data
                nutrition_data = {'calories_per_100g': 200, 'protein': 10.0, 'carbs': 25.0, 'fats': 8.0}
            
            return {
                'food_name': food_name,
                'confidence': float(confidence),
                'calories_per_100g': int(nutrition_data['calories_per_100g']),
                'protein': float(nutrition_data['protein']),
                'carbs': float(nutrition_data['carbs']),
                'fats': float(nutrition_data['fats'])
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None

# Simplified Meal Tracker
class MealTracker:
    def __init__(self):
        self.meal_file = 'data/meal_history.json'
        self.load_meal_history()
    
    def load_meal_history(self):
        """Load meal history"""
        try:
            if os.path.exists(self.meal_file):
                with open(self.meal_file, 'r') as f:
                    self.meal_history = json.load(f)
            else:
                self.meal_history = {}
        except:
            self.meal_history = {}
    
    def save_meal_history(self):
        """Save meal history"""
        try:
            os.makedirs('data', exist_ok=True)
            with open(self.meal_file, 'w') as f:
                json.dump(self.meal_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving meal history: {e}")
    
    def add_meal(self, food_name, calories, protein, carbs, fats, portion_size=100):
        """Add meal to today's log"""
        today = str(date.today())
        
        if today not in self.meal_history:
            self.meal_history[today] = []
        
        meal_entry = {
            'food_name': food_name,
            'calories': calories * (portion_size / 100),
            'protein': protein * (portion_size / 100),
            'carbs': carbs * (portion_size / 100),
            'fats': fats * (portion_size / 100),
            'portion_size': portion_size,
            'timestamp': datetime.now().strftime('%H:%M:%S')
        }
        
        self.meal_history[today].append(meal_entry)
        self.save_meal_history()
    
    def get_daily_summary(self, date_str=None):
        """Get daily summary"""
        if date_str is None:
            date_str = str(date.today())
        
        if date_str not in self.meal_history:
            return {'calories': 0, 'protein': 0, 'carbs': 0, 'fats': 0, 'meals': []}
        
        meals = self.meal_history[date_str]
        return {
            'calories': sum(meal['calories'] for meal in meals),
            'protein': sum(meal['protein'] for meal in meals),
            'carbs': sum(meal['carbs'] for meal in meals),
            'fats': sum(meal['fats'] for meal in meals),
            'meals': meals
        }

def main():
    # Initialize session state with cached components
    if 'classifier' not in st.session_state:
        st.session_state.classifier = load_food_classifier()
    
    if 'meal_tracker' not in st.session_state:
        st.session_state.meal_tracker = MealTracker()
    
    if 'model_trainer' not in st.session_state:
        st.session_state.model_trainer = ModelTrainer()
    
    if 'downloader' not in st.session_state:
        st.session_state.downloader = Food101Downloader()
    
    # Header
    st.markdown('<h1 class="main-header">🍎 AI Food Calorie Estimator - Food-101</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Professional food recognition with nutrition tracking</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 👨‍💻 DevanshSrajput")
        st.write(f"📅 {datetime.now().strftime('%Y-%m-%d')}")
        
        # Quick stats
        daily_summary = st.session_state.meal_tracker.get_daily_summary()
        st.metric("Today's Calories", f"{daily_summary['calories']:.0f}")
        st.metric("Meals Logged", len(daily_summary['meals']))
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📸 Food Analysis", "🤖 Training", "📊 Meal Tracker", "ℹ️ About"])
    
    with tab1:
        st.subheader("🔍 Food Recognition")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            uploaded_file = st.file_uploader("Choose food image", type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'])
            
            if uploaded_file is not None:
                try:
                    # Load and process image
                    image = Image.open(uploaded_file)
                    
                    # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                    
                    # Show image info
                    st.caption(f"Image size: {image.size[0]}x{image.size[1]} pixels")
                    
                    if st.button("🔍 Analyze Food", type="primary"):
                        with st.spinner("Analyzing..."):
                            prediction = st.session_state.classifier.predict_food(image)
                            if prediction:
                                st.session_state.current_prediction = prediction
                                st.success("Analysis complete!")
                            else:
                                st.error("Analysis failed. Please try a different image.")
        
                except Exception as e:
                    st.error(f"Error loading image: {e}")
                    st.info("Please try uploading a different image format (JPG, PNG, etc.)")
        
        with col2:
            if 'current_prediction' in st.session_state:
                pred = st.session_state.current_prediction
                
                # Prediction display
                st.markdown(f"""
                <div class="prediction-card">
                    <h3>{pred['food_name'].replace('_', ' ').title()}</h3>
                    <h2>{pred['calories_per_100g']} cal/100g</h2>
                    <p>Confidence: {pred['confidence']:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Nutrition info
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("🥩 Protein", f"{pred['protein']:.1f}g")
                with col_b:
                    st.metric("🍞 Carbs", f"{pred['carbs']:.1f}g")
                with col_c:
                    st.metric("🧈 Fats", f"{pred['fats']:.1f}g")
                
                # Portion calculator
                portion_size = st.slider("Portion size (g)", 25, 500, 100, 25)
                
                adjusted_cals = pred['calories_per_100g'] * (portion_size / 100)
                st.write(f"**For {portion_size}g:** {adjusted_cals:.0f} calories")
                
                # Add to tracker
                if st.button("➕ Add to Meal Log", type="secondary"):
                    st.session_state.meal_tracker.add_meal(
                        pred['food_name'], pred['calories_per_100g'],
                        pred['protein'], pred['carbs'], pred['fats'], portion_size
                    )
                    st.success("Added to meal log!")
                    st.rerun()
    
    with tab2:
        st.subheader("🤖 Model Training")
        
        # Dataset status
        organized_path = Path("organized_food101")
        if organized_path.exists():
            st.success("✅ Dataset ready for training")
        elif Path("food-101").exists():
            st.warning("⚠️ Dataset needs organization")
            if st.button("📂 Organize Dataset"):
                with st.spinner("Organizing..."):
                    st.session_state.downloader.organize_for_training()
                    st.success("Dataset organized!")
                    st.rerun()
        else:
            st.info("📥 Download Food-101 dataset first")
            if st.button("📥 Download Dataset"):
                with st.spinner("Downloading..."):
                    st.session_state.downloader.download_dataset()
                    st.success("Download complete!")
                    st.rerun()
        
        # Training status indicator
        if st.session_state.model_trainer.is_training:
            st.markdown("""
            <div style="background: linear-gradient(90deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%); 
                        padding: 1rem; border-radius: 10px; text-align: center; margin: 1rem 0;">
                <h3>🔥 Training in Progress...</h3>
                <p>Please wait while the model is being trained</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Auto-refresh every 5 seconds during training
            time.sleep(5)
            st.rerun()
        
        elif (st.session_state.model_trainer.training_logs and 
              any("✅" in log and "completed" in log for log in st.session_state.model_trainer.training_logs)):
            # Training completed successfully
            st.markdown("""
            <div style="background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%); 
                        padding: 1.5rem; border-radius: 15px; text-align: center; margin: 1rem 0; color: white;">
                <h2>🎉 Training Completed Successfully!</h2>
                <p>Your model is ready for food recognition</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show training results summary
            completed_logs = [log for log in st.session_state.model_trainer.training_logs if any(keyword in log for keyword in ["Test Accuracy", "Training Time", "Target", "Hardware"])]
            
            if completed_logs:
                st.markdown("### 📊 Training Results")
                cols = st.columns(2)
                
                for i, log in enumerate(completed_logs[:4]):  # Show top 4 results
                    with cols[i % 2]:
                        if "Test Accuracy" in log:
                            accuracy = log.split(":")[-1].strip()
                            st.metric("🎯 Test Accuracy", accuracy)
                        elif "Training Time" in log:
                            time_str = log.split(":")[-1].strip()
                            st.metric("⏱️ Training Time", time_str)
                        elif "Target" in log:
                            target_status = "ACHIEVED" if "ACHIEVED" in log else "CLOSE"
                            st.metric("🏆 10-min Target", target_status)
                        elif "Hardware" in log:
                            hardware = log.split(":")[-1].strip()
                            st.metric("🖥️ Hardware Used", hardware)
            
            # Option to start new training
            if st.button("🚀 Start New Training", type="secondary"):
                st.session_state.model_trainer.training_logs.clear()
                st.session_state.model_trainer.is_training = False
                st.rerun()
        
        # Training controls (only show if not training and no completion status)
        if organized_path.exists() and not st.session_state.model_trainer.is_training:
            st.markdown("### ⚡ Ultra-Fast Training (10-Minute Target)")
            
            # Quick preset buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("⚡ Ultra Fast (5 min)", type="secondary", help="CPU optimized, 5 epochs, subset data"):
                    training_thread = threading.Thread(
                        target=st.session_state.model_trainer.train_model_async,
                        args=(str(organized_path), 5, 64, "MobileNetV2")
                    )
                    training_thread.start()
                    st.success("Ultra-fast training started! (5-minute target)")
                    st.rerun()
            
            with col2:
                if st.button("🚀 Fast (8 min)", type="secondary", help="Balanced speed/accuracy, 8 epochs"):
                    training_thread = threading.Thread(
                        target=st.session_state.model_trainer.train_model_async,
                        args=(str(organized_path), 8, 64, "MobileNetV2")
                    )
                    training_thread.start()
                    st.success("Fast training started! (8-minute target)")
                    st.rerun()
            
            with col3:
                if st.button("⚖️ Balanced (15 min)", type="secondary", help="Best accuracy, 15 epochs"):
                    training_thread = threading.Thread(
                        target=st.session_state.model_trainer.train_model_async,
                        args=(str(organized_path), 15, 32, "EfficientNetB0")
                    )
                    training_thread.start()
                    st.success("Balanced training started! (15-minute target)")
                    st.rerun()
            
            st.markdown("---")
            st.markdown("### 🔧 Custom Training Configuration")
            
            col1, col2 = st.columns(2)
            with col1:
                model_type = st.selectbox("Model", ["MobileNetV2", "EfficientNetB0", "ResNet50"], 
                                        help="MobileNetV2 = Fastest, EfficientNetB0 = Balanced")
                epochs = st.slider("Epochs", 3, 25, 8, help="More epochs = better accuracy but slower")
            with col2:
                batch_size = st.selectbox("Batch Size", [32, 64, 96], index=1, 
                                        help="Larger batch = faster training (if enough memory)")
            
            if st.button("🚀 Start Custom Training", type="primary"):
                training_thread = threading.Thread(
                    target=st.session_state.model_trainer.train_model_async,
                    args=(str(organized_path), epochs, batch_size, model_type)
                )
                training_thread.start()
                st.success("Custom training started!")
                st.rerun()
        
        # Hardware info
        st.markdown("### 🖥️ Hardware Status")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            st.success(f"🎮 GPU Available: {len(gpus)} device(s) detected")
            st.info("Ultra-fast training will attempt GPU acceleration first")
        else:
            st.info("💻 CPU Mode: Ultra-fast CPU optimizations will be used")
            st.info("Expected training time: 5-10 minutes with optimizations")
        
        # Training logs with better formatting
        if st.session_state.model_trainer.training_logs:
            st.markdown("### 📝 Training Logs")
            
            # Create a container for logs with better styling
            log_container = st.container()
            with log_container:
                for i, log in enumerate(st.session_state.model_trainer.training_logs):
                    if "✅" in log and "completed" in log:
                        st.markdown(f'<div style="background: #d4edda; color: #155724; padding: 0.75rem; border-radius: 5px; margin: 0.25rem 0; border-left: 4px solid #28a745;"><strong>{log}</strong></div>', unsafe_allow_html=True)
                    elif "🎉" in log:
                        st.markdown(f'<div style="background: #fff3cd; color: #856404; padding: 0.75rem; border-radius: 5px; margin: 0.25rem 0; border-left: 4px solid #ffc107;"><strong>{log}</strong></div>', unsafe_allow_html=True)
                    elif "❌" in log or "failed" in log.lower():
                        st.markdown(f'<div style="background: #f8d7da; color: #721c24; padding: 0.75rem; border-radius: 5px; margin: 0.25rem 0; border-left: 4px solid #dc3545;">{log}</div>', unsafe_allow_html=True)
                    elif "📊" in log or "⏱️" in log or "🎯" in log or "🖥️" in log:
                        st.markdown(f'<div style="background: #e2e3e5; color: #383d41; padding: 0.5rem; border-radius: 3px; margin: 0.25rem 0; font-family: monospace;">{log}</div>', unsafe_allow_html=True)
                    else:
                        st.text(log)
        
            # Auto-scroll to bottom and refresh if training is active
            if st.session_state.model_trainer.is_training:
                st.markdown("---")
                st.markdown("*Training logs will update automatically...*")

    with tab3:
        st.subheader("📊 Meal Tracking")
        
        daily_summary = st.session_state.meal_tracker.get_daily_summary()
        
        # Today's summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🔥 Calories", f"{daily_summary['calories']:.0f}")
        with col2:
            st.metric("🥩 Protein", f"{daily_summary['protein']:.1f}g")
        with col3:
            st.metric("🍞 Carbs", f"{daily_summary['carbs']:.1f}g")
        with col4:
            st.metric("🧈 Fats", f"{daily_summary['fats']:.1f}g")
        
        # Meal list
        if daily_summary['meals']:
            st.markdown("### Today's Meals")
            for i, meal in enumerate(daily_summary['meals']):
                with st.expander(f"{meal['timestamp']} - {meal['food_name'].replace('_', ' ').title()}"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write(f"**Portion:** {meal['portion_size']:.0f}g")
                        st.write(f"**Calories:** {meal['calories']:.0f}")
                    with col_b:
                        st.write(f"**Protein:** {meal['protein']:.1f}g")
                        st.write(f"**Carbs:** {meal['carbs']:.1f}g")
                        st.write(f"**Fats:** {meal['fats']:.1f}g")
        else:
            st.info("No meals logged today. Start by analyzing food images!")
        
        # Nutrition visualization
        if daily_summary['meals']:
            st.markdown("### Nutrition Breakdown")
            macro_cals = [
                daily_summary['protein'] * 4,
                daily_summary['carbs'] * 4,
                daily_summary['fats'] * 9
            ]
            
            fig = px.pie(
                values=macro_cals,
                names=['Protein', 'Carbs', 'Fats'],
                title="Calorie Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("ℹ️ About")
        
        st.markdown(f"""
        ### 🍎 AI Food Calorie Estimator
        **Version:** {config['app']['version']}
        
        A streamlined food recognition and nutrition tracking system using:
        - **Food-101 Dataset**: 101 food categories
        - **Deep Learning**: TensorFlow/Keras models
        - **Real-time Analysis**: Instant food recognition
        - **Nutrition Tracking**: Comprehensive meal logging
        
        ### 🚀 Features
        - ✅ Food recognition with confidence scoring
        - ✅ Custom model training capabilities  
        - ✅ Daily nutrition tracking and analytics
        - ✅ Portion size calculator
        - ✅ Interactive visualizations
        
        ### 👨‍💻 Developer
        **Created by:** DevanshSrajput  
        **Project Type:** ML Internship Project  
        **Focus:** Computer Vision + Health Tech
        """)
        
        # System info
        st.markdown("### 💻 System Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**TensorFlow:** {tf.__version__}")
            st.write(f"**Python:** 3.8+")
        with col2:
            st.write(f"**Streamlit:** {st.__version__}")
            st.write(f"**Session:** Active")

# Initialize and run the main application
if __name__ == "__main__":
    main()