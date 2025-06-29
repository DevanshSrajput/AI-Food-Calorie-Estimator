"""
Food-101 Dataset Downloader and Organizer
Handles downloading, extracting, and organizing the Food-101 dataset for training
"""

import os
import requests
import tarfile
import shutil
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Food101Downloader:
    """
    Comprehensive Food-101 dataset downloader and organizer
    """
    
    def __init__(self, base_dir="./"):
        self.base_dir = Path(base_dir)
        self.url = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
        self.dataset_dir = self.base_dir / "food-101"
        self.download_dir = self.base_dir / "downloads"
        self.organized_dir = self.base_dir / "organized_food101"
        
        # Create necessary directories
        self.download_dir.mkdir(exist_ok=True)
        
    def download_dataset(self, force_download=False):
        """
        Download Food-101 dataset with progress tracking
        
        Args:
            force_download (bool): Force re-download even if file exists
            
        Returns:
            Path: Path to extracted dataset
        """
        tar_path = self.download_dir / "food-101.tar.gz"
        
        # Check if already downloaded
        if tar_path.exists() and not force_download:
            logger.info("Food-101 dataset archive already exists")
        else:
            logger.info("Starting Food-101 dataset download (~5GB)")
            self._download_with_progress(tar_path)
        
        # Extract if not already extracted
        if not self.dataset_dir.exists():
            logger.info("Extracting Food-101 dataset...")
            self._extract_dataset(tar_path)
        else:
            logger.info("Food-101 dataset already extracted")
        
        return self.dataset_dir
    
    def _download_with_progress(self, tar_path):
        """Download file with progress bar"""
        try:
            response = requests.get(self.url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(tar_path, 'wb') as file, tqdm(
                desc="Downloading Food-101",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        size = file.write(chunk)
                        pbar.update(size)
            
            logger.info("Download completed successfully")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Download failed: {e}")
            raise
    
    def _extract_dataset(self, tar_path):
        """Extract the downloaded tar.gz file"""
        try:
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(path=self.base_dir)
            logger.info("Extraction completed successfully")
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            raise
    
    def organize_for_training(self, validation_split=0.2, test_from_train=False):
        """
        Organize Food-101 dataset for training with proper train/val/test splits
        
        Args:
            validation_split (float): Fraction of training data to use for validation
            test_from_train (bool): Create test set from training data instead of using official test set
            
        Returns:
            dict: Statistics about the organized dataset
        """
        logger.info("Organizing Food-101 dataset for training")
        
        if not self.dataset_dir.exists():
            raise FileNotFoundError("Food-101 dataset not found. Download first.")
        
        meta_dir = self.dataset_dir / "meta"
        images_dir = self.dataset_dir / "images"
        
        # Read official train/test splits
        with open(meta_dir / "train.txt", 'r') as f:
            train_files = [line.strip() for line in f]
        
        with open(meta_dir / "test.txt", 'r') as f:
            test_files = [line.strip() for line in f]
        
        # Get class names
        with open(meta_dir / "classes.txt", 'r') as f:
            classes = [line.strip() for line in f]
        
        logger.info(f"Found {len(classes)} classes, {len(train_files)} train files, {len(test_files)} test files")
        
        # Create organized directory structure
        splits = ['train', 'val', 'test']
        for split in splits:
            for class_name in classes:
                (self.organized_dir / split / class_name).mkdir(parents=True, exist_ok=True)
        
        # Organize files
        if test_from_train:
            # Create train/val/test from training data only
            np.random.shuffle(train_files)
            
            n_total = len(train_files)
            n_test = int(n_total * 0.15)  # 15% for test
            n_val = int(n_total * validation_split)  # 20% for validation
            n_train = n_total - n_test - n_val
            
            actual_train = train_files[:n_train]
            actual_val = train_files[n_train:n_train + n_val]
            actual_test = train_files[n_train + n_val:n_train + n_val + n_test]
            
        else:
            # Use official test set and create validation from training
            np.random.shuffle(train_files)
            
            n_val = int(len(train_files) * validation_split)
            actual_train = train_files[n_val:]
            actual_val = train_files[:n_val]
            actual_test = test_files
        
        # Copy files to organized structure
        file_splits = {
            'train': actual_train,
            'val': actual_val,
            'test': actual_test
        }
        
        stats = {}
        
        for split_name, file_list in file_splits.items():
            logger.info(f"Organizing {split_name} split ({len(file_list)} files)")
            
            split_stats = self._copy_files_to_split(
                file_list, 
                images_dir, 
                self.organized_dir / split_name
            )
            stats[split_name] = split_stats
        
        # Save organization info
        self._save_organization_info(stats, classes)
        
        logger.info("Dataset organization completed")
        return stats
    
    def _copy_files_to_split(self, file_list, source_dir, target_dir):
        """Copy files to organized split directory"""
        stats = {'total_files': 0, 'classes': {}}
        
        for file_path in tqdm(file_list, desc=f"Copying {target_dir.name} files"):
            class_name = file_path.split('/')[0]
            file_name = file_path.split('/')[-1]
            
            source_file = source_dir / f"{file_path}.jpg"
            target_file = target_dir / class_name / f"{file_name}.jpg"
            
            if source_file.exists():
                shutil.copy2(source_file, target_file)
                stats['total_files'] += 1
                
                if class_name not in stats['classes']:
                    stats['classes'][class_name] = 0
                stats['classes'][class_name] += 1
        
        return stats
    
    def _save_organization_info(self, stats, classes):
        """Save organization statistics and metadata"""
        info_dir = self.base_dir / "data"
        info_dir.mkdir(exist_ok=True)
        
        # Save detailed statistics
        with open(info_dir / "organization_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save class names
        with open(info_dir / "food101_classes.json", 'w') as f:
            json.dump(classes, f, indent=2)
        
        # Create summary statistics
        summary = {
            'total_classes': len(classes),
            'splits': {}
        }
        
        for split_name, split_stats in stats.items():
            summary['splits'][split_name] = {
                'total_files': split_stats['total_files'],
                'avg_files_per_class': split_stats['total_files'] / len(classes) if classes else 0
            }
        
        with open(info_dir / "dataset_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Organization info saved to {info_dir}")
    
    def create_nutrition_database(self):
        """Create comprehensive nutrition database for Food-101 classes"""
        logger.info("Creating Food-101 nutrition database")
        
        # Comprehensive Food-101 nutritional data (calories per 100g)
        nutrition_data = {
            'apple_pie': {'calories': 237, 'protein': 2.4, 'carbs': 34.0, 'fats': 11.0, 'fiber': 1.9},
            'baby_back_ribs': {'calories': 292, 'protein': 26.0, 'carbs': 0.0, 'fats': 20.0, 'fiber': 0.0},
            'baklava': {'calories': 307, 'protein': 4.1, 'carbs': 29.0, 'fats': 20.0, 'fiber': 1.2},
            'beef_carpaccio': {'calories': 190, 'protein': 22.0, 'carbs': 1.0, 'fats': 11.0, 'fiber': 0.0},
            'beef_tartare': {'calories': 196, 'protein': 20.0, 'carbs': 3.0, 'fats': 12.0, 'fiber': 0.0},
            'beet_salad': {'calories': 43, 'protein': 1.6, 'carbs': 10.0, 'fats': 0.2, 'fiber': 2.8},
            'beignets': {'calories': 347, 'protein': 6.0, 'carbs': 35.0, 'fats': 20.0, 'fiber': 1.5},
            'bibimbap': {'calories': 121, 'protein': 8.0, 'carbs': 15.0, 'fats': 3.5, 'fiber': 2.0},
            'bread_pudding': {'calories': 180, 'protein': 4.5, 'carbs': 31.0, 'fats': 4.5, 'fiber': 1.0},
            'breakfast_burrito': {'calories': 219, 'protein': 12.0, 'carbs': 20.0, 'fats': 11.0, 'fiber': 2.5},
            'bruschetta': {'calories': 194, 'protein': 6.0, 'carbs': 29.0, 'fats': 6.5, 'fiber': 2.0},
            'caesar_salad': {'calories': 158, 'protein': 3.0, 'carbs': 6.0, 'fats': 14.0, 'fiber': 2.0},
            'cannoli': {'calories': 297, 'protein': 6.0, 'carbs': 27.0, 'fats': 19.0, 'fiber': 1.0},
            'caprese_salad': {'calories': 166, 'protein': 11.0, 'carbs': 5.0, 'fats': 12.0, 'fiber': 1.0},
            'carrot_cake': {'calories': 415, 'protein': 4.0, 'carbs': 56.0, 'fats': 20.0, 'fiber': 2.0},
            'ceviche': {'calories': 134, 'protein': 18.0, 'carbs': 8.0, 'fats': 3.0, 'fiber': 1.0},
            'cheese_plate': {'calories': 368, 'protein': 25.0, 'carbs': 1.0, 'fats': 30.0, 'fiber': 0.0},
            'cheesecake': {'calories': 321, 'protein': 5.5, 'carbs': 26.0, 'fats': 23.0, 'fiber': 0.5},
            'chicken_curry': {'calories': 165, 'protein': 14.0, 'carbs': 7.0, 'fats': 9.0, 'fiber': 1.5},
            'chicken_quesadilla': {'calories': 234, 'protein': 15.0, 'carbs': 18.0, 'fats': 12.0, 'fiber': 1.0},
            'chicken_wings': {'calories': 203, 'protein': 30.0, 'carbs': 0.0, 'fats': 8.1, 'fiber': 0.0},
            'chocolate_cake': {'calories': 389, 'protein': 4.3, 'carbs': 56.0, 'fats': 18.0, 'fiber': 2.0},
            'chocolate_mousse': {'calories': 168, 'protein': 2.8, 'carbs': 16.0, 'fats': 11.0, 'fiber': 1.0},
            'churros': {'calories': 117, 'protein': 1.3, 'carbs': 12.0, 'fats': 7.3, 'fiber': 0.5},
            'clam_chowder': {'calories': 78, 'protein': 4.8, 'carbs': 9.8, 'fats': 2.3, 'fiber': 0.5},
            'club_sandwich': {'calories': 282, 'protein': 21.0, 'carbs': 24.0, 'fats': 12.0, 'fiber': 2.0},
            'crab_cakes': {'calories': 160, 'protein': 11.0, 'carbs': 5.0, 'fats': 11.0, 'fiber': 0.5},
            'creme_brulee': {'calories': 343, 'protein': 4.6, 'carbs': 22.0, 'fats': 27.0, 'fiber': 0.0},
            'croque_madame': {'calories': 295, 'protein': 18.0, 'carbs': 15.0, 'fats': 19.0, 'fiber': 1.0},
            'cup_cakes': {'calories': 305, 'protein': 3.8, 'carbs': 53.0, 'fats': 9.0, 'fiber': 1.0},
            'deviled_eggs': {'calories': 124, 'protein': 6.2, 'carbs': 0.6, 'fats': 11.0, 'fiber': 0.0},
            'donuts': {'calories': 452, 'protein': 4.9, 'carbs': 51.0, 'fats': 25.0, 'fiber': 1.5},
            'dumplings': {'calories': 41, 'protein': 1.7, 'carbs': 8.5, 'fats': 0.4, 'fiber': 0.5},
            'edamame': {'calories': 121, 'protein': 11.0, 'carbs': 8.9, 'fats': 5.2, 'fiber': 5.2},
            'eggs_benedict': {'calories': 230, 'protein': 12.0, 'carbs': 16.0, 'fats': 14.0, 'fiber': 1.0},
            'escargots': {'calories': 90, 'protein': 16.0, 'carbs': 2.0, 'fats': 1.4, 'fiber': 0.0},
            'falafel': {'calories': 333, 'protein': 13.0, 'carbs': 32.0, 'fats': 18.0, 'fiber': 4.9},
            'filet_mignon': {'calories': 227, 'protein': 25.0, 'carbs': 0.0, 'fats': 15.0, 'fiber': 0.0},
            'fish_and_chips': {'calories': 265, 'protein': 14.0, 'carbs': 23.0, 'fats': 14.0, 'fiber': 2.0},
            'foie_gras': {'calories': 462, 'protein': 11.0, 'carbs': 4.7, 'fats': 44.0, 'fiber': 0.0},
            'french_fries': {'calories': 365, 'protein': 4.0, 'carbs': 63.0, 'fats': 17.0, 'fiber': 3.8},
            'french_onion_soup': {'calories': 57, 'protein': 3.8, 'carbs': 8.0, 'fats': 1.7, 'fiber': 1.0},
            'french_toast': {'calories': 166, 'protein': 5.9, 'carbs': 18.0, 'fats': 7.0, 'fiber': 1.0},
            'fried_calamari': {'calories': 175, 'protein': 15.0, 'carbs': 8.0, 'fats': 9.0, 'fiber': 0.5},
            'fried_rice': {'calories': 163, 'protein': 2.9, 'carbs': 20.0, 'fats': 8.0, 'fiber': 0.5},
            'frozen_yogurt': {'calories': 127, 'protein': 3.0, 'carbs': 22.0, 'fats': 4.0, 'fiber': 0.0},
            'garlic_bread': {'calories': 300, 'protein': 8.0, 'carbs': 42.0, 'fats': 12.0, 'fiber': 2.0},
            'gnocchi': {'calories': 131, 'protein': 3.8, 'carbs': 23.0, 'fats': 2.9, 'fiber': 1.5},
            'greek_salad': {'calories': 107, 'protein': 2.8, 'carbs': 8.0, 'fats': 8.0, 'fiber': 3.0},
            'grilled_cheese_sandwich': {'calories': 291, 'protein': 12.0, 'carbs': 28.0, 'fats': 15.0, 'fiber': 2.0},
            'grilled_salmon': {'calories': 231, 'protein': 25.0, 'carbs': 0.0, 'fats': 14.0, 'fiber': 0.0},
            'guacamole': {'calories': 160, 'protein': 2.0, 'carbs': 9.0, 'fats': 15.0, 'fiber': 7.0},
            'gyoza': {'calories': 64, 'protein': 2.7, 'carbs': 6.6, 'fats': 3.0, 'fiber': 0.5},
            'hamburger': {'calories': 540, 'protein': 25.0, 'carbs': 40.0, 'fats': 31.0, 'fiber': 3.0},
            'hot_and_sour_soup': {'calories': 91, 'protein': 3.7, 'carbs': 8.0, 'fats': 5.0, 'fiber': 1.0},
            'hot_dog': {'calories': 290, 'protein': 10.0, 'carbs': 2.0, 'fats': 26.0, 'fiber': 0.0},
            'huevos_rancheros': {'calories': 153, 'protein': 8.9, 'carbs': 12.0, 'fats': 8.0, 'fiber': 2.0},
            'hummus': {'calories': 166, 'protein': 8.0, 'carbs': 14.0, 'fats': 10.0, 'fiber': 6.0},
            'ice_cream': {'calories': 207, 'protein': 3.5, 'carbs': 24.0, 'fats': 11.0, 'fiber': 0.7},
            'lasagna': {'calories': 135, 'protein': 8.1, 'carbs': 11.0, 'fats': 6.9, 'fiber': 1.0},
            'lobster_bisque': {'calories': 104, 'protein': 4.7, 'carbs': 7.0, 'fats': 6.9, 'fiber': 0.2},
            'lobster_roll_sandwich': {'calories': 436, 'protein': 18.0, 'carbs': 44.0, 'fats': 21.0, 'fiber': 2.0},
            'macaroni_and_cheese': {'calories': 164, 'protein': 6.4, 'carbs': 20.0, 'fats': 6.6, 'fiber': 1.0},
            'macarons': {'calories': 300, 'protein': 5.0, 'carbs': 50.0, 'fats': 10.0, 'fiber': 2.0},
            'miso_soup': {'calories': 40, 'protein': 2.2, 'carbs': 7.0, 'fats': 1.0, 'fiber': 1.0},
            'mussels': {'calories': 172, 'protein': 24.0, 'carbs': 7.4, 'fats': 4.6, 'fiber': 0.0},
            'nachos': {'calories': 346, 'protein': 9.0, 'carbs': 36.0, 'fats': 19.0, 'fiber': 3.0},
            'omelette': {'calories': 154, 'protein': 11.0, 'carbs': 1.0, 'fats': 12.0, 'fiber': 0.0},
            'onion_rings': {'calories': 411, 'protein': 5.9, 'carbs': 38.0, 'fats': 26.0, 'fiber': 2.5},
            'oysters': {'calories': 81, 'protein': 9.5, 'carbs': 4.7, 'fats': 2.9, 'fiber': 0.0},
            'pad_thai': {'calories': 181, 'protein': 9.0, 'carbs': 23.0, 'fats': 6.5, 'fiber': 2.0},
            'paella': {'calories': 172, 'protein': 8.0, 'carbs': 21.0, 'fats': 6.5, 'fiber': 1.5},
            'pancakes': {'calories': 227, 'protein': 6.0, 'carbs': 28.0, 'fats': 10.0, 'fiber': 1.5},
            'panna_cotta': {'calories': 185, 'protein': 2.8, 'carbs': 15.0, 'fats': 13.0, 'fiber': 0.0},
            'peking_duck': {'calories': 337, 'protein': 19.0, 'carbs': 0.0, 'fats': 28.0, 'fiber': 0.0},
            'pho': {'calories': 194, 'protein': 15.0, 'carbs': 25.0, 'fats': 3.0, 'fiber': 1.0},
            'pizza': {'calories': 266, 'protein': 11.0, 'carbs': 33.0, 'fats': 10.0, 'fiber': 2.3},
            'pork_chop': {'calories': 231, 'protein': 23.0, 'carbs': 0.0, 'fats': 15.0, 'fiber': 0.0},
            'poutine': {'calories': 740, 'protein': 37.0, 'carbs': 93.0, 'fats': 27.0, 'fiber': 6.0},
            'prime_rib': {'calories': 338, 'protein': 25.0, 'carbs': 0.0, 'fats': 26.0, 'fiber': 0.0},
            'pulled_pork_sandwich': {'calories': 415, 'protein': 29.0, 'carbs': 41.0, 'fats': 15.0, 'fiber': 2.0},
            'ramen': {'calories': 436, 'protein': 18.0, 'carbs': 54.0, 'fats': 16.0, 'fiber': 2.0},
            'ravioli': {'calories': 220, 'protein': 8.0, 'carbs': 31.0, 'fats': 7.0, 'fiber': 2.0},
            'red_velvet_cake': {'calories': 478, 'protein': 5.1, 'carbs': 73.0, 'fats': 19.0, 'fiber': 1.5},
            'risotto': {'calories': 166, 'protein': 3.0, 'carbs': 20.0, 'fats': 8.0, 'fiber': 0.5},
            'samosa': {'calories': 308, 'protein': 5.0, 'carbs': 25.0, 'fats': 21.0, 'fiber': 3.0},
            'sashimi': {'calories': 127, 'protein': 20.0, 'carbs': 0.0, 'fats': 4.4, 'fiber': 0.0},
            'scallops': {'calories': 111, 'protein': 20.0, 'carbs': 5.4, 'fats': 0.8, 'fiber': 0.0},
            'seaweed_salad': {'calories': 45, 'protein': 1.7, 'carbs': 9.0, 'fats': 0.6, 'fiber': 0.3},
            'shrimp_and_grits': {'calories': 258, 'protein': 18.0, 'carbs': 27.0, 'fats': 9.0, 'fiber': 1.5},
            'spaghetti_bolognese': {'calories': 158, 'protein': 8.2, 'carbs': 20.0, 'fats': 5.6, 'fiber': 2.0},
            'spaghetti_carbonara': {'calories': 370, 'protein': 14.0, 'carbs': 40.0, 'fats': 17.0, 'fiber': 2.0},
            'spring_rolls': {'calories': 78, 'protein': 1.8, 'carbs': 13.0, 'fats': 2.5, 'fiber': 1.5},
            'steak': {'calories': 271, 'protein': 26.0, 'carbs': 0.0, 'fats': 19.0, 'fiber': 0.0},
            'strawberry_shortcake': {'calories': 344, 'protein': 4.4, 'carbs': 55.0, 'fats': 13.0, 'fiber': 2.0},
            'sushi': {'calories': 143, 'protein': 6.0, 'carbs': 21.0, 'fats': 3.9, 'fiber': 0.2},
            'tacos': {'calories': 226, 'protein': 15.0, 'carbs': 20.0, 'fats': 12.0, 'fiber': 3.0},
            'takoyaki': {'calories': 40, 'protein': 1.9, 'carbs': 4.3, 'fats': 1.6, 'fiber': 0.1},
            'tiramisu': {'calories': 240, 'protein': 4.0, 'carbs': 29.0, 'fats': 12.0, 'fiber': 1.0},
            'tuna_tartare': {'calories': 144, 'protein': 23.0, 'carbs': 0.0, 'fats': 5.0, 'fiber': 0.0},
            'waffles': {'calories': 291, 'protein': 7.9, 'carbs': 37.0, 'fats': 13.0, 'fiber': 1.9}
        }
        
        # Convert to DataFrame
        nutrition_df = pd.DataFrame.from_dict(nutrition_data, orient='index')
        nutrition_df.reset_index(inplace=True)
        nutrition_df.rename(columns={'index': 'food_name'}, inplace=True)
        
        # Save to CSV
        data_dir = self.base_dir / "data"
        data_dir.mkdir(exist_ok=True)
        nutrition_df.to_csv(data_dir / "food101_calories.csv", index=False)
        
        logger.info(f"Nutrition database created with {len(nutrition_data)} food items")
        return nutrition_df
    
    def validate_dataset(self):
        """Validate the organized dataset"""
        if not self.organized_dir.exists():
            return {"status": "error", "message": "Organized dataset not found"}
        
        validation_results = {
            "status": "success",
            "splits": {},
            "total_images": 0,
            "total_classes": 0
        }
        
        for split in ['train', 'val', 'test']:
            split_dir = self.organized_dir / split
            if split_dir.exists():
                split_stats = self._validate_split(split_dir)
                validation_results["splits"][split] = split_stats
                validation_results["total_images"] += split_stats["total_images"]
        
        # Count unique classes across all splits
        all_classes = set()
        for split_stats in validation_results["splits"].values():
            all_classes.update(split_stats["classes"].keys())
        
        validation_results["total_classes"] = len(all_classes)
        
        return validation_results
    
    def _validate_split(self, split_dir):
        """Validate a single split directory"""
        stats = {
            "total_images": 0,
            "classes": {},
            "empty_classes": [],
            "min_images_per_class": float('inf'),
            "max_images_per_class": 0
        }
        
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
                num_images = len(image_files)
                
                stats["classes"][class_dir.name] = num_images
                stats["total_images"] += num_images
                
                if num_images == 0:
                    stats["empty_classes"].append(class_dir.name)
                else:
                    stats["min_images_per_class"] = min(stats["min_images_per_class"], num_images)
                    stats["max_images_per_class"] = max(stats["max_images_per_class"], num_images)
        
        if stats["min_images_per_class"] == float('inf'):
            stats["min_images_per_class"] = 0
        
        return stats
    
    def cleanup_downloads(self, keep_organized=True):
        """Clean up downloaded files to save space"""
        if self.download_dir.exists():
            shutil.rmtree(self.download_dir)
            logger.info("Download cache cleaned up")
        
        if not keep_organized and self.dataset_dir.exists():
            shutil.rmtree(self.dataset_dir)
            logger.info("Original dataset cleaned up")

def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Food-101 Dataset Downloader and Organizer")
    parser.add_argument("--download", action="store_true", help="Download Food-101 dataset")
    parser.add_argument("--organize", action="store_true", help="Organize dataset for training")
    parser.add_argument("--nutrition", action="store_true", help="Create nutrition database")
    parser.add_argument("--validate", action="store_true", help="Validate organized dataset")
    parser.add_argument("--cleanup", action="store_true", help="Cleanup download cache")
    parser.add_argument("--all", action="store_true", help="Run all operations")
    parser.add_argument("--base-dir", default="./", help="Base directory for dataset")
    
    args = parser.parse_args()
    
    downloader = Food101Downloader(args.base_dir)
    
    if args.all or args.download:
        print("📥 Downloading Food-101 dataset...")
        downloader.download_dataset()
    
    if args.all or args.organize:
        print("📁 Organizing dataset for training...")
        stats = downloader.organize_for_training()
        print(f"✅ Organized {sum(s['total_files'] for s in stats.values())} files")
    
    if args.all or args.nutrition:
        print("📊 Creating nutrition database...")
        nutrition_df = downloader.create_nutrition_database()
        print(f"✅ Created nutrition database with {len(nutrition_df)} entries")
    
    if args.all or args.validate:
        print("🔍 Validating dataset...")
        validation = downloader.validate_dataset()
        if validation["status"] == "success":
            print(f"✅ Dataset validation successful:")
            print(f"   - Total classes: {validation['total_classes']}")
            print(f"   - Total images: {validation['total_images']}")
            if "splits" in validation and isinstance(validation["splits"], dict):
                for split, stats in validation["splits"].items():
                    print(f"   - {split}: {stats['total_images']} images")
        else:
            print(f"❌ Dataset validation failed: {validation['message']}")
    
    if args.cleanup:
        print("🧹 Cleaning up downloads...")
        downloader.cleanup_downloads()
        print("✅ Cleanup completed")

if __name__ == "__main__":
    main()