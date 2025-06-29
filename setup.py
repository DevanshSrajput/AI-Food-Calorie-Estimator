"""
Setup script for AI Food Calorie Estimator - Food-101
Comprehensive setup and installation management
"""

import os
import sys
import subprocess
import platform
import pkg_resources
from pathlib import Path
from setuptools import setup, find_packages
import json
from datetime import datetime

# Project metadata
PROJECT_NAME = "ai-food-calorie-estimator"
VERSION = "2.0.0"
AUTHOR = "DevanshSrajput"
AUTHOR_EMAIL = "devansh.srajput@example.com"
DESCRIPTION = "AI-powered food recognition and calorie estimation system using Food-101 dataset"
LONG_DESCRIPTION = """
# 🍎 AI Food Calorie Estimator - Food-101

A comprehensive computer vision system that identifies food items from images and provides detailed nutritional information. Built with TensorFlow and the Food-101 dataset, featuring custom model training, real-time nutrition tracking, and advanced analytics.

## Features
- 🔍 **Food Recognition**: 101 food categories with high accuracy
- 🤖 **Custom Training**: Train your own models with various architectures
- 📊 **Nutrition Tracking**: Comprehensive calorie and macro tracking
- 📈 **Analytics**: Interactive charts and insights
- 💾 **Data Management**: Export and backup functionality

## Technology Stack
- **ML Framework**: TensorFlow 2.13+
- **Frontend**: Streamlit
- **Computer Vision**: OpenCV
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly

Perfect for machine learning projects, health-tech applications, and educational purposes.
"""

URL = "https://github.com/DevanshSrajput/ai-food-calorie-estimator"
PROJECT_URLS = {
    "Bug Reports": "https://github.com/DevanshSrajput/ai-food-calorie-estimator/issues",
    "Source": "https://github.com/DevanshSrajput/ai-food-calorie-estimator",
    "Documentation": "https://github.com/DevanshSrajput/ai-food-calorie-estimator/blob/main/README.md"
}

# Python version requirement
PYTHON_REQUIRES = ">=3.8"

# Core dependencies
INSTALL_REQUIRES = [
    "streamlit>=1.28.0",
    "tensorflow>=2.13.0",
    "opencv-python>=4.8.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "Pillow>=10.0.0",
    "plotly>=5.15.0",
    "scikit-learn>=1.3.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "requests>=2.31.0",
    "tqdm>=4.65.0",
    "PyYAML>=6.0",
    "psutil>=5.9.0"
]

# Optional dependencies for development
EXTRAS_REQUIRE = {
    "dev": [
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "black>=23.7.0",
        "flake8>=6.0.0",
        "mypy>=1.5.0",
        "pre-commit>=3.3.0",
        "jupyterlab>=4.0.0",
        "notebook>=7.0.0"
    ],
    "gpu": [
        "tensorflow-gpu>=2.13.0"
    ],
    "accelerated": [
        "tensorflow-metal>=1.0.0; platform_system=='Darwin'",
        "tensorrt>=8.6.0; platform_system=='Linux'"
    ],
    "export": [
        "tensorflowjs>=4.10.0",
        "tensorflow-lite-runtime>=2.13.0"
    ]
}

# All optional dependencies
EXTRAS_REQUIRE["all"] = list(set(sum(EXTRAS_REQUIRE.values(), [])))

# Classifiers
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Healthcare Industry",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Image Recognition",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Internet :: WWW/HTTP :: Dynamic Content",
    "Topic :: Multimedia :: Graphics :: Graphics Conversion",
    "Typing :: Typed"
]

# Keywords
KEYWORDS = [
    "food-recognition", "computer-vision", "machine-learning", "tensorflow",
    "nutrition", "calorie-estimation", "streamlit", "food-101", "deep-learning",
    "health-tech", "opencv", "image-classification", "transfer-learning"
]

class SetupManager:
    """Comprehensive setup and environment management"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        self.platform_info = platform.platform()
        
    def check_python_version(self):
        """Check if Python version meets requirements"""
        required_version = tuple(map(int, PYTHON_REQUIRES.strip(">=").split(".")))
        current_version = sys.version_info[:2]
        
        if current_version < required_version:
            print(f"❌ Python {PYTHON_REQUIRES} or higher is required.")
            print(f"Current version: {self.python_version}")
            sys.exit(1)
        else:
            print(f"✅ Python version {self.python_version} meets requirements")
    
    def check_system_requirements(self):
        """Check system requirements and compatibility"""
        try:
            import psutil
            
            # Check RAM
            ram_gb = psutil.virtual_memory().total / (1024**3)
            if ram_gb < 4:
                print("⚠️  Warning: Less than 4GB RAM detected. Training may be slow.")
            else:
                print(f"✅ RAM: {ram_gb:.1f}GB available")
            
            # Check disk space
            disk_usage = psutil.disk_usage('/')
            free_gb = disk_usage.free / (1024**3)
            if free_gb < 10:
                print("⚠️  Warning: Less than 10GB free disk space. Food-101 dataset requires ~5GB.")
            else:
                print(f"✅ Disk space: {free_gb:.1f}GB available")
        except ImportError:
            print("ℹ️  psutil not available. Skipping system requirements check.")
        
        # Check for GPU (TensorFlow)
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"✅ GPU detected: {len(gpus)} device(s)")
                for i, gpu in enumerate(gpus):
                    print(f"   GPU {i}: {gpu.name}")
            else:
                print("ℹ️  No GPU detected. Training will use CPU (slower).")
        except ImportError:
            print("ℹ️  TensorFlow not installed yet. GPU detection will be available after installation.")
    
    def create_project_structure(self):
        """Create necessary project directories"""
        directories = [
            "data",
            "models", 
            "logs",
            "downloads",
            "config",
            "exports",
            "tests",
            "notebooks",
            "assets/images",
            "assets/icons"
        ]
        
        print("📁 Creating project structure...")
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"   Created: {directory}/")
        
        # Create .gitignore if it doesn't exist
        gitignore_path = self.project_root / ".gitignore"
        if not gitignore_path.exists():
            gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter
.ipynb_checkpoints

# Data and models
data/food-101/
organized_food101/
downloads/
models/*.h5
models/*.pb
*.pkl

# Logs
logs/
*.log

# OS
.DS_Store
Thumbs.db

# Streamlit
.streamlit/secrets.toml

# Temporary files
*.tmp
*.temp
""".strip()
            
            with open(gitignore_path, 'w') as f:
                f.write(gitignore_content)
            print("   Created: .gitignore")
    
    def create_config_files(self):
        """Create default configuration files"""
        print("⚙️  Creating configuration files...")
        
        # Create streamlit config
        streamlit_config_dir = self.project_root / ".streamlit"
        streamlit_config_dir.mkdir(exist_ok=True)
        
        streamlit_config = {
            "theme": {
                "primaryColor": "#2E8B57",
                "backgroundColor": "#FFFFFF", 
                "secondaryBackgroundColor": "#F0F2F6",
                "textColor": "#262730"
            },
            "server": {
                "maxUploadSize": 200,
                "enableCORS": False,
                "enableXsrfProtection": False
            },
            "browser": {
                "gatherUsageStats": False
            },
            "global": {
                "developmentMode": False
            }
        }
        
        with open(streamlit_config_dir / "config.toml", 'w') as f:
            for section, options in streamlit_config.items():
                f.write(f"[{section}]\n")
                for key, value in options.items():
                    if isinstance(value, str):
                        f.write(f'{key} = "{value}"\n')
                    else:
                        f.write(f'{key} = {value}\n')
                f.write("\n")
        
        print("   Created: .streamlit/config.toml")
    
    def install_requirements(self, dev=False, gpu=False):
        """Install requirements with progress tracking"""
        requirements = INSTALL_REQUIRES.copy()
        
        if dev:
            requirements.extend(EXTRAS_REQUIRE["dev"])
        
        if gpu:
            requirements.extend(EXTRAS_REQUIRE["gpu"])
        
        print(f"📦 Installing {len(requirements)} packages...")
        
        failed_packages = []
        for i, package in enumerate(requirements, 1):
            try:
                print(f"   [{i}/{len(requirements)}] Installing {package}...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", package, "--quiet"
                ])
            except subprocess.CalledProcessError as e:
                print(f"   ❌ Failed to install {package}")
                failed_packages.append(package)
        
        if failed_packages:
            print(f"\n⚠️  Failed to install {len(failed_packages)} packages:")
            for package in failed_packages:
                print(f"   - {package}")
            print("\nTry installing them manually:")
            for package in failed_packages:
                print(f"   pip install {package}")
        else:
            print("✅ All packages installed successfully!")
    
    def verify_installation(self):
        """Verify that all critical packages are installed and working"""
        print("🔍 Verifying installation...")
        
        critical_packages = [
            ("streamlit", "streamlit"),
            ("tensorflow", "tensorflow"),
            ("cv2", "opencv-python"),
            ("pandas", "pandas"),
            ("numpy", "numpy"),
            ("PIL", "Pillow"),
            ("plotly", "plotly"),
            ("sklearn", "scikit-learn")
        ]
        
        failed_imports = []
        
        for import_name, package_name in critical_packages:
            try:
                __import__(import_name)
                print(f"   ✅ {package_name}")
            except ImportError:
                print(f"   ❌ {package_name}")
                failed_imports.append(package_name)
        
        if failed_imports:
            print(f"\n⚠️  Some packages failed to import:")
            for package in failed_imports:
                print(f"   pip install {package}")
            return False
        else:
            print("✅ All critical packages verified!")
            return True
    
    def create_demo_data(self):
        """Create demo data for immediate testing"""
        print("🎬 Creating demo data...")
        
        # Create sample meal history
        demo_meals = {
            "2025-06-29": [
                {
                    "food_name": "grilled_salmon",
                    "calories": 231.0,
                    "protein": 25.0,
                    "carbs": 0.0,
                    "fats": 14.0,
                    "portion_size": 150,
                    "timestamp": "08:30:00"
                },
                {
                    "food_name": "caesar_salad", 
                    "calories": 158.0,
                    "protein": 3.0,
                    "carbs": 6.0,
                    "fats": 14.0,
                    "portion_size": 200,
                    "timestamp": "12:45:00"
                }
            ]
        }
        
        meal_history_path = self.project_root / "data" / "meal_history.json"
        with open(meal_history_path, 'w') as f:
            json.dump(demo_meals, f, indent=2)
        
        print("   Created: data/meal_history.json")
        print("✅ Demo data created!")
    
    def run_health_check(self):
        """Run comprehensive health check"""
        print("\n🏥 Running health check...")
        
        checks = [
            ("Python version", self.check_python_version),
            ("System requirements", self.check_system_requirements),
            ("Installation verification", self.verify_installation)
        ]
        
        passed_checks = 0
        for check_name, check_func in checks:
            try:
                result = check_func()
                if result != False:  # None or True counts as pass
                    passed_checks += 1
            except Exception as e:
                print(f"❌ {check_name} failed: {e}")
        
        print(f"\n📊 Health check results: {passed_checks}/{len(checks)} checks passed")
        
        if passed_checks == len(checks):
            print("🎉 System is ready! You can now run:")
            print("   streamlit run app.py")
        else:
            print("⚠️  Some issues detected. Please resolve them before running the application.")
    
    def generate_requirements_txt(self):
        """Generate comprehensive requirements.txt files"""
        print("📝 Generating requirements files...")
        
        # Generate main requirements.txt
        requirements_path = self.project_root / "requirements.txt"
        with open(requirements_path, 'w') as f:
            f.write("# AI Food Calorie Estimator - Food-101\n")
            f.write("# Core Dependencies\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Author: {AUTHOR}\n\n")
            
            for req in INSTALL_REQUIRES:
                f.write(f"{req}\n")
                
            f.write("\n# Optional GPU support (uncomment if using GPU):\n")
            if "gpu" in EXTRAS_REQUIRE:
                for dep in EXTRAS_REQUIRE["gpu"]:
                    f.write(f"# {dep}\n")
        
        # Generate development requirements
        dev_requirements_path = self.project_root / "requirements-dev.txt"
        with open(dev_requirements_path, 'w') as f:
            f.write("# AI Food Calorie Estimator - Development Requirements\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Author: {AUTHOR}\n\n")
            f.write("-r requirements.txt\n\n")
            
            if "dev" in EXTRAS_REQUIRE:
                f.write("# Development Dependencies\n")
                for dep in EXTRAS_REQUIRE["dev"]:
                    f.write(f"{dep}\n")
        
        print(f"✅ Generated requirements files:")
        print(f"   - {requirements_path}")
        print(f"   - {dev_requirements_path}")
        
        return requirements_path, dev_requirements_path

def main():
    """Main setup function"""
    print("🍎 AI Food Calorie Estimator - Food-101 Setup")
    print("=" * 50)
    print(f"Version: {VERSION}")
    print(f"Author: {AUTHOR}")
    print(f"Date: 2025-06-29 09:23:21")
    print("=" * 50)
    
    setup_manager = SetupManager()
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Setup AI Food Calorie Estimator")
    parser.add_argument("--dev", action="store_true", help="Install development dependencies")
    parser.add_argument("--gpu", action="store_true", help="Install GPU support")
    parser.add_argument("--quick", action="store_true", help="Quick setup (skip some checks)")
    parser.add_argument("--health-check", action="store_true", help="Run health check only")
    
    args = parser.parse_args()
    
    try:
        if args.health_check:
            setup_manager.run_health_check()
            return
        
        # Initial checks
        setup_manager.check_python_version()
        
        if not args.quick:
            setup_manager.check_system_requirements()
        
        # Setup project
        setup_manager.create_project_structure()
        setup_manager.create_config_files()
        setup_manager.generate_requirements_txt()
        
        # Install dependencies
        setup_manager.install_requirements(dev=args.dev, gpu=args.gpu)
        
        # Verify and finalize
        setup_manager.verify_installation()
        setup_manager.create_demo_data()
        
        # Final health check
        if not args.quick:
            setup_manager.run_health_check()
        
        print("\n🎉 Setup completed successfully!")
        print("\n🚀 Next steps:")
        print("1. Download Food-101 dataset: python food101_downloader.py --all")
        print("2. Run the application: streamlit run app.py")
        print("3. Open your browser to: http://localhost:8501")
        
    except KeyboardInterrupt:
        print("\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)

# Standard setuptools configuration
setup(
    name=PROJECT_NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=URL,
    project_urls=PROJECT_URLS,
    packages=find_packages(),
    classifiers=CLASSIFIERS,
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    keywords=KEYWORDS,
    include_package_data=True,
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "food-estimator=app:main",
            "food-trainer=model_trainer:main",
            "food-downloader=food101_downloader:main",
        ],
    },
)

if __name__ == "__main__":
    main()