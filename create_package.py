#!/usr/bin/env python3
"""
AI Trading Bot - Package Creator
Creates downloadable packages for Windows, macOS, and Linux
"""

import os
import shutil
import zipfile
import platform
from pathlib import Path

# Core files needed for the trading bot
CORE_FILES = [
    "advanced_ai_system.py",
    "consolidated_bot.py",
    "production_ready_bot.py",
    "simple_app.py",
    "app.py", 
    "install.py",
    "run_trading_bot.py",
    "README.md",
    "institutional_config.json",
    "stress_test.py",
    "SYSTEM_ARCHITECTURE.md",
    "SYSTEM_VALIDATION.md",
    "pyproject.toml",
    ".encryption_key",
    "pre_trained_knowledge_system.py",
    "real_time_market_analyzer.py", 
    "advanced_market_data_collector.py",
    "speed_optimization_engine.py", 
    "session_manager.py",
    "README_PRODUCTION.md",
    "ai_models_downloader.py",
    "REQUISITI_HARDWARE_REALI.md",
    "GUIDA_COMPLETA_RASPBERRY_PI.md",
    "cloud_deployment_guide.md"
]

# Directories to include
CORE_DIRECTORIES = [
    "core/",
    "models/", 
    "utils/",
    "config/",
    "logs/"
]

def create_package_structure(package_name):
    """Create the package directory structure"""
    package_dir = Path(package_name)
    
    # Remove existing package if it exists
    if package_dir.exists():
        shutil.rmtree(package_dir)
    
    # Create main package directory
    package_dir.mkdir()
    
    # Copy core files
    for file in CORE_FILES:
        if os.path.exists(file):
            shutil.copy2(file, package_dir / file)
            print(f"Added: {file}")
    
    # Copy directories
    for directory in CORE_DIRECTORIES:
        if os.path.exists(directory):
            shutil.copytree(directory, package_dir / directory)
            print(f"Added directory: {directory}")
    
    return package_dir

def create_windows_package():
    """Create Windows-specific package"""
    print("Creating Windows package...")
    
    package_dir = create_package_structure("AI_Trading_Bot_Windows")
    
    # Create Windows batch launcher
    batch_content = '''@echo off
title AI Trading Bot
echo =========================================
echo    AI Trading Bot - Windows Launcher
echo =========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    echo.
    pause
    exit /b 1
)

REM Install dependencies if needed
if not exist "installed.flag" (
    echo Installing dependencies...
    python install.py
    if %errorlevel% equ 0 (
        echo. > installed.flag
    ) else (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)

REM Start the trading bot
echo Starting AI Trading Bot...
echo Dashboard will open at: http://localhost:5000
echo.
echo Press Ctrl+C to stop the bot
echo.
streamlit run simple_app.py --server.port 5000 --server.address 0.0.0.0

pause
'''
    
    with open(package_dir / "START_TRADING_BOT.bat", "w") as f:
        f.write(batch_content)
    
    # Create install script for Windows
    install_content = '''@echo off
echo Installing AI Trading Bot dependencies...
python -m pip install --upgrade pip
python install.py
echo.
echo Installation complete! 
echo Run START_TRADING_BOT.bat to launch the application.
pause
'''
    
    with open(package_dir / "INSTALL.bat", "w") as f:
        f.write(install_content)
    
    return package_dir

def create_macos_package():
    """Create macOS-specific package"""
    print("Creating macOS package...")
    
    package_dir = create_package_structure("AI_Trading_Bot_macOS")
    
    # Create macOS shell launcher
    script_content = '''#!/bin/bash
clear
echo "========================================="
echo "   AI Trading Bot - macOS Launcher"
echo "========================================="
echo

cd "$(dirname "$0")"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8+ from https://python.org"
    echo "Or install via Homebrew: brew install python"
    read -p "Press Enter to exit..."
    exit 1
fi

# Install dependencies if needed
if [ ! -f "installed.flag" ]; then
    echo "Installing dependencies..."
    python3 install.py
    if [ $? -eq 0 ]; then
        touch installed.flag
    else
        echo "ERROR: Failed to install dependencies"
        read -p "Press Enter to exit..."
        exit 1
    fi
fi

# Start the trading bot
echo "Starting AI Trading Bot..."
echo "Dashboard will open at: http://localhost:5000"
echo
echo "Press Ctrl+C to stop the bot"
echo
python3 -m streamlit run simple_app.py --server.port 5000 --server.address 0.0.0.0

read -p "Press Enter to exit..."
'''
    
    script_path = package_dir / "START_TRADING_BOT.sh"
    with open(script_path, "w") as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)
    
    # Create install script for macOS
    install_content = '''#!/bin/bash
echo "Installing AI Trading Bot dependencies..."
python3 -m pip install --upgrade pip
python3 install.py
echo
echo "Installation complete!"
echo "Run ./START_TRADING_BOT.sh to launch the application."
read -p "Press Enter to continue..."
'''
    
    install_path = package_dir / "INSTALL.sh"
    with open(install_path, "w") as f:
        f.write(install_content)
    os.chmod(install_path, 0o755)
    
    return package_dir

def create_linux_package():
    """Create Linux-specific package"""
    print("Creating Linux package...")
    
    package_dir = create_package_structure("AI_Trading_Bot_Linux")
    
    # Create Linux shell launcher
    script_content = '''#!/bin/bash
clear
echo "========================================="
echo "   AI Trading Bot - Linux Launcher"
echo "========================================="
echo

cd "$(dirname "$0")"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Install via package manager:"
    echo "  Ubuntu/Debian: sudo apt install python3 python3-pip"
    echo "  CentOS/RHEL: sudo yum install python3 python3-pip"
    echo "  Arch: sudo pacman -S python python-pip"
    read -p "Press Enter to exit..."
    exit 1
fi

# Install dependencies if needed
if [ ! -f "installed.flag" ]; then
    echo "Installing dependencies..."
    python3 install.py
    if [ $? -eq 0 ]; then
        touch installed.flag
    else
        echo "ERROR: Failed to install dependencies"
        read -p "Press Enter to exit..."
        exit 1
    fi
fi

# Start the trading bot
echo "Starting AI Trading Bot..."
echo "Dashboard will open at: http://localhost:5000"
echo
echo "Press Ctrl+C to stop the bot"
echo
python3 -m streamlit run simple_app.py --server.port 5000 --server.address 0.0.0.0

read -p "Press Enter to exit..."
'''
    
    script_path = package_dir / "START_TRADING_BOT.sh"
    with open(script_path, "w") as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)
    
    # Create install script for Linux
    install_content = '''#!/bin/bash
echo "Installing AI Trading Bot dependencies..."
python3 -m pip install --upgrade pip --user
python3 install.py
echo
echo "Installation complete!"
echo "Run ./START_TRADING_BOT.sh to launch the application."
read -p "Press Enter to continue..."
'''
    
    install_path = package_dir / "INSTALL.sh"
    with open(install_path, "w") as f:
        f.write(install_content)
    os.chmod(install_path, 0o755)
    
    return package_dir

def create_zip_package(package_dir):
    """Create ZIP archive of the package"""
    zip_name = f"{package_dir.name}.zip"
    
    print(f"Creating ZIP archive: {zip_name}")
    
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(package_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arc_name = os.path.relpath(file_path, package_dir.parent)
                zipf.write(file_path, arc_name)
    
    print(f"Created: {zip_name}")
    return zip_name

def create_all_packages():
    """Create packages for all platforms"""
    print("Creating downloadable packages for all platforms...")
    print("=" * 60)
    
    packages = []
    
    # Create Windows package
    try:
        win_package = create_windows_package()
        win_zip = create_zip_package(win_package)
        packages.append(win_zip)
        shutil.rmtree(win_package)  # Clean up directory after zipping
    except Exception as e:
        print(f"Error creating Windows package: {e}")
    
    # Create macOS package
    try:
        mac_package = create_macos_package()
        mac_zip = create_zip_package(mac_package)
        packages.append(mac_zip)
        shutil.rmtree(mac_package)  # Clean up directory after zipping
    except Exception as e:
        print(f"Error creating macOS package: {e}")
    
    # Create Linux package
    try:
        linux_package = create_linux_package()
        linux_zip = create_zip_package(linux_package)
        packages.append(linux_zip)
        shutil.rmtree(linux_package)  # Clean up directory after zipping
    except Exception as e:
        print(f"Error creating Linux package: {e}")
    
    print("\n" + "=" * 60)
    print("Package creation completed!")
    print(f"Created {len(packages)} downloadable packages:")
    
    for package in packages:
        size_mb = os.path.getsize(package) / (1024 * 1024)
        print(f"  - {package} ({size_mb:.1f} MB)")
    
    print("\nPackages are ready for distribution!")
    print("Users can download, extract, and run the trading bot on their systems.")

if __name__ == "__main__":
    create_all_packages()