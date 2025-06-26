#!/usr/bin/env python3
"""
AI Crypto Trading Bot - Installer Package Creator
Creates executable installers for Windows, macOS, and Linux
"""

import os
import sys
import shutil
import zipfile
import subprocess
from pathlib import Path
import json
import datetime

class InstallerCreator:
    """Creates installation packages for all platforms"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.output_dir = self.project_root / "installers"
        self.temp_dir = self.project_root / "temp_build"
        
        # Core files to include
        self.core_files = [
            "advanced_ai_system.py",
            "advanced_quant_engine.py", 
            "advanced_order_system.py",
            "arctic_data_manager.py",
            "smart_performance_optimizer.py",
            "autonomous_ai_trader.py",
            "real_ai_integration.py",
            "multilayer_api_protection.py",
            "system_health_check.py",
            "PERFORMANCE_CALCULATOR.py",
            "check_install.py",
            "healthcheck.sh"
        ]
        
        # Directories to include
        self.core_dirs = [
            "config", "data", "logs", "backups", 
            "models", "strategies", "utils", "core",
            "config_templates"
        ]
        
        # Documentation files
        self.doc_files = [
            "README.md",
            "replit.md", 
            "REMOTE_ACCESS_DEPLOYMENT_GUIDE.md",
            "INSTALLATION_GUIDE_LINUX.md",
            "INSTALLATION_GUIDE_MACOS.md", 
            "INSTALLATION_GUIDE_WINDOWS.md",
            "SYSTEM_HEALTH_REPORT.md"
        ]
        
    def create_output_dirs(self):
        """Create output directories"""
        self.output_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)
        
    def copy_core_files(self):
        """Copy core application files to temp directory"""
        print("Copying core application files...")
        
        # Copy core Python files
        for file in self.core_files:
            if (self.project_root / file).exists():
                shutil.copy2(self.project_root / file, self.temp_dir / file)
                print(f"  ✓ {file}")
            else:
                print(f"  ⚠ Missing: {file}")
                
        # Copy directories
        for dir_name in self.core_dirs:
            src_dir = self.project_root / dir_name
            if src_dir.exists():
                dst_dir = self.temp_dir / dir_name
                if dst_dir.exists():
                    shutil.rmtree(dst_dir)
                shutil.copytree(src_dir, dst_dir)
                print(f"  ✓ {dir_name}/")
            else:
                # Create empty directory
                (self.temp_dir / dir_name).mkdir(exist_ok=True)
                print(f"  + Created: {dir_name}/")
                
        # Copy documentation
        for doc in self.doc_files:
            if (self.project_root / doc).exists():
                shutil.copy2(self.project_root / doc, self.temp_dir / doc)
                print(f"  ✓ {doc}")
                
    def create_requirements_file(self):
        """Create requirements.txt file"""
        requirements = [
            "streamlit>=1.28.0",
            "pandas>=2.0.0", 
            "numpy>=1.24.0",
            "plotly>=5.15.0",
            "scikit-learn>=1.3.0",
            "requests>=2.31.0",
            "cryptography>=41.0.0",
            "apscheduler>=3.10.0",
            "yfinance>=0.2.0",
            "beautifulsoup4>=4.12.0",
            "feedparser>=6.0.0",
            "psutil>=5.9.0",
            "joblib>=1.3.0",
            "scipy>=1.11.0",
            "trafilatura>=1.6.0",
            "sendgrid>=6.10.0"
        ]
        
        with open(self.temp_dir / "requirements.txt", "w") as f:
            f.write("\n".join(requirements))
        print("  ✓ requirements.txt")
        
    def create_windows_package(self):
        """Create Windows installer package"""
        print("\nCreating Windows installer package...")
        
        windows_dir = self.temp_dir / "windows"
        windows_dir.mkdir(exist_ok=True)
        
        # Copy Windows installer script
        shutil.copy2(
            self.project_root / "tradingbot-installer-windows.exe.ps1",
            windows_dir / "install.ps1"
        )
        
        # Create batch launcher
        batch_content = '''@echo off
echo AI Crypto Trading Bot - Windows Installer
echo ==========================================
echo.
echo This will install the AI Crypto Trading Bot on your Windows system.
echo Installation directory: %USERPROFILE%\\ai-trading-bot
echo.
pause
echo.
echo Starting installation...
powershell -ExecutionPolicy Bypass -File "%~dp0install.ps1"
pause
'''
        with open(windows_dir / "install.bat", "w") as f:
            f.write(batch_content)
            
        # Create Windows README
        windows_readme = '''# AI Crypto Trading Bot - Windows Installation

## Quick Start

1. Right-click "install.bat" and select "Run as administrator"
2. Follow the installation prompts
3. Configure API keys in .env file
4. Run "tradingbot" from command line or use desktop shortcut

## Requirements

- Windows 10/11 x64
- Administrator privileges (for installation only)
- Internet connection for downloading dependencies

## Manual Installation

If the automatic installer fails:

1. Install Python 3.11 from python.org
2. Run: `powershell -ExecutionPolicy Bypass -File install.ps1`

## Support

- Health check: `python check_install.py`
- Documentation: See README.md files
- Troubleshooting: Check logs/ directory
'''
        with open(windows_dir / "README.txt", "w") as f:
            f.write(windows_readme)
            
        print("  ✓ Windows package created")
        
    def create_macos_package(self):
        """Create macOS installer package"""
        print("\nCreating macOS installer package...")
        
        macos_dir = self.temp_dir / "macos"
        macos_dir.mkdir(exist_ok=True)
        
        # Copy macOS installer script
        shutil.copy2(
            self.project_root / "tradingbot-installer-macos.sh",
            macos_dir / "install.sh"
        )
        
        # Make executable
        os.chmod(macos_dir / "install.sh", 0o755)
        
        # Create macOS README
        macos_readme = '''# AI Crypto Trading Bot - macOS Installation

## Quick Start

1. Open Terminal
2. Navigate to this folder: `cd /path/to/this/folder`
3. Run: `./install.sh`
4. Follow the installation prompts
5. Configure API keys in .env file
6. Run `tradingbot` from terminal or use AI Trading Bot.app

## Requirements

- macOS 10.15+ (Intel or Apple Silicon)
- Xcode Command Line Tools
- Internet connection

## Architecture Support

- Intel x64: Full support with standard optimizations
- Apple Silicon (M1/M2/M3): ARM-optimized with Metal acceleration

## Manual Installation

If the automatic installer fails:

1. Install Homebrew: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
2. Install Python: `brew install python@3.11`
3. Run the installer: `./install.sh`

## Support

- Health check: `./healthcheck.sh`
- Documentation: See README.md files
- Troubleshooting: Check ~/ai-trading-bot/logs/
'''
        with open(macos_dir / "README.txt", "w") as f:
            f.write(macos_readme)
            
        print("  ✓ macOS package created")
        
    def create_linux_package(self):
        """Create Linux installer package"""
        print("\nCreating Linux installer package...")
        
        linux_dir = self.temp_dir / "linux"
        linux_dir.mkdir(exist_ok=True)
        
        # Copy Linux installer script
        shutil.copy2(
            self.project_root / "tradingbot-installer-linux.sh",
            linux_dir / "install.sh"
        )
        
        # Make executable
        os.chmod(linux_dir / "install.sh", 0o755)
        
        # Copy health check script
        shutil.copy2(
            self.project_root / "healthcheck.sh",
            linux_dir / "healthcheck.sh"
        )
        os.chmod(linux_dir / "healthcheck.sh", 0o755)
        
        # Create Linux README
        linux_readme = '''# AI Crypto Trading Bot - Linux Installation

## Quick Start

1. Open terminal
2. Navigate to this folder: `cd /path/to/this/folder`
3. Run: `./install.sh`
4. Follow the installation prompts
5. Configure API keys in .env file
6. Start service: `sudo systemctl start ai-trading-bot`

## Requirements

- Ubuntu 18.04+ / Debian 10+ / CentOS 7+ / RHEL 7+
- sudo privileges (for system packages and service installation)
- Internet connection

## Distribution Support

- Ubuntu/Debian: Full automatic installation
- CentOS/RHEL/Fedora: Full automatic installation  
- Other distributions: Manual dependency installation may be required

## Server Deployment

Perfect for VPS, cloud instances, and dedicated servers:

- Systemd service for 24/7 operation
- Automatic startup on boot
- Remote web access on port 5000/8501
- SSH tunnel support for secure access

## Manual Installation

If the automatic installer fails:

1. Install Python 3.8+: `sudo apt install python3 python3-pip python3-venv`
2. Install system packages: `sudo apt install build-essential git curl`
3. Run the installer: `./install.sh`

## Remote Access

- Direct: `http://server-ip:8501`
- SSH tunnel: `ssh -L 8501:localhost:8501 user@server-ip`
- See REMOTE_ACCESS_DEPLOYMENT_GUIDE.md for complete setup

## Support

- Health check: `./healthcheck.sh`
- Service status: `sudo systemctl status ai-trading-bot`
- View logs: `sudo journalctl -u ai-trading-bot -f`
- Documentation: See README.md files
'''
        with open(linux_dir / "README.txt", "w") as f:
            f.write(linux_readme)
            
        print("  ✓ Linux package created")
        
    def create_universal_package(self):
        """Create universal package with all installers"""
        print("\nCreating universal package...")
        
        universal_dir = self.temp_dir / "universal"
        universal_dir.mkdir(exist_ok=True)
        
        # Copy all core files to universal package
        for file in self.core_files:
            if (self.temp_dir / file).exists():
                shutil.copy2(self.temp_dir / file, universal_dir / file)
                
        for dir_name in self.core_dirs:
            src_dir = self.temp_dir / dir_name
            if src_dir.exists():
                dst_dir = universal_dir / dir_name
                if dst_dir.exists():
                    shutil.rmtree(dst_dir)
                shutil.copytree(src_dir, dst_dir)
                
        # Copy installer scripts
        installers_dir = universal_dir / "installers"
        installers_dir.mkdir(exist_ok=True)
        
        # Copy platform installers
        for platform in ["windows", "macos", "linux"]:
            platform_src = self.temp_dir / platform
            platform_dst = installers_dir / platform
            if platform_src.exists():
                shutil.copytree(platform_src, platform_dst)
                
        # Copy requirements and documentation
        shutil.copy2(self.temp_dir / "requirements.txt", universal_dir / "requirements.txt")
        
        for doc in self.doc_files:
            if (self.temp_dir / doc).exists():
                shutil.copy2(self.temp_dir / doc, universal_dir / doc)
                
        # Create main README
        main_readme = '''# AI Crypto Trading Bot - Universal Installation Package

## Platform-Specific Installation

### Windows 10/11 x64
```
cd installers/windows
Right-click install.bat → "Run as administrator"
```

### macOS (Intel + Apple Silicon)
```
cd installers/macos
./install.sh
```

### Linux (Ubuntu/Debian/CentOS/RHEL)
```
cd installers/linux
./install.sh
```

## Manual Installation

If you prefer manual installation or the automatic installers fail:

1. Install Python 3.8+ for your platform
2. Install required packages: `pip install -r requirements.txt`
3. Run: `python advanced_ai_system.py`

## Quick Start

1. Run the appropriate installer for your platform
2. Configure API keys in `.env` file
3. Start the bot: `tradingbot` or `python advanced_ai_system.py`
4. Access web interface: `http://localhost:5000`

## Configuration

- Environment variables: `.env` file
- Main configuration: `config/config.yaml`
- Templates available in: `config_templates/`

## Documentation

- Complete deployment guide: `REMOTE_ACCESS_DEPLOYMENT_GUIDE.md`
- Platform-specific guides: `INSTALLATION_GUIDE_*.md`
- System health: Run `python check_install.py`

## Support

- Health check: `python check_install.py` (all platforms)
- Quick check: `./healthcheck.sh` (Linux/macOS)
- Logs: Check `logs/` directory
- System status: `python system_health_check.py`

## Features

- 24/7 autonomous trading with AI decision making
- Multi-exchange support (Binance, Coinbase, Bybit, OKX, Kraken)
- Advanced quantitative analysis with backtesting
- Real-time market sentiment analysis
- HuggingFace AI models integration
- Smart performance optimization
- Enterprise-grade security and encryption
- Remote access and deployment ready

Ready for live trading with real capital when properly configured.
'''
        with open(universal_dir / "README.md", "w") as f:
            f.write(main_readme)
            
        print("  ✓ Universal package created")
        
    def create_zip_packages(self):
        """Create ZIP archives for distribution"""
        print("\nCreating distribution archives...")
        
        packages = {
            "AI_Trading_Bot_Windows_Installer.zip": "windows",
            "AI_Trading_Bot_macOS_Installer.zip": "macos", 
            "AI_Trading_Bot_Linux_Installer.zip": "linux",
            "AI_Trading_Bot_Universal_Package.zip": "universal"
        }
        
        for zip_name, folder in packages.items():
            zip_path = self.output_dir / zip_name
            folder_path = self.temp_dir / folder
            
            if folder_path.exists():
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(folder_path):
                        for file in files:
                            file_path = Path(root) / file
                            arc_path = file_path.relative_to(folder_path)
                            zipf.write(file_path, arc_path)
                            
                file_size = zip_path.stat().st_size / (1024 * 1024)
                print(f"  ✓ {zip_name} ({file_size:.1f} MB)")
            else:
                print(f"  ⚠ Skipped {zip_name} (source not found)")
                
    def create_manifest(self):
        """Create installation manifest"""
        manifest = {
            "name": "AI Crypto Trading Bot",
            "version": "1.0.0",
            "build_date": datetime.datetime.now().isoformat(),
            "platforms": {
                "windows": {
                    "supported": ["Windows 10", "Windows 11"],
                    "architecture": ["x64"],
                    "installer": "AI_Trading_Bot_Windows_Installer.zip"
                },
                "macos": {
                    "supported": ["macOS 10.15+"],
                    "architecture": ["x64", "arm64"],
                    "installer": "AI_Trading_Bot_macOS_Installer.zip"
                },
                "linux": {
                    "supported": ["Ubuntu 18.04+", "Debian 10+", "CentOS 7+", "RHEL 7+"],
                    "architecture": ["x64"],
                    "installer": "AI_Trading_Bot_Linux_Installer.zip"
                }
            },
            "features": [
                "24/7 Autonomous AI Trading",
                "Multi-Exchange Support (5+ exchanges)",
                "Advanced Quantitative Analysis",
                "Real-time Sentiment Analysis", 
                "HuggingFace AI Models Integration",
                "Smart Performance Optimization",
                "Enterprise Security & Encryption",
                "Remote Access & Deployment Ready"
            ],
            "requirements": {
                "python": "3.8+",
                "memory": "2GB minimum, 4GB+ recommended",
                "disk": "1GB minimum, 5GB+ recommended",
                "network": "Internet connection required"
            }
        }
        
        with open(self.output_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
            
        print("  ✓ manifest.json")
        
    def cleanup(self):
        """Remove temporary build directory"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        print("  ✓ Cleanup completed")
        
    def create_all_installers(self):
        """Create all installer packages"""
        print("🚀 AI Crypto Trading Bot - Installer Creator")
        print("=" * 50)
        
        try:
            self.create_output_dirs()
            self.copy_core_files()
            self.create_requirements_file()
            
            self.create_windows_package()
            self.create_macos_package() 
            self.create_linux_package()
            self.create_universal_package()
            
            self.create_zip_packages()
            self.create_manifest()
            
            print(f"\n✅ Installation packages created successfully!")
            print(f"📦 Output directory: {self.output_dir}")
            print(f"📋 Files created:")
            
            for file in self.output_dir.glob("*"):
                if file.is_file():
                    size = file.stat().st_size / (1024 * 1024)
                    print(f"   • {file.name} ({size:.1f} MB)")
                    
            print(f"\n🎯 Ready for distribution!")
            print(f"   Windows: AI_Trading_Bot_Windows_Installer.zip")
            print(f"   macOS: AI_Trading_Bot_macOS_Installer.zip") 
            print(f"   Linux: AI_Trading_Bot_Linux_Installer.zip")
            print(f"   Universal: AI_Trading_Bot_Universal_Package.zip")
            
        except Exception as e:
            print(f"❌ Error creating installers: {e}")
            return False
        finally:
            self.cleanup()
            
        return True

def main():
    """Main function"""
    creator = InstallerCreator()
    success = creator.create_all_installers()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()