#!/bin/bash
set -euo pipefail

# ==============================================
# LightRAG Fully Automatic One-Click Installation Script
# ==============================================

clear
echo "========================================"
echo "  LightRAG Fully Automatic Installer    "
echo "========================================"

# --------------------------
# 1. Only Linux systems allowed
# --------------------------
OS=$(uname -s)
if [ "$OS" != "Linux" ]; then
    echo "❌ Only Linux systems are supported. Installation aborted."
    exit 1
fi
echo "✅ System check passed: Linux"

# --------------------------
# Utility functions
# --------------------------
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# --------------------------
# 2. Install dependencies: git python3 pip3
# --------------------------
echo -e "\n📦 Checking and installing system dependencies..."

if command_exists apt; then
    PKG_INSTALL="sudo apt update && sudo apt install -y"
elif command_exists yum || command_exists dnf; then
    PKG_INSTALL="sudo yum install -y"
else
    echo "❌ Unsupported Linux distribution"
    exit 1
fi

# Install git
if ! command_exists git; then
    echo "🔧 Installing git..."
    $PKG_INSTALL git
fi

# Install python3
if ! command_exists python3; then
    echo "🔧 Installing python3..."
    $PKG_INSTALL python3
fi

# Install pip3
if ! command_exists pip3; then
    echo "🔧 Installing python3-pip..."
    $PKG_INSTALL python3-pip
fi

echo "✅ Dependencies installed"

# --------------------------
# 3. Create directory + clone code (recommended structure)
# --------------------------
echo -e "\n📥 Automatically creating LightRAG directory and pulling code..."
REPO_URL="git@github.com:taokong1017/LightRAG.git"
INSTALL_DIR="LightRAG"

# Create and enter directory
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Pull code
if [ -z "$(ls -A ./)" ]; then
    git clone "$REPO_URL" .
else
    git pull
fi

echo "✅ Code pulled to directory: $INSTALL_DIR"

# --------------------------
# 4. Install lightrag library
# --------------------------
echo -e "\n🚀 Installing LightRAG Python library..."
python3 -m pip install --upgrade pip
python3 -m pip install . -q >/dev/null 2>&1
echo "✅ LightRAG installed successfully"

# --------------------------
# 5. Automatically configure ~/.lightrag directory
# --------------------------
echo -e "\n⚙️  Configuring LightRAG working directory..."
USER_NAME=$(whoami) 
HOME_DIR="/home/$USER_NAME"
LIGHTRAG_DIR="$HOME_DIR/.lightrag"

mkdir -p "$LIGHTRAG_DIR"/{logs,inputs,storage}

# Copy and configure .env
if [ -f ".env.example" ]; then
    cp .env.example "$LIGHTRAG_DIR/.env"
    sed -i "s|^LOG_DIR=.*|LOG_DIR=$LIGHTRAG_DIR/logs|" "$LIGHTRAG_DIR/.env"
    sed -i "s|^INPUT_DIR=.*|INPUT_DIR=$LIGHTRAG_DIR/inputs|" "$LIGHTRAG_DIR/.env"
    sed -i "s|^WORKING_DIR=.*|WORKING_DIR=$LIGHTRAG_DIR/storage|" "$LIGHTRAG_DIR/.env"
    echo "✅ .env configured"
fi

# Copy service.py
if [ -f "service.py" ]; then
    cp service.py "$LIGHTRAG_DIR/"
    echo "✅ service.py copied"
fi

# --------------------------
# Installation complete
# --------------------------
echo -e "\n========================================"
echo "🎉 Installation complete!"
echo "📂 Project directory: $(pwd)"
echo "📂 Config directory: $LIGHTRAG_DIR"
echo "========================================"