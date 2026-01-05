#!/bin/bash

# Document Anonymizer - Quick Installation Script
# Usage: ./install.sh

set -e

echo "========================================"
echo "  Document Anonymizer Installation"
echo "========================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default ocr-preprocessor repo (update this with actual repo URL)
OCR_PREPROCESSOR_REPO="${OCR_PREPROCESSOR_REPO:-https://github.com/kloia/ocr-preprocessor.git}"

# Check Python version
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}Error: Python not found. Please install Python 3.10+${NC}"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "Python version: ${GREEN}$PYTHON_VERSION${NC}"

# Check if Python version is >= 3.10
MAJOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.major)')
MINOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.minor)')

if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 10 ]); then
    echo -e "${RED}Error: Python 3.10+ required (found $PYTHON_VERSION)${NC}"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# ocr-preprocessor path (check multiple locations)
OCR_PREPROCESSOR_PATH=""
POSSIBLE_PATHS=(
    "../ocr-preprocessor"
    "./ocr-preprocessor"
    "../ocr_preprocessor"
    "./ocr_preprocessor"
)

for path in "${POSSIBLE_PATHS[@]}"; do
    if [ -d "$path" ] && [ -f "$path/pyproject.toml" ]; then
        OCR_PREPROCESSOR_PATH="$path"
        break
    fi
done

# If not found, clone it
if [ -z "$OCR_PREPROCESSOR_PATH" ]; then
    echo ""
    echo -e "${YELLOW}ocr-preprocessor not found locally.${NC}"
    echo ""

    # Check if we have the repo URL
    if [[ "$OCR_PREPROCESSOR_REPO" == *"your-org"* ]]; then
        echo "Please provide ocr-preprocessor repository URL:"
        echo ""
        echo "Options:"
        echo "  1. Enter the git clone URL"
        echo "  2. Or set OCR_PREPROCESSOR_REPO environment variable"
        echo "  3. Or manually clone to ../ocr-preprocessor and re-run"
        echo ""
        read -p "Git URL (or press Enter to skip): " USER_REPO

        if [ -n "$USER_REPO" ]; then
            OCR_PREPROCESSOR_REPO="$USER_REPO"
        else
            echo ""
            echo -e "${YELLOW}Skipping ocr-preprocessor clone.${NC}"
            echo "You can install it manually later:"
            echo "  git clone <repo-url> ../ocr-preprocessor"
            echo "  pip install -e ../ocr-preprocessor"
            echo ""
            OCR_PREPROCESSOR_REPO=""
        fi
    fi

    if [ -n "$OCR_PREPROCESSOR_REPO" ]; then
        echo ""
        echo -e "${BLUE}Cloning ocr-preprocessor...${NC}"
        git clone "$OCR_PREPROCESSOR_REPO" ../ocr-preprocessor
        OCR_PREPROCESSOR_PATH="../ocr-preprocessor"
        echo -e "${GREEN}ocr-preprocessor cloned successfully${NC}"
    fi
fi

if [ -n "$OCR_PREPROCESSOR_PATH" ]; then
    echo -e "ocr-preprocessor: ${GREEN}$OCR_PREPROCESSOR_PATH${NC}"
else
    echo -e "ocr-preprocessor: ${YELLOW}not installed (optional, will use fallback)${NC}"
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    $PYTHON_CMD -m venv venv
    echo -e "${GREEN}Virtual environment created${NC}"
else
    echo -e "${YELLOW}Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip -q

# Install ocr-preprocessor if available
if [ -n "$OCR_PREPROCESSOR_PATH" ] && [ -d "$OCR_PREPROCESSOR_PATH" ]; then
    echo ""
    echo "Installing ocr-preprocessor..."
    pip install -e "$OCR_PREPROCESSOR_PATH" -q
    echo -e "${GREEN}ocr-preprocessor installed${NC}"
fi

# Install document-anonymizer
echo ""
echo "Installing document-anonymizer..."
pip install -e . -q
echo -e "${GREEN}document-anonymizer installed${NC}"

# Copy .env file if not exists
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${GREEN}.env file created from .env.example${NC}"
    fi
fi

# Verify installation
echo ""
echo "Verifying installation..."

# Check docanon command
if docanon --version &> /dev/null; then
    VERSION=$(docanon --version 2>/dev/null || echo "installed")
    echo -e "${GREEN}✓ docanon command available${NC}"
else
    echo -e "${YELLOW}! docanon not in PATH (activate venv first)${NC}"
fi

# Check ocr-preprocessor
if $PYTHON_CMD -c "import ocr_preprocessor" 2>/dev/null; then
    echo -e "${GREEN}✓ ocr-preprocessor available${NC}"
else
    echo -e "${YELLOW}! ocr-preprocessor not installed (using fallback preprocessing)${NC}"
fi

# Check EasyOCR
if $PYTHON_CMD -c "import easyocr" 2>/dev/null; then
    echo -e "${GREEN}✓ EasyOCR available${NC}"
else
    echo -e "${YELLOW}! EasyOCR loading... (first run may take time to download models)${NC}"
fi

# Setup global command access
echo ""
echo "----------------------------------------"
echo "Global Command Setup"
echo "----------------------------------------"

DOCANON_PATH="$SCRIPT_DIR/venv/bin/docanon"
ALIAS_LINE="alias docanon='$DOCANON_PATH'"

# Detect shell config file
SHELL_CONFIG=""
if [ -n "$ZSH_VERSION" ] || [ "$SHELL" = "/bin/zsh" ]; then
    SHELL_CONFIG="$HOME/.zshrc"
elif [ -n "$BASH_VERSION" ] || [ "$SHELL" = "/bin/bash" ]; then
    if [ -f "$HOME/.bash_profile" ]; then
        SHELL_CONFIG="$HOME/.bash_profile"
    else
        SHELL_CONFIG="$HOME/.bashrc"
    fi
fi

# Check if alias already exists
ALIAS_EXISTS=false
if [ -n "$SHELL_CONFIG" ] && [ -f "$SHELL_CONFIG" ]; then
    if grep -q "alias docanon=" "$SHELL_CONFIG" 2>/dev/null; then
        ALIAS_EXISTS=true
    fi
fi

if [ "$ALIAS_EXISTS" = true ]; then
    echo -e "${GREEN}✓ docanon alias already configured in $SHELL_CONFIG${NC}"
else
    echo ""
    echo "Would you like to add 'docanon' command globally?"
    echo "This will add an alias to $SHELL_CONFIG"
    echo ""
    read -p "Add global command? [Y/n]: " ADD_ALIAS

    if [ -z "$ADD_ALIAS" ] || [ "$ADD_ALIAS" = "y" ] || [ "$ADD_ALIAS" = "Y" ]; then
        if [ -n "$SHELL_CONFIG" ]; then
            echo "" >> "$SHELL_CONFIG"
            echo "# Document Anonymizer" >> "$SHELL_CONFIG"
            echo "$ALIAS_LINE" >> "$SHELL_CONFIG"
            echo -e "${GREEN}✓ Alias added to $SHELL_CONFIG${NC}"
            echo ""
            echo -e "${YELLOW}Run this to activate now:${NC}"
            echo -e "  ${BLUE}source $SHELL_CONFIG${NC}"
        else
            echo -e "${YELLOW}Could not detect shell config file.${NC}"
            echo "Add this line manually to your shell config:"
            echo -e "  ${BLUE}$ALIAS_LINE${NC}"
        fi
    else
        echo -e "${YELLOW}Skipped global command setup.${NC}"
        echo "You can always run docanon using:"
        echo -e "  ${BLUE}$DOCANON_PATH${NC}"
    fi
fi

echo ""
echo "========================================"
echo -e "${GREEN}Installation complete!${NC}"
echo "========================================"
echo ""
echo "Usage:"
echo ""
echo "  Run anonymization:"
echo -e "     ${BLUE}docanon input.pdf -o ./output/${NC}"
echo ""
echo "  Test with dry-run first:"
echo -e "     ${BLUE}docanon input.pdf --dry-run${NC}"
echo ""
echo "  (Optional) Edit configuration:"
echo -e "     ${BLUE}nano $SCRIPT_DIR/.env${NC}"
echo ""
