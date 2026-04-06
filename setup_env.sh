#!/bin/bash
mkdir -p data_raw tools results

if ! command -v wget &> /dev/null; then
    echo "⚠️ wget could not be found. Trying to install..."
fi

if [ ! -f "tools/spmf.jar" ]; then
    echo "Downloading spmf.jar..."
    wget https://www.philippe-fournier-viger.com/spmf/spmf.jar -O tools/spmf.jar
else
    echo "spmf.jar already exists in tools/."
fi

echo -e "${GREEN}[1/4] Checking System Dependencies...${NC}"

if ! command -v g++ &> /dev/null; then
    echo "❌ g++ could not be found. Please install it (e.g., sudo apt install g++)."
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "❌ python3 could not be found. Please install it."
    exit 1
fi

echo -e "${GREEN}[2/4] Setting up Python Virtual Environment...${NC}"
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "Created virtual environment in .venv"
else
    echo "Virtual environment already exists."
fi

source .venv/bin/activate

echo -e "${GREEN}[3/4] Installing Python Dependencies...${NC}"
pip install --upgrade pip

pip install pandas numpy matplotlib seaborn tqdm psutil mlxtend

pip freeze > requirements.txt

echo -e "${GREEN}[4/4] Compiling C++ Project...${NC}"
if [ -f "src/main.cpp" ]; then
    g++ -O3 -o tools/hamm src/main.cpp
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Environment Setup Complete!${NC}"
        echo "To activate environment manually, run: source .venv/bin/activate"
    else
        echo "❌ Compilation failed."
        exit 1
    fi
else
    echo "❌ src/main.cpp not found. Please ensure you are in the project root."
    exit 1
fi
