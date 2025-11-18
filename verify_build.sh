#!/bin/bash
# Quick verification script for Docker builds
# Run this to verify everything is ready

set -e  # Exit on error

echo "========================================="
echo "GroundedDINO-VL Docker Build Verification"
echo "========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Docker is available
echo "1. Checking Docker availability..."
if command -v docker &> /dev/null; then
    echo -e "${GREEN}✓${NC} Docker is installed"
    docker --version
else
    echo -e "${RED}✗${NC} Docker is not installed"
    exit 1
fi
echo ""

# Check if required files exist
echo "2. Checking required files..."
files=("Dockerfile" "Dockerfile.test" "pyproject.toml" "setup.py" "tests/test_import.py")
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $file exists"
    else
        echo -e "${RED}✗${NC} $file is missing"
        exit 1
    fi
done
echo ""

# Check version files
echo "3. Checking version files..."
if grep -q "v2.0.0" groundeddino_vl/version.py; then
    echo -e "${GREEN}✓${NC} groundeddino_vl/version.py is correct"
else
    echo -e "${RED}✗${NC} groundeddino_vl/version.py version mismatch"
fi

if grep -q "v2.0.0" groundingdino/version.py; then
    echo -e "${GREEN}✓${NC} groundingdino/version.py is correct"
else
    echo -e "${RED}✗${NC} groundingdino/version.py version mismatch"
fi
echo ""

# Check test files
echo "4. Checking test files..."
test_count=$(find tests -name "test_*.py" | wc -l)
echo -e "${GREEN}✓${NC} Found $test_count test files"
echo ""

echo "========================================="
echo -e "${GREEN}All checks passed!${NC}"
echo "========================================="
echo ""
echo "Ready to build Docker images!"
echo ""
echo "Next steps:"
echo "  1. Build main image:  docker build -t groundeddino_vl:v2.0.0 ."
echo "  2. Build test image:  docker build -f Dockerfile.test -t groundeddino_vl:test ."
echo "  3. Run tests:         docker run --rm groundeddino_vl:test"
echo ""
