#!/bin/bash
# Comprehensive Docker build and test script
# Tests both main and test images, including CUDA extension verification

set -e  # Exit on error

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}GroundedDINO-VL Docker Build & Test${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

# Function to print status
print_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $1"
    else
        echo -e "${RED}✗${NC} $1"
        exit 1
    fi
}

# Step 1: Build main image
echo -e "${YELLOW}Step 1: Building main Docker image...${NC}"
docker build -t groundeddino_vl:2025.11.0 .
print_status "Main image built successfully"
echo ""

# Step 2: Test basic imports
echo -e "${YELLOW}Step 2: Testing basic imports...${NC}"

docker run --rm groundeddino_vl:2025.11.0 python -c "import groundeddino_vl; print(f'groundeddino_vl version: {groundeddino_vl.__version__}')"
print_status "groundeddino_vl import works"

docker run --rm groundeddino_vl:2025.11.0 python -c "import groundingdino; print(f'groundingdino version: {groundingdino.__version__}')"
print_status "groundingdino backward compatibility works"

docker run --rm groundeddino_vl:2025.11.0 python -c "import shadow_dino; print(f'shadow_dino version: {shadow_dino.__version__}')"
print_status "shadow_dino wrapper works"
echo ""

# Step 3: Test CUDA extension
echo -e "${YELLOW}Step 3: Verifying CUDA extension (_C)...${NC}"

# Check if _C module exists
docker run --rm groundeddino_vl:2025.11.0 python -c "
import groundeddino_vl
if hasattr(groundeddino_vl, '_C'):
    print('✓ groundeddino_vl._C extension exists')
    import groundeddino_vl._C as C
    print(f'✓ _C module loaded successfully')
    print(f'  _C attributes: {len(dir(C))} items')
else:
    print('⚠ _C extension not found (CPU-only build)')
" 2>&1
print_status "CUDA extension check completed"
echo ""

# Step 4: Build test image
echo -e "${YELLOW}Step 4: Building test Docker image...${NC}"
docker build -f Dockerfile.test -t groundeddino_vl:test .
print_status "Test image built successfully"
echo ""

# Step 5: Run all tests
echo -e "${YELLOW}Step 5: Running test suite...${NC}"
docker run --rm groundeddino_vl:test
print_status "All tests passed"
echo ""

# Step 6: Run specific _C extension tests
echo -e "${YELLOW}Step 6: Running CUDA extension specific tests...${NC}"
docker run --rm groundeddino_vl:test python -m pytest tests/test_import_names.py::test_cuda_extension_loadable -v
print_status "CUDA extension tests passed"
echo ""

# Step 7: Summary
echo -e "${BLUE}=========================================${NC}"
echo -e "${GREEN}✓ All verification checks passed!${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""
echo "Summary:"
echo "  • Main image: groundeddino_vl:2025.11.0"
echo "  • Test image: groundeddino_vl:test"
echo "  • All imports working"
echo "  • CUDA extension verified"
echo "  • All tests passed"
echo ""
echo "Images are ready for use!"
echo ""
