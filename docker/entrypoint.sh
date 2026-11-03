#!/bin/bash
set -euo pipefail

echo "============================================================"
echo "  DEFECT-GAN RGB TRAINING & INFERENCE CONTAINER"
echo "============================================================"

# Show help if requested or no arguments provided
if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ] || [ -z "${1:-}" ]; then
    echo ""
    echo "Usage commands:"
    echo "  train        Explicit training:  docker run ... defectgan:latest train [--config configs/defectgan_1024.yaml]"
    echo "  infer        Run inference:      docker run ... defectgan:latest infer --checkpoint outputs/.../best.pth --input-dir data/S942/OK --output-dir outputs/s942_defect"
    echo "  smoke-test   Run CPU smoke test: docker run ... defectgan:latest smoke-test"
    echo "  gpu-smoke    Bounded GPU check:  docker run ... --gpus all defectgan:latest gpu-smoke"
    echo "  check-env    Inspect GPU/driver: docker run ... defectgan:latest check-env"
    echo "  validate-data Read-only approved dataset validation"
    echo "  pytest       Run unit tests:     docker run ... defectgan:latest pytest"
    echo "  shell        Interactive bash:   docker run -it ... defectgan:latest shell"
    echo ""
    exit 0
fi

COMMAND="$1"
shift

case "$COMMAND" in
    train)
        echo "[INFO] Launching Defect-GAN training..."
        exec python -m src.train "$@"
        ;;
    infer|inference)
        echo "[INFO] Launching Defect-GAN inference..."
        exec python -m src.inference "$@"
        ;;
    smoke-test|smoketest)
        echo "[INFO] Launching complete smoke test suite..."
        exec python scripts/run_smoke_test.py "$@"
        ;;
    gpu-smoke)
        exec python scripts/run_smoke_test.py --gpu-smoke "$@"
        ;;
    check-env|check_env)
        echo "[INFO] Inspecting hardware environment..."
        exec python scripts/check_environment.py "$@"
        ;;
    validate-data)
        exec python scripts/validate_dataset.py "$@"
        ;;
    pytest|test)
        echo "[INFO] Running pytest suite..."
        exec python -m pytest tests -v "$@"
        ;;
    shell|bash)
        exec /bin/bash
        ;;
    *)
        # Pass arbitrary command directly to python or system
        exec "$COMMAND" "$@"
        ;;
esac
