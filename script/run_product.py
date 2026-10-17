"""Check the local Linux runtime, then start AudioLab without training tools."""
from src.product.runtime import main

if __name__ == '__main__':
    raise SystemExit(main())
