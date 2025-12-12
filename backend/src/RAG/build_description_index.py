import sys, os
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))
from description_index import build_index

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    build_index(path) if path else build_index()