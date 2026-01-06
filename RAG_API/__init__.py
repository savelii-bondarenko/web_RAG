from pathlib import Path
from dotenv import load_dotenv
from .graph_logic import RAGGraph
from .engine import prepare_rag_assets

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

