from pathlib import Path
from dotenv import load_dotenv

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

