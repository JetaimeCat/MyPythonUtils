import sys
import warnings

warnings.filterwarnings("ignore")

current_file = str(__file__).replace("\\", "/").split("/")[-2]
sys.path.append(current_file)
