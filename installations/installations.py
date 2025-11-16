import subprocess
import sys
import os
import pathlib
os.environ["TOKENIZERS_PARALLELISM"] = "false"
REQ_FILE = pathlib.Path(__file__).parent / "requirements.txt"


def run(cmd):
    print(f"$ {' '.join(cmd)}")
    subprocess.check_call(cmd)


def main():
    run([sys.executable, "-m", "pip", "install",
        "--upgrade", "pip", "setuptools", "wheel"])
    # Uncomment next line if you choose Option 1 (no torchvision)
    # run([sys.executable, "-m", "pip", "uninstall", "-y", "torchvision", "torchaudio"])
    run([sys.executable, "-m", "pip", "install", "-r", str(REQ_FILE)])
    run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    run([sys.executable, "-c", "import nltk; nltk.download('punkt')"])
    print("✅ Clean environment ready.")


if __name__ == "__main__":
    main()
