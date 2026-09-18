import argparse, json
from .dataset import generate_dataset
from .evaluate import run_evaluation

def main():
    p = argparse.ArgumentParser(description="VisionCheck computer-vision classifier")
    p.add_argument("--generate-data", action="store_true")
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--data", default="data")
    p.add_argument("--output", default="outputs")
    a = p.parse_args()
    if a.generate_data:
        print(generate_dataset(a.data))
    if a.evaluate:
        print(json.dumps(run_evaluation(f"{a.data}/train", f"{a.data}/test", a.output), indent=2))
    if not a.generate_data and not a.evaluate:
        p.print_help()

if __name__ == "__main__":
    main()
