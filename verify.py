import sys

def load_patterns(filepath):
    patterns = set()
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or "#SUP:" not in line:
                    continue
                
                parts = line.split("#SUP:")
                items_str = parts[0].strip()
                support = int(parts[1].strip())
                
                items = tuple(sorted([int(x) for x in items_str.split()]))
                patterns.add((items, support))
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        sys.exit(1)
    return patterns

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python verify.py <file_hamm> <file_spmf>")
        sys.exit(1)

    file_h = sys.argv[1]
    file_s = sys.argv[2]

    print(f"Comparing:\n  [H] {file_h}\n  [S] {file_s}\n")

    set_h = load_patterns(file_h)
    set_s = load_patterns(file_s)

    if set_h == set_s:
        print("✅ [SUCCESS] 100% Match! Hamm results are correct.")
        print(f"Total patterns found: {len(set_h)}")
    else:
        print("❌ [FAILURE] Mismatch detected!")
        print(f"Hamm count: {len(set_h)}")
        print(f"SPMF count: {len(set_s)}")
        
        only_h = set_h - set_s
        only_s = set_s - set_h
        
        if only_h:
            print(f"\nItems only in Hamm (Potential False Positives): {list(only_h)[:5]} ...")
        if only_s:
            print(f"\nItems missing in Hamm (Potential False Negatives): {list(only_s)[:5]} ...")