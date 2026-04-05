import os
import subprocess
import sys
import time

DATASETS = [
    "connect-4.data",
    "kr-vs-kp.data",
    "car.data",
    "tic-tac-toe.data",
    "agaricus-lepiota.data"
]

MIN_SUP_RATE = 0.05 

SRC_FILE = "src/Hamm.cpp"
EXE_FILE = "tools/hamm"
DATA_DIR = "data_raw"
OUTPUT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def compile_cpp():
    print("🔨 Compiling C++ code...")
    os.makedirs("tools", exist_ok=True)
    cmd = f"g++ -O3 -o {EXE_FILE} {SRC_FILE}"
    ret = os.system(cmd)
    if ret != 0:
        print("❌ Compilation Failed!")
        sys.exit(1)
    print("✅ Compilation Successful.\n")

def run_test(filename):
    data_path = os.path.join(DATA_DIR, filename)
    
    if not os.path.exists(data_path):
        print(f"⚠️  Skipping {filename}: File not found in {DATA_DIR}")
        return False

    output_filename = f"{filename}_out.txt"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    print(f"🔹 Testing {filename} (MinSup: {MIN_SUP_RATE*100}%)")
    
    start_time = time.time()
    cmd_hamm = f"./{EXE_FILE} {MIN_SUP_RATE} {data_path} {output_path}"
    
    process = subprocess.run(cmd_hamm, shell=True, capture_output=True, text=True)
    
    if process.returncode != 0:
        print(f"❌ C++ Runtime Error:\n{process.stderr}")
        return False
    
    cpp_time = time.time() - start_time
    print(f"   C++ Finished in {cpp_time:.4f}s")

    cmd_verify = f"python3 verify.py {data_path} {output_path}"
    
    verify_process = subprocess.run(cmd_verify, shell=True, capture_output=True, text=True)
    
    output_log = verify_process.stdout
    if "PASSED" in output_log:
        print(f"   ✅ VERIFY PASSED!")
        return True
    else:
        print(f"   ❌ VERIFY FAILED!")
        print("   --- Verify Log (First 10 lines) ---")
        print("\n".join(output_log.split('\n')[:10]))
        print("   -----------------------------------")
        return False

def main():
    compile_cpp()
    
    results = {}
    print("🚀 Starting Batch Verification...\n" + "="*40)
    
    for ds in DATASETS:
        success = run_test(ds)
        results[ds] = "PASS" if success else "FAIL"
        print("-" * 40)

    print("\n📊 Final Summary:")
    all_pass = True
    for ds, status in results.items():
        icon = "✅" if status == "PASS" else "❌"
        print(f"{icon} {ds}: {status}")
        if status == "FAIL": all_pass = False
    
    if all_pass:
        print("\n🎉 CONGRATULATIONS! All datasets passed verification.")
    else:
        print("\n⚠️  Some tests failed. Please check the logs.")

if __name__ == "__main__":
    main()