import os

# Input and output
INPUT_FILE = "combined_text_total.txt"
OUTPUT_DIR = "chunks"
CHUNK_SIZE = 3 * 1024 * 1024  # 10 MB per chunk


def split_file(file_path, out_dir, max_chunk_size):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Read the whole file into memory once
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()

    total_size = len(data.encode("utf-8"))
    print(f"Total file size in memory: {total_size/1024/1024:.2f} MB")

    # Write out chunks
    start = 0
    chunk_index = 0
    while start < len(data):
        end = start + max_chunk_size
        chunk_data = data[start:end]

        chunk_path = os.path.join(out_dir, f"chunk_{chunk_index}.txt")
        with open(chunk_path, "w", encoding="utf-8") as chunk_file:
            chunk_file.write(chunk_data)

        print(f"Saved {chunk_path} ({len(chunk_data.encode('utf-8'))/1024/1024:.2f} MB)")
        start = end
        chunk_index += 1


def run():
    split_file(INPUT_FILE, OUTPUT_DIR, CHUNK_SIZE)


if __name__ == "__main__":
    run()
