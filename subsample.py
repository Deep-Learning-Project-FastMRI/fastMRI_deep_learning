import os
import random
import shutil

def subsample_files(source_dir, dest_dir, subsample_fraction=0., file_extension=".h5"):
    """
    Randomly select a fraction of files from source_dir and copy them to dest_dir.
    :param source_dir: Path to the folder containing the original files.
    :param dest_dir: Path to the folder where the subsampled files will be placed.
    :param subsample_fraction: Fraction of files to be subsampled (e.g., 0.05 for 5%).
    :param file_extension: Extension of files to subsample (default: ".h5").
    """
    files = [f for f in os.listdir(source_dir) if f.endswith(file_extension)]
    subsample_count = int(len(files) * subsample_fraction)
    random.shuffle(files)

    print(len(files))
    selected_files = files[:subsample_count]
    os.makedirs(dest_dir, exist_ok=True)
    for f in selected_files:
        src = os.path.join(source_dir, f)
        dst = os.path.join(dest_dir, f)
        shutil.copy(src, dst)
        print(f"Copied {f} to {dest_dir}")

if __name__ == "__main__":
    # change the source directory
    source_directory = "multicoil_test_full"
    destination_directory = "subsampled_multicoil_test"
    subsample_files(source_directory, destination_directory, subsample_fraction=0.10)
