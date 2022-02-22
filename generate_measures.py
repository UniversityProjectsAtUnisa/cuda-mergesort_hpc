import subprocess
from tqdm import tqdm
from generate_file import generate_file
import statistics as stats
from pathlib import Path
import csv
import shutil


def run_command(command):
    """Runs a command in bash synchronously

    Args:
        command (str): Command to run

    Returns:
        CompletedProcess: A process that has finished running.

        Attributes:
            args: The list or str args passed to run(). (["bash", "-c", command])
            returncode: The exit code of the process, negative for signals.
            stdout: The standard output
            stderr: The standard error
    """
    commands = ["bash", "-c", command]
    return subprocess.run(commands, capture_output=True)

def single_measure(source, size, block_size, task_size):
    """Measure the execution time for one run of the program with given parameters

    Args:
        source (str): the source cuda file to execute
        size (int): the size of the array to sort
        block_size (int): the number of threads in a block
        task_size (int): the number of elements every thread starts the algorithm with.

    Returns:
        float: The execution time
    """
    return float(
        run_command(
            f"nvcc {source} -o a.out -m=32 && a.out {size} {block_size} {task_size}")
        .stdout.decode()
    )

def avg_measure(source, size, block_size, task_size, repetitions):
    """Measure the average execution time and the standard deviation for repetitions runs of the program with given parameters

    Args:
        source (str): the source cuda file to execute
        size (int): the size of the array to sort
        block_size (int): the number of threads in a block
        task_size (int): the number of elements every thread starts the algorithm with
        repetitions (int): the number of times to execute the measurement.

    Returns:
        (float, float): (avg_time, std_time)

        Attributes:
            avg_time: the average time
            std_time: the standard deviation 
    """
    measures = []
    for _ in tqdm(range(repetitions)):
        measures.append(single_measure(source, size, block_size, task_size))

    return round(stats.fmean(measures), 4), round(stats.stdev(measures), 4)

def write_table(header, rows, path):
    """Exports the data in header and rows in a csv file in path location

    Args:
        header (list of str): the names of the columns
        rows (list of list): the data to be exported
        path (str): the path for the output csv file
    """

    # Creates the parent folder if doesn't exist
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as csvfile:
        # Uses semicolon delimiter for Microsoft Excel compatibility
        writer = csv.writer(csvfile, delimiter=";")
        writer.writerow(header)
        writer.writerows(rows)

def main():
    SIZES = [1 << a for a in [12, 14, 16, 18, 20]]
    BLOCK_SIZES = [8, 16, 32]
    TASK_SIZES = [2, 4]
    SOURCES = ['mergesort.cu', 'mergesort_shared.cu']
    N_REPETITIONS = 10
    MEASURES_DIR = 'measures'
    fieldnames = ['block_size', 'task_size', 'grid_size', 'time', 'stdev']

    # Deletes every old measure (graphs and tables included)
    shutil.rmtree(MEASURES_DIR, ignore_errors=True)
    for source, size in zip(SOURCES, SIZES):
        rows, path = [], f"{MEASURES_DIR}/{source[:-3]}/SIZE_{size}.csv"
        for block_size, task_size in zip(BLOCK_SIZES, TASK_SIZES):
            avg_time, std_time = avg_measure(source, size, block_size, task_size, N_REPETITIONS)
            grid_size = size / task_size / block_size
            print(f"SIZE {size}, {block_size} threads per block, {grid_size} blocks in a grid, {task_size} as task size. AVG_TIME: {avg_time}, STD_TIME {std_time}\n")
            rows.append([block_size, task_size, grid_size, avg_time, std_time])

        # Save results in csv file
        write_table(fieldnames, rows, path)


if __name__ == "__main__":
    main()