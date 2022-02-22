import subprocess
from tqdm import tqdm
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
    return subprocess.run(command, capture_output=True)


def build_source(source):
    """Builds the source file

    Args:
        source (str): The source file
    """
    run_command(f"nvcc {source} -o a.out -m32")


def single_measure(size, block_size, shared_block_size, task_size):
    """Measure the execution time for one run of the program with given parameters

    Args:
        source (str): the source cuda file to execute
        size (int): the size of the array to sort
        block_size (int): the number of threads in a block
        shared_block_size (int): the number of threads in a block with shared memory
        task_size (int): the number of elements every thread starts the algorithm with

    Returns:
        float: The execution time
    """
    command = f"a.out {size} {block_size}"
    if shared_block_size is not None:
        command += f" {shared_block_size}"
    elif task_size is not None:
        command += f" {task_size}"
    return float(
        run_command(command)
        .stdout.decode()
    )


def avg_measure(repetitions, size, block_size, shared_block_size=None, task_size=None):
    """Measure the average execution time and the standard deviation for repetitions runs of the program with given parameters

    Args:
        repetitions (int): the number of times to execute the measurement.
        size (int): the size of the array to sort
        block_size (int): the number of threads in a block with global memory
        shared_block_size (int, optional): the number of threads in a block with shared memory
        task_size (int, optional): the number of elements every thread starts the algorithm with

    Returns:
        (float, float): (avg_time, std_time)

        Attributes:
            avg_time: the average time
            std_time: the standard deviation 
    """
    measures = []
    for _ in tqdm(range(repetitions)):
        measures.append(single_measure(size, block_size, shared_block_size, task_size))

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
    SHARED_BLOCK_SIZES = [1, 4, 16, 256, 1024]
    BLOCK_SIZES = [1, 8, 16, 32, 256, 1024]
    TASK_SIZES = [2, 4]
    SOURCES = ['mergesort.cu', 'mergesort_shared.cu']
    N_REPETITIONS = 10
    MAX_SHARED_SIZE = 4096
    MEASURES_DIR = 'measures'
    fieldnames = {
        "mergesort.cu": ['block_size', 'task_size', 'grid_size', 'time', 'stdev'],
        'mergesort_shared.cu': ['block_size', 'shared_block_size', 'shared_task_size', 'grid_size', 'time', 'stdev']
    }

    # Deletes every old measure (graphs and tables included)
    shutil.rmtree(MEASURES_DIR, ignore_errors=True)
    for source in SOURCES:
        build_source(source)
        for size in SIZES:
            rows, path = [], f"{MEASURES_DIR}/{source[:-3]}/SIZE_{size}.csv"
            for block_size in BLOCK_SIZES:
                if source == "mergesort.cu":
                    for task_size in sorted(list(set(TASK_SIZES+[size >> 10]))):
                        avg_time, std_time = avg_measure(N_REPETITIONS, size, block_size, task_size=task_size)
                        grid_size = size / task_size / block_size
                        print(f"SIZE {size}, BLOCK_SIZE {block_size}, GRID_SIZE {grid_size}, {task_size} as task size. AVG_TIME: {avg_time}, STD_TIME {std_time}\n")
                        rows.append([block_size, task_size, grid_size, avg_time, std_time])
                else:  # mergesort_shared.cu
                    for shared_block_size in SHARED_BLOCK_SIZES:
                        shared_task_size = MAX_SHARED_SIZE / shared_block_size
                        avg_time, std_time = avg_measure(N_REPETITIONS, size, block_size, shared_block_size=shared_block_size)
                        grid_size = size / MAX_SHARED_SIZE
                        print(f"SIZE {size}, SHARED_BLOCK_SIZE {shared_block_size}, BLOCK_SIZE {block_size}, GRID_SIZE {grid_size}, {shared_task_size} as task size. AVG_TIME: {avg_time}, STD_TIME {std_time}\n")
                        rows.append([block_size, shared_block_size, shared_task_size, grid_size, avg_time, std_time])

            # Save results in csv file
            write_table(fieldnames[source], rows, path)


if __name__ == "__main__":
    main()
