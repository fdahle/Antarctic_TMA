import glob
import os.path
import subprocess
import shutil

from datetime import datetime
from tqdm import tqdm

class BaseCommand():

    # specifications defined in the child class
    required_args = []
    allowed_args = []

    def __init__(self, project_folder,
                 mm_args,
                 print_all_output=False,
                 stat_folder=None,
                 raw_folder=None,
                 clean_up=True,
                 overwrite=False):

        self.project_folder = project_folder
        self.mm_args = mm_args

        # already validate the input arguments
        self.validate_mm_args()

        # save the optional arguments
        self.print_all_output = print_all_output
        self.stat_folder = stat_folder
        self.raw_folder = raw_folder
        self.clean_up = clean_up
        self.overwrite = overwrite

    def execute_shell_cmd(self, mm_path=None):

        # Get the current date and time
        start_time = datetime.now()

        # validate the required files
        self.validate_required_files()

        # build the shell command
        shell_string = self.build_shell_string()

        if mm_path is not None:
            shell_string = f"{mm_path} {shell_string}"

        # here the raw output is saved
        raw_output = []

        if self.print_all_output:
            print(shell_string)
            print("********")

        with subprocess.Popen(["/bin/bash", "-c", shell_string],
                              cwd=self.project_folder,
                              stdout=subprocess.PIPE,
                              universal_newlines=True
                              ) as p:

            for stdout_line in p.stdout:

                if self.print_all_output:
                    print(stdout_line, end="")

                raw_output.append(stdout_line)

            print(f"{self.__class__.__name__} finished")

        if self.raw_folder is not None:

            # Format the current date and time as desired
            timestamp = start_time.strftime("%Y_%m_%d_%H_%M_%S")

            filename = f"{self.project_folder}/stats/" \
                       f"{timestamp}_{self.__class__.__name__}_raw.txt"
            with open(filename, "w") as f:

                f.write(f"{shell_string}\n")
                f.write("************\n")

                f.writelines(raw_output)

        if self.clean_up:
            self.clean_up_files()

    def clean_up_files(self):

        file_patterns = ["Tmp-MM-Dir", "mm3d-LogFile.txt", "WarnApero.txt", "MM-Error*.txt", "MkStdMM*"]
        for pattern in file_patterns:
            full_path = os.path.join(self.project_folder, pattern)
            if "*" in pattern:  # For wildcard patterns
                for file in glob.glob(full_path):
                    os.remove(file)
            elif os.path.isdir(full_path):
                shutil.rmtree(full_path)
            elif os.path.isfile(full_path):
                os.remove(full_path)

    def validate_mm_args(self):

        print("Validate mm args")

        # check if we have the required arguments
        for r_arg in self.required_args:
            if r_arg not in self.mm_args:
                raise ValueError(f"{r_arg} is a required argument")

        # check if only allowed arguments were used
        for arg in self.mm_args:
            if arg not in self.allowed_args:
                raise ValueError(f"{arg} is not an allowed argument")
