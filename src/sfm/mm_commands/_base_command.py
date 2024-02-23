import glob
import os.path
import subprocess
import shutil

class BaseCommand():

    # specifications defined in the child class
    required_args = []
    allowed_args = []

    def __init__(self, project_folder,
                 save_stats=True,
                 save_raw_output=True,
                 delete_temp_files=True,
                 overwrite=False):

        self.project_folder = project_folder
        self.save_stats = save_stats
        self.save_raw_output = save_raw_output
        self.delete_temp_files = delete_temp_files
        self.overwrite = overwrite

    def execute_shell_cmd(self, args, mm_path=None):

        # validate the input arguments
        self.validate_args(args)

        # validate the input parameters
        self.validate_parameters(args)

        # validate the required files
        self.validate_required_files()

        # build the shell command
        shell_string = self.build_shell_string(args)

        if mm_path is not None:
            shell_string = f"{mm_path} {shell_string}"

        print(shell_string)

        # here the raw output is saved
        raw_output = []

        with subprocess.Popen(["/bin/bash", "-c", shell_string],
                              cwd=self.project_folder,
                              stdout=subprocess.PIPE,
                              universal_newlines=True
                              ) as p:

            for stdout_line in p.stdout:

                raw_output.append(stdout_line)

            print(f"{self.__class__.__name__} finished")

        if self.save_raw_output:
            filename = f"{self.project_folder}/stats/{self.__class__.__name__}_raw_log.txt"
            with open(filename, "w") as f:
                f.writelines(raw_output)

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

    def validate_args(self, args):

        # check if we have the required arguments
        for r_arg in self.required_args:
            if r_arg not in args:
                raise ValueError(f"{r_arg} is a required argument")

        # check if only allowed arguments were used
        for arg in args:
            if arg not in self.allowed_args:
                raise ValueError(f"{arg} is not an allowed argument")
