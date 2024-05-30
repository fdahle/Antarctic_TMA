import glob
import json
import os.path
import subprocess
import shutil

from abc import ABC, abstractmethod
# from datetime import datetime


class BaseCommand(ABC):
    # specifications defined in the child class
    required_args = []
    allowed_args = []
    additional_args = []

    def __init__(self, project_folder,
                 mm_args,
                 command_name,
                 print_all_output=False,
                 save_stats=False,
                 save_raw=False,
                 clean_up=True,
                 auto_enter=False,
                 debug=False,
                 overwrite=False):

        self.project_folder = project_folder
        self.mm_args = mm_args
        self.command_name = command_name

        # does this command has additional arguments?
        if len(self.additional_args) > 0:
            self.extend_additional_args()

        # save the optional arguments
        self.print_all_output = print_all_output
        self.save_stats = save_stats
        self.save_raw = save_raw
        self.clean_up = clean_up
        self.auto_enter = auto_enter
        self.debug = debug
        self.overwrite = overwrite

    @abstractmethod
    def build_shell_dict(self):
        pass

    @abstractmethod
    def before_execution(self):
        pass

    @abstractmethod
    def after_execution(self):
        pass

    def execute_shell_cmd(self, mm_path=None):

        # Get the current date and time
        # start_time = datetime.now()

        # execute tasks that need to be done before the shell command
        self.before_execution()

        # validate the required files
        self.validate_required_files()

        # build the shell command(s)
        shell_dict = self.build_shell_dict()

        # iterate shell dict
        for key, shell_string in shell_dict.items():

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
                                  stderr=subprocess.PIPE,
                                  universal_newlines=True
                                  ) as p:

                for stdout_line in p.stdout:

                    if self.print_all_output:
                        print(stdout_line, end="")

                    raw_output.append(stdout_line)

                    # Check for the warning message
                    if "Warn tape enter to continue" in stdout_line and self.auto_enter:
                        p.stdin.write('\n')
                        p.stdin.flush()

            # save the raw output
            if self.save_raw:
                # Format the current date and time as desired
                # timestamp = start_time.strftime("%Y_%m_%d_%H_%M_%S")

                filename = f"{self.project_folder}/stats/" \
                           f"{self.command_name}_raw.txt"
                with open(filename, "w") as f:
                    f.write(f"{shell_string}\n")
                    f.write("************\n")

                    f.writelines(raw_output)

            # check if the command was successful
            if "Sorry, the following FATAL ERROR happened" in str(raw_output):

                error_obj = self._handle_error(raw_output)

                raise MicMacError(key, error_obj)

            else:
                print(f"{key} finished successfully.")

            if self.save_stats:
                self.extract_stats(self.command_name, raw_output)

            if self.print_all_output:
                print("********")

        self.after_execution()

        if self.clean_up:
            self._clean_up_files()

    # method is not abstract as only prevalent in some files
    def extend_additional_args(self):
        raise AssertionError("This method should only be called in child classes")

    def _clean_up_files(self):
        """Remove the temporary files created by MicMac."""

        # define the file patterns to be removed
        file_patterns = ["Tmp-MM-Dir", "mm3d-LogFile.txt", "MM-Error*.txt", "MkStdMM*", "TestOpenMM"]

        # iterate all patterns
        for pattern in file_patterns:
            full_path = os.path.join(self.project_folder, pattern)
            if "*" in pattern:  # For wildcard patterns
                for file in glob.glob(full_path):
                    os.remove(file)
            # remove directories
            elif os.path.isdir(full_path):
                shutil.rmtree(full_path)
            # remove files
            elif os.path.isfile(full_path):
                os.remove(full_path)

    def _handle_error(self, raw_output):

        # get the error dict
        # Get the directory of the current Python file
        current_directory = os.path.dirname(os.path.abspath(__file__))
        path_error_json = os.path.join(current_directory, "_error_dict.json")
        with open(path_error_json, "r") as file:
            error_dict = json.load(file)

        # filter the error messages to this command
        error_dict = error_dict[self.__class__.__name__]

        # iterate over the raw output and find the error message
        for line in raw_output:

            # iterate over the error dict to find the equivalent error object
            for error_obj in error_dict:

                if error_obj['MMErrorMsg'] in line:
                    return error_obj

    def validate_mm_args(self):

        # check if we have the required arguments
        for r_arg in self.required_args:
            if r_arg not in self.mm_args:
                raise ValueError(f"{r_arg} is a required argument")

        # check if only allowed arguments were used
        for arg in self.mm_args:
            if arg not in self.allowed_args:
                raise ValueError(f"{arg} is not an allowed argument")

    @abstractmethod
    def validate_required_files(self):
        pass

    @abstractmethod
    def extract_stats(self, name, raw_output):
        pass


class MicMacError(Exception):
    """
    Exception raised for MicMac errors.
    """

    def __init__(self, class_name: str, error_obj: dict):
        """
        Args:
            class_name (str): The name of the class where the error occurred.
            error_obj (dict): A dictionary with keys 'MMErrorMsg', 'ErrorMsg', and 'Fix',
                providing details of the MicMac error, a general error message, and a possible fix, respectively.
        """

        # get the error details from the object
        try:
            mm_error_msg = error_obj['MMErrorMsg']
            error_msg = error_obj['ErrorMsg']
            fix = error_obj['Fix']

        # that means the error is not yet defined in the error_dict
        except (Exception,):
            mm_error_msg = "Unknown error"
            error_msg = "An unknown error occurred. Please check the raw output for more details"
            fix = ""

        # create the error message
        error_str = f"{class_name} failed with the following error: '{mm_error_msg}'. ({error_msg}). "

        # add fix if known
        if len(fix) > 0:
            error_str += f"For a (possible) fix follow these instructions: {fix}"

        # call the Exception constructor
        super().__init__(error_str)
