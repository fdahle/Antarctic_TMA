import subprocess


class BaseCommand():

    # specifications defined in the child class
    required_args = []
    allowed_args = []

    def __init__(self, project_folder):

        # the path to the project folder
        self.project_folder = project_folder

        # arguments of this command
        self.args = {}

    def execute_shell_cmd(self, shell_string):
        with subprocess.Popen(["/bin/bash", "-i", "-c", shell_string],
                              cwd=project_folder,
                              stdout=subprocess.PIPE,
                              universal_newlines=True
                              ) as po:
            pass

    def clean_up_files(self):
        pass

    def validate_args(self):

        # check if we have the required arguments
        for r_arg in self.required_args:
            assert r_arg in self.args, f"{r_arg} is a required argument"

        # check if only allowed arguments were used
        for arg in self.args:
            assert arg in self.allowed_args, f"{arg} is not an allowed argument"
