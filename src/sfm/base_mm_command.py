import subprocess


class base_mm_command():

    def __init__(self):
        pass

    def execute_shell_string(self, shell_string):
        with subprocess.Popen(["/bin/bash", "-i", "-c", shell_string],
                              cwd=project_folder,
                              stdout=subprocess.PIPE,
                              universal_newlines=True
                              ) as po:

    def clean_up_files(self):
        pass