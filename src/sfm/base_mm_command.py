import subprocess


class base_mm_command():

    def __init__(self):
        pass


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
            assert r_arg in m_args, f"{r_arg} is a required argument"

        # check if only allowed arguments were used
        for arg in m_args:
            assert arg in allowed_args, f"{arg} is not an allowed argument"
