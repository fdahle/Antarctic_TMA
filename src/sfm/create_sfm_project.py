import os.path
import subprocess


class SFMProject(object):
    def __init__(self, project_name, fld):
        self.project_name = project_name
        self.project_folder = fld + "/" + project_name

        self.overwrite = False

    def start(self):
        pass

    def _create_project_structure(self):

        if os.path.isdir(self.project_folder) and not self.overwrite:
            raise Exception("Project folder already exists")


    def _execute_command(self):

