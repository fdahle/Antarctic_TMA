from src.sfm.mm_commands._base_command import BaseCommand

class ReSampFid(BaseCommand):

    required_args = ["ImagePattern", "ScanResolution"]
    allowed_args = ["ImagePattern", "ScanResolution"]

    def __init__(self, *args, **kwargs):
        # Initialize the base class with all arguments passed to ReSampFid
        super().__init__(*args, **kwargs)

    def build_shell_string(self, args):

        # build the shell command
        shell_string = f"ReSampFid {args['ImagePattern']} {args['ScanResolution']}"
        return shell_string

    def validate_parameters(self, args):

        if args['ScanResolution'] <= 0:
            raise ValueError("ScanResolution must be greater than 0")

    def validate_required_files(self):
        pass