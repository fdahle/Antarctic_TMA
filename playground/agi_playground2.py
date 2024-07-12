import Metashape
import src.base.load_credentials as lc

# get the license key
licence_key = lc.load_credentials("agisoft")['licence']

# Activate the license
Metashape.License().activate(licence_key)


path_relative_psx = "/data/ATM/data_1/sfm/agi_projects/test_gcps4/test_gcps4_relative.psx"
path_absolute_psx = "/data/ATM/data_1/sfm/agi_projects/test_gcps4/test_gcps4_absolute.psx"

doc_relative = Metashape.Document()
doc_relative.open(path_relative_psx)
chunk_relative = doc_relative.chunks[0]

doc_absolute = Metashape.Document()
doc_absolute.open(path_absolute_psx)
chunk_absolute = doc_absolute.chunks[0]

print(chunk_relative.transform.matrix)
print(chunk_absolute.transform.matrix)

chunk_absolute.
