import os


ids = ['CA035131L0077', 'CA135431L0337', 'CA135431L0343', 'CA135431L0352', 'CA135432V0337', 'CA135433R0343', 'CA135433R0350', 'CA135631L0036', 'CA135632V0031', 'CA135632V0032', 'CA135633R0037', 'CA139132V0154', 'CA168431L0207', 'CA172032V0190', 'CA172733R0183', 'CA179231L0038', 'CA180031L0060', 'CA180031L0079', 'CA181331L0123', 'CA181332V0125', 'CA181333R0125', 'CA182033R0051', 'CA182431L0059', 'CA182433R0047', 'CA182433R0050', 'CA182933R0037', 'CA183032V0009', 'CA183431L0012', 'CA183432V0005', 'CA183432V0034', 'CA183432V0041', 'CA183432V0045', 'CA183433R0044', 'CA183531L0087', 'CA183532V0060', 'CA183532V0067', 'CA183533R0058', 'CA184332V0060', 'CA184333R0078', 'CA184431L0143', 'CA184432V0094', 'CA184432V0105', 'CA184432V0113', 'CA184432V0115', 'CA184432V0154', 'CA184531L0226', 'CA184532V0199', 'CA184532V0201', 'CA184532V0219', 'CA184532V0229', 'CA184532V0231', 'CA184533R0206', 'CA184533R0229', 'CA184533R0238', 'CA184733R0095', 'CA212333R0050', 'CA213731L0035', 'CA213731L0038', 'CA213733R0050', 'CA214732V0011', 'CA214831L0099', 'CA214832V0090', 'CA214833R0100', 'CA214932V0146', 'CA215032V0257', 'CA215131L0274', 'CA215131L0288', 'CA215132V0275', 'CA215331L0411', 'CA215333R0402', 'CA215731L0063', 'CA216631L0328', 'CA216632V0331', 'CA216633R0325', 'CA216633R0332', 'CA216731L0333', 'CA216733R0338', 'CA216733R0346', 'CA216733R0367', 'CA512933R0013']

for id in ids:
    fld = "downloaded"

    path = f"/home/fdahle/SFTP/staff-umbrella/ATM/data_1/aerial/TMA/{fld}/{id}.tif"
    path1 = f"/data_1/ATM/data_1/aerial/TMA/{fld}/{id}.tif"

    print(path)
    print(path1)

    test = os.path.isfile(path)
    test1 = os.path.isfile(path1)

    print(test, test1)

    import shutil
    if test is True and test1 is False:
        shutil.copyfile(path, path1)
        print("FINISHED")
    else:
        print("NO NEED")