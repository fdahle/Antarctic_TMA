possible_commands = [
    "AperiCloud",  # visualize relative orientation
    "Campari",  # bundle adjustment and refinement of camera
    "HomolFilterMasq",  # filtering on tie points
    "Malt",  # compute DEM
    "Nuage2Ply",  # create a point cloud
    "ReSampFid",  # resample the images
    "Schnaps",  # tie point reduction tool
    "Tapas",  # compute relative orientation
    "Tapioca",  # find tie points between the images
    "Tapioca_own",  # own methods of tie point matching
    "Tarama",  # create a pseudo ortho-image
    "Tawny",  # create an ortho-image
]


def execute_mm_cmd(command, args, project_fld,
                   image_ids=None,
                   save_stats=False, stats_folder="",
                   delete_temp_files=False,
                   print_output=True, print_orig_errors=True):
    # check if we can use ths command
    assert command in possible_commands, f"{command} is not a valid command."

    if command == "AperiCloud":
        import sfm_modelling.mm_commands.apericaloud as ape

        ape.apericloud(project_fld, m_args=args,
                       save_stats=save_stats, stats_folder=stats_folder,
                       delete_temp_files=delete_temp_files,
                       print_output=print_output, print_orig_errors=print_orig_errors)

    if command == "Campari":
        import sfm_modelling.mm_commands.campari as cam
        cam.campari(project_fld, m_args=args,
                    save_stats=save_stats, stats_folder=stats_folder,
                    delete_temp_files=delete_temp_files,
                    print_output=print_output, print_orig_errors=print_orig_errors)

    if command == "HomolFilterMasq":
        import sfm_modelling.mm_commands.homolfiltermasq as hom
        hom.homolfiltermasq(project_fld, m_args=args,
                            save_stats=save_stats, stats_folder=stats_folder,
                            delete_temp_files=delete_temp_files,
                            print_output=print_output, print_orig_errors=print_orig_errors)

    if command == "Malt":
        import sfm_modelling.mm_commands.malt as mal
        mal.malt(project_fld, m_args=args,
                 save_stats=save_stats, stats_folder=stats_folder,
                 delete_temp_files=delete_temp_files,
                 print_output=print_output, print_orig_errors=print_orig_errors)

    if command == "Nuage2Ply":
        import sfm_modelling.mm_commands.nuage2ply as nua
        nua.nuage2ply(project_fld, m_args=args,
                      save_stats=save_stats, stats_folder=stats_folder,
                      delete_temp_files=delete_temp_files,
                      print_output=print_output, print_orig_errors=print_orig_errors)

    if command == "ReSampFid":
        import sfm_modelling.mm_commands.resampfid as res
        res.resampfid(project_fld, image_ids, m_args=args,
                      save_stats=save_stats, stats_folder=stats_folder,
                      delete_temp_files=delete_temp_files,
                      print_output=print_output, print_orig_errors=print_orig_errors)

    if command == "Schnaps":
        import sfm_modelling.mm_commands.schnaps as sch
        sch.schnaps(project_fld, m_args=args,
                    save_stats=save_stats, stats_folder=stats_folder,
                    delete_temp_files=delete_temp_files,
                    print_output=print_output, print_orig_errors=print_orig_errors)

    if command == "Tapas":
        import sfm_modelling.mm_commands.tapas as tapa
        tapa.tapas(project_fld, m_args=args,
                   save_stats=save_stats, stats_folder=stats_folder,
                   delete_temp_files=delete_temp_files,
                   print_output=print_output, print_orig_errors=print_orig_errors)

    if command == "Tapioca":
        import sfm_modelling.mm_commands.tapioca as tapi
        tapi.tapioca(project_fld, m_args=args,
                     save_stats=save_stats, stats_folder=stats_folder,
                     delete_temp_files=delete_temp_files,
                     print_output=print_output, print_orig_errors=print_orig_errors)

    # create an own tie-point structure
    if command == "Tapioca_own":
        import sfm_modelling.sub.create_tie_point_structure as ctps
        ctps.create_tie_point_structure(project_fld)

    if command == "Tarama":
        import sfm_modelling.mm_commands.tarama as tar
        tar.tarama(project_fld, m_args=args,
                   save_stats=save_stats, stats_folder=stats_folder,
                   delete_temp_files=delete_temp_files,
                   print_output=print_output, print_orig_errors=print_orig_errors)

    if command == "Tawny":
        import sfm_modelling.mm_commands.tawny as taw
        taw.tawny(project_fld, m_args=args,
                  save_stats=save_stats, stats_folder=stats_folder,
                  delete_temp_files=delete_temp_files,
                  print_output=print_output, print_orig_errors=print_orig_errors)
