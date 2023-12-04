import json
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def display_single_comparison():

    # dice
    # models_to_compare = ["model_loss_dice", "test_dice_no_augment"]
    # model_names = ["Normal", "Test"]
    # title="Test"

    # overfitting
    # models_to_compare = ["model_overfitting_none", "model_overfitting_drop", "model_overfitting_batch",
    #                     "model_baseline"]
    # model_names = ["None", "Dropout", "Batch normalization", "Both"]
    # title = "Results for additional parameters"

    # learning rate
    #models_to_compare = ["model_learning_rate_01_2", "model_learning_rate_001",
    #                     "model_baseline"]
    #model_names = ["0.1", "0.01", "0.001"]
    #title = "Results for learning rate"

    # model depth
    # models_to_compare = ["model_depth_smaller", "model_depth_small", "model_baseline", "model_depth_large"]
    # model_names = ["2 layers", "3 layers", "4 layers", "5 layers"]
    # title="Results for model depth"

    # loss (with augmentation)
    # models_to_compare = ["model_baseline", "model_loss_focal", "model_loss_dice"]
    # model_names = ["cross-entropy", "focal", "dice",]
    # title="Results for losses"

    # augmentation
    # models_to_compare = ["model_baseline", "model_augmentation_flipped", "model_augmentation_rotation",
    #                     "model_augmentation_brightness", "model_augmentation_noise",
    #                     "model_augmentation_normalize", "model_augmentation_all"]
    # model_names = ["None", "Flipped", "Rotation", "Brightness", "Noise", "Normalize", "All"]
    # title = "Results for augmentation"

    # input size
    # models_to_compare = ["model_baseline", "test_cross_cropped"]
    # model_names = ["Resized", "Cropped"]
    # title = "Results for input size"

    # base model
    models_to_compare = ["model_baseline"]
    model_names = ["Base model"]
    title="Result for base model"

    # some basic settings for the models
    path_models = "/data_1/ATM/data_1/machine_learning/semantic_segmentation/models_paper2"
    stats = ["loss", "acc", "precision", "recall", "f1"]  # , "kappa"]
    stat_titles = ["Loss", "Accuracy", "precision", "recall", "Precision, Recall, F1-score"]
    max_epoch = 500

    # here we store all data
    all_data = {}

    # define the colors of the lines
    colors = ["#ffe119", "#4363d8", "#f58231", "#dcbeff", "#800000", "#000075", "#a9a9a9", "#006400"]
    colors = ["red", "green", "blue"]

    # here we get all data
    for j, model in enumerate(models_to_compare):

        # open json file and get the statistics
        json_file = open(path_models + "/" + model + ".json")
        json_data = json.load(json_file)
        json_statistics = json_data["statistics"]

        # iterate all stats in the json-file
        for stat in stats:

            # distinguish between train and val
            for _iter in ["train", "val"]:

                print("Load", model, stat, _iter)

                data_stat_iter = []
                for i in range(max_epoch):
                    data_stat_iter.append(json_statistics[_iter + "_" + stat][str(i+1)])

                all_data[stat + "_" + _iter + "_" + model_names[j]] = data_stat_iter

    x_plots = 3  # 2

    a_x, a_y = 0, 0
    fig, ax = plt.subplots(1, x_plots)

    col_id = 0

    for i, stat in enumerate(stats):

        for key in all_data:
            if key.startswith(stat) is False:
                continue

            print(key, a_x, a_y)

            iteration = key.split("_")[1]
            name_model = key.split("_")[-1]
            col = colors[col_id]

            if i == 2:
                name_model = "Precision"
            if i == 3:
                name_model = "Recall"
            if i == 4:
                name_model = "F1-Score"

            y = all_data[key]
            x = list(range(len(all_data[key])))

            # create trend-line for train
            z_train = np.polyfit(x, y, 25)
            p_train = np.poly1d(z_train)
            if iteration == "train":
                # ax[a_x][a_y].plot(x, y, "--", color=col)
                ax[a_y].plot(x, p_train(x), "--", color=col)
            else:
                # ax[a_x][a_y].plot(x, y, color=col)
                ax[a_y].plot(x, p_train(x), label=name_model, color=col)

        ratio = 1/1.3

        x_left, x_right = ax[a_y].get_xlim()
        y_low, y_high = ax[a_y].get_ylim()
        ax[a_y].set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

        if stat_titles[i] == "Precision, Recall, F1-score":
            ax[a_y].legend(loc="upper left")
        ax[a_y].title.set_text(stat_titles[i])

        a_y += 1
        #if a_y == x_plots:
        #    a_x += 1
        #   a_y = 0
        print("TEEMP")
        if a_y == x_plots:
            a_y -= 1
            col_id += 1

    import matplotlib.lines as mlines
    black_line = mlines.Line2D([], [], color='black', label='Validation data')
    black_line_dashed = mlines.Line2D([], [], color='black', linestyle="--", label='Training data')

    # fig.suptitle(title, y=0.92)
    fig.legend(handles=[black_line, black_line_dashed], loc="upper right", bbox_to_anchor=(0.9, 0.76))
    plt.show()


def display_baseline_classes():

    # base model
    model = "model_baseline_per_class"
    model_name = "Base model"
    title = "Result for base model"

    # classes
    class_names = ["ice", "snow", "rocks", "water", "clouds", "sky", "unknown"]

    # some basic settings for the models
    path_models = "/data_1/ATM/data_1/machine_learning/semantic_segmentation/models_revision"
    stats = ["precision", "recall", "f1"]  # , "kappa"]
    stat_titles = ["Accuracy", "precision", "recall", "F1"]
    max_epoch = 500


    # open json file and get the statistics
    json_file = open(path_models + "/" + model + ".json")
    json_data = json.load(json_file)
    json_statistics = json_data["statistics"]

    all_data = {}

    # iterate the immportant stats
    for stat in stats:

        stat = stat + "_class"

        # distinguish between train and val
        for _iter in ["train", "val"]:
            print("Load", model, stat, _iter)

            data_stat_iter = []
            for i in range(max_epoch):
                data_stat_iter.append(json_statistics[_iter + "_" + stat][str(i + 1)])

            all_data[stat + "_" + _iter] = data_stat_iter

    print(all_data)

    # Define colors
    colors = {"precision": "green", "recall": "red", "f1": "blue"}

    # Plotting
    fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharey=True)

    for class_idx, class_name in enumerate(class_names[:-1]):
        ax = axs[class_idx // 3, class_idx % 3]  # Select subplot
        for stat in stats:
            color = colors[stat]
            # Extract per-epoch statistics for the current class
            train_data = [epoch_stat[class_idx] for epoch_stat in all_data[stat + "_class_train"]]
            val_data = [epoch_stat[class_idx] for epoch_stat in all_data[stat + "_class_val"]]

            # Plot actual data (lightly)
            ax.plot(train_data, alpha=0.1, linestyle='-', color=color)
            ax.plot(val_data, alpha=0.1, linestyle='--', color=color)

            # Smooth data with a polynomial fit and plot
            train_poly = np.polyfit(range(max_epoch), train_data, 5)
            val_poly = np.polyfit(range(max_epoch), val_data, 5)
            ax.plot(np.polyval(train_poly, range(max_epoch)), linestyle='-', color=color)
            ax.plot(np.polyval(val_poly, range(max_epoch)), linestyle='--', color=color)

        ax.set_title(class_name)
        ax.set_xlabel('Epoch')

        # Create legend for each subplot
        import matplotlib.lines as mlines
        subplot_handles = [mlines.Line2D([], [], color=colors[stat], label=stat.capitalize()) for stat in stats]
        ax.legend(handles=subplot_handles, loc='upper left')

    # Create custom legends
    black_line = mlines.Line2D([], [], color='black', label='Validation data')
    black_line_dashed = mlines.Line2D([], [], color='black', linestyle="--", label='Training data')

    # fig.suptitle(title, y=0.92)
    fig.legend(handles=[black_line, black_line_dashed], loc="upper right", bbox_to_anchor=(0.99, 0.95))

    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.ylabel("Score")
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def display_matrix():

    color_mode = "colors"  # can be 'colors' or 'best_worst'

    # define the colors of the lines
    colors = ["#ffe119", "#4363d8", "#f58231", "#dcbeff", "#800000", "#000075", "#a9a9a9", "#006400"]

    # additional components
    models_to_compare_ac = ["model_overfitting_none", "model_overfitting_drop", "model_overfitting_batch",
                         "model_baseline"]
    model_names_ac = ["None", "Dropout", "Batch normalization", "Both"]

    # learning rate
    #models_to_compare_lr = ["model_learning_rate_01", "model_learning_rate_001",
    #                        "model_baseline"]
    #model_names_lr = ["0.1", "0.01", "0.001"]
    models_to_compare_lr = ["model_learning_rate_01", "model_learning_rate_001",
                            "model_baseline", "model_learning_rate_00001"]
    model_names_lr = ["0.1", "0.01", "0.001", "0.0001"]

                            # loss (with augmentation)
    models_to_compare_lo = ["model_baseline", "model_loss_focal", "model_loss_dice"]
    model_names_lo = ["cross-entropy", "focal", "dice"]

    # model depth
    models_to_compare_md = ["model_depth_smaller", "model_depth_small", "model_baseline", "model_depth_large"]
    model_names_md = ["2 layers", "3 layers", "4 layers", "5 layers"]

    # input size
    models_to_compare_is = ["model_baseline", "test_cross_cropped"]
    model_names_is = ["Resized", "Cropped"]

    # augmentation
    models_to_compare_a = ["model_baseline", "model_augmentation_flipped", "model_augmentation_rotation",
                           "model_augmentation_brightness", "model_augmentation_noise",
                           "model_augmentation_normalize", "model_augmentation_all"]
    model_names_a = ["None", "Flipped", "Rotation", "Brightness", "Noise", "Normalize", "All"]
    #models_to_compare_a = ["model_baseline", "model_augmentation_all", "model_augmentation_all_light"]
    #model_names_a = ["None", "All", "All-light"]

    path_models = "/data_1/ATM/data_1/machine_learning/semantic_segmentation/models_paper2"
    stats = ["loss", "acc", "f1"]  # , "kappa"]
    stat_titles = ["loss", "accuracy", "F1-score"]
    max_epoch = 500

    all_models = [models_to_compare_ac, models_to_compare_lr, models_to_compare_lo,
                  models_to_compare_md, models_to_compare_is, models_to_compare_a]
    all_model_names = [model_names_ac, model_names_lr, model_names_lo,
                       model_names_md, model_names_is, model_names_a]

    #all_models = [models_to_compare_ac, models_to_compare_a]
    #all_model_names = [model_names_ac, model_names_a]

    all_data = []
    best_models = []
    worst_models = []
    for i, elem in enumerate(all_models):
        best_model = [-1, -1, -1]
        worst_model = [-1, -1, -1]
        best_values = [100, 0, 0]
        worst_values = [0, 100, 100]

        for j, model in enumerate(elem):

            print(f"Load {model}")

            json_file = open(path_models + "/" + model + ".json")
            json_data = json.load(json_file)

            json_statistics = json_data["statistics"]

            _data = {}

            for stat in stats:

                for _iter in ["train", "val"]:

                    data_stat_iter = []
                    for k in range(max_epoch):
                        data_stat_iter.append(json_statistics[_iter + "_" + stat][str(k + 1)])

                    if _iter == "val":
                        json_value = json_statistics[_iter + "_" + stat][str(max_epoch)]
                        if stat == "loss":
                            if json_value < best_values[0]:
                                best_values[0] = json_value
                                best_model[0] = model
                            if json_value > worst_values[0]:
                                worst_values[0] = json_value
                                worst_model[0] = model
                        if stat == "acc":
                            if json_value > best_values[1]:
                                best_values[1] = json_value
                                best_model[1] = model
                            if json_value < worst_values[1]:
                                worst_values[1] = json_value
                                worst_model[1] = model
                        if stat == "f1":
                            if json_value > best_values[2]:
                                best_values[2] = json_value
                                best_model[2] = model
                            if json_value < worst_values[2]:
                                worst_values[2] = json_value
                                worst_model[2] = model

                    if model == "model_loss_focal" and stat == "loss":
                        print(data_stat_iter[0:10])
                        data_stat_iter = [x * 10 for x in data_stat_iter]
                        print(data_stat_iter[0:10])

                    _data[stat + "_" + _iter + "_" + elem[j]] = data_stat_iter

            all_data.append(_data)

        best_models.append(best_model)
        worst_models.append(worst_model)

    fig, ax = plt.subplots(len(all_models), 3)

    counter = 0
    for y_pos in range(len(all_models)):

        models = all_models[y_pos]
        model_names = all_model_names[y_pos]
        best_model = best_models[y_pos]
        worst_model = worst_models[y_pos]

        for x_pos in range(len(model_names)):

            model_data = all_data[counter]
            counter += 1

            model = models[x_pos]
            model_name = model_names[x_pos]

            for z_pos in range(3):
                if z_pos == 0:
                    ptr = "loss"
                elif z_pos == 1:
                    ptr = "acc"
                elif z_pos == 2:
                    ptr = "f1"

                if color_mode == "best_worst":
                    if model == best_model[z_pos]:
                        col = "green"
                    elif model == worst_model[z_pos]:
                        col = "red"
                    else:
                        if model == "model_baseline":
                            col = "gray"
                        else:
                            col = "black"
                elif color_mode == "colors":
                    col = colors[x_pos]

                y_tr = model_data[ptr + "_train_" + model]
                x_tr = list(range(len(y_tr)))
                y_val = model_data[ptr + "_val_" + model]
                x_val = list(range(len(y_val)))

                z_tr = np.polyfit(x_tr, y_tr, 15)
                p_tr = np.poly1d(z_tr)
                z_val = np.polyfit(x_val, y_val, 15)
                p_val = np.poly1d(z_val)

                if model_name == "focal":
                    model_name = "focal (x10)"
                if col in ["red", "green"]:
                    ax[y_pos, z_pos].plot(x_tr, p_tr(x_tr), "--", color=col, zorder=1)
                    ax[y_pos, z_pos].plot(x_val, p_val(x_val), color=col, label=model_name, zorder=1)
                else:
                    ax[y_pos, z_pos].plot(x_tr, p_tr(x_tr), "--", color=col)
                    ax[y_pos, z_pos].plot(x_val, p_val(x_val), color=col, label=model_name, zorder=0)

                # create the legend
                leg = ax[y_pos, z_pos].legend(loc="upper left", prop={"size": 6})
                leg.set_zorder(2)
                leg.get_frame().set_facecolor('w')
                leg.get_frame().set_fill(True)

                if z_pos == 0:
                    ax[y_pos, z_pos].set_ylim(bottom=0, top=2.5)
                elif z_pos >= 1:
                    ax[y_pos, z_pos].set_ylim(bottom=0.1, top=0.75)

    import matplotlib.lines as mlines
    black_line = mlines.Line2D([], [], color='black', label='Validation data')
    black_line_dashed = mlines.Line2D([], [], color='black', linestyle="--", label='Training data')

    # fig.suptitle(title, y=0.92)
    fig.legend(handles=[black_line, black_line_dashed], loc="upper right", bbox_to_anchor=(0.9, 0.96))

    ax[0, 0].set_xlabel("loss")
    ax[0, 1].set_xlabel("accuracy")
    ax[0, 2].set_xlabel("F1-score")

    ax[0, 0].xaxis.set_label_position("top")
    ax[0, 1].xaxis.set_label_position("top")
    ax[0, 2].xaxis.set_label_position("top")

    ax[0, 0].set_ylabel("additional\ncomponents")
    #ax[1, 0].set_ylabel("learning\nrate")
    #ax[2, 0].set_ylabel("loss type")
    #ax[3, 0].set_ylabel("model depth")
    #ax[4, 0].set_ylabel("input size")
    #ax[5, 0].set_ylabel("augmentation")
    ax[1, 0].set_ylabel("additional\ncomponents")

    fig.set_size_inches(8.27, 11.69)

    plt.savefig("/data_1/ATM/test.png", dpi=600, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":

    display_baseline_classes()

    #display_matrix()
