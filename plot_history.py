from matplotlib import pyplot as plt


def plot_history(history,
                 save_graph_img_path,
                 fig_size_width,
                 fig_size_height,
                 lim_font_size):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(len(acc))

    # グラフ表示
    plt.figure(figsize=(fig_size_width, fig_size_height))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = lim_font_size
    plt.subplot(121)

    # plot accuracy values
    plt.plot(epochs, acc, color="blue", linestyle="solid", label="train acc")
    plt.plot(epochs, val_acc, color="green", linestyle="solid", label="valid acc")
    plt.title("Training and Validation acc")
    plt.grid()
    plt.legend()

    # plot loss values
    # plt.subplot(122)
    plt.plot(epochs, loss, color="red", linestyle="solid", label="train loss")
    plt.plot(epochs, val_loss, color="orange", linestyle="solid", label="valid loss")
    plt.title("Training and Validation loss")
    plt.legend()
    plt.grid()

    plt.savefig(save_graph_img_path)
    plt.close()