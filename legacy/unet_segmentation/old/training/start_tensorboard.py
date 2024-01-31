from tensorboard import program
import webbrowser

tracking_address = "../../../../data/models/segmentation/UNET/logs"

if __name__ == "__main__":
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    url = tb.launch()
    tb.main()
    print(f"Tensorflow listening on {url}")
    webbrowser.open(url, new=1, autoraise=True)
