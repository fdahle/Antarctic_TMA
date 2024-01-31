import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

def segment_any_image(image, path_model_folder=None, model=None):
    folder = "/data_1/ATM/data_1/playground/segment_anything"
    model = "default_model.pth"

    print(image.shape)

    print("step 1")
    sam = sam_model_registry["default"](checkpoint=folder + "/" + model)
    print("step 2")
    mask_generator = SamAutomaticMaskGenerator(sam)
    print("step 3")
    masks = mask_generator.generate(image)

    def show_anns(anns):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)
        polygons = []
        color = []
        for ann in sorted_anns:
            m = ann['segmentation']
            img = np.ones((m.shape[0], m.shape[1], 3))
            color_mask = np.random.random((1, 3)).tolist()[0]
            for i in range(3):
                img[:, :, i] = color_mask[i]
            ax.imshow(np.dstack((img, m * 0.35)))

    import matplotlib.pyplot as plt

    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.show()
    print(masks)

if __name__ == "__main__":



    #from PIL import Image
    img_id = "CA214032V0350"

    import base.load_image_from_file as liff
    img = liff.load_image_from_file(img_id)

    import base.remove_borders as rb
    img = rb.remove_borders(img, img_id)

    import cv2
    img = cv2.resize(img, (1000, 1000))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    segment_any_image(img)