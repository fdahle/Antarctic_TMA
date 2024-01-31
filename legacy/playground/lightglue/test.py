from external.lightglue import LightGlue, SuperPoint, DISK
from external.lightglue.utils import load_image, rbd
from external.lightglue import viz2d

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 'mps', 'cpu'

extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
matcher = LightGlue(features='superpoint').eval().to(device)

image0 = load_image("/home/fdahle/Desktop/lg_img.png")
image1 = load_image("/home/fdahle/Desktop/lg_sat.png")

print(image1.shape)


feats0 = extractor.extract(image0.to(device))
feats1 = extractor.extract(image1.to(device))
matches01 = matcher({'image0': feats0, 'image1': feats1})
feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension

print(matches01['scores'])

kpts0, kpts1, matches = feats0['keypoints'], feats1['keypoints'], matches01['matches']
m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

pts0 = m_kpts0.cpu().numpy()
pts1 = m_kpts1.cpu().numpy()
conf = matches01['scores'].detach().cpu().numpy()

print(conf)
exit()

axes = viz2d.plot_images([image0, image1])
viz2d.plot_matches(m_kpts0, m_kpts1, color='lime', lw=0.2)
viz2d.add_text(0, f'Stop after {matches01["stop"]} layers')


kpc0, kpc1 = viz2d.cm_prune(matches01['prune0']), viz2d.cm_prune(matches01['prune1'])
viz2d.plot_images([image0, image1])
viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=6)

print(matches.shape)

import matplotlib.pyplot as plt
plt.show()