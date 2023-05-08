import numpy as np
import cv2

def plot_3views(re):
    volumn_MxNxN = re.detach().cpu().numpy()[0, -1]

    # get rid of bad points
    zdim = volumn_MxNxN.shape[0] * 100 // 128
    volumn_MxNxN = volumn_MxNxN[:zdim]
    print('volumn min, %f' % volumn_MxNxN.min())
    print('volumn max, %f' % volumn_MxNxN.max())
    # volumn_MxNxN[:5] = 0
    # volumn_MxNxN[-5:] = 0

    volumn_MxNxN[volumn_MxNxN < 0] = 0
    front_view = np.max(volumn_MxNxN, axis=0)
    cv2.imshow("front", front_view / np.max(front_view))
    # cv2.imshow("gt", imgt)
    cv2.waitKey()

    left_view = np.max(volumn_MxNxN, axis=1)
    cv2.imshow("left", left_view / np.max(left_view))
    cv2.waitKey()

    top_view = np.max(volumn_MxNxN, axis=2)
    cv2.imshow("top", top_view / np.max(top_view))
    cv2.waitKey()