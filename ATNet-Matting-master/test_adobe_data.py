import math
import os
import random

import cv2 as cv
import keras.backend as K
import numpy as np

from data_generator import generate_trimap, get_alpha_test
from ATNet_MobileNetV2 import build_encoder_decoder, build_refinement
from utils import compute_mse_loss, compute_sad_loss
from utils import get_final_output, safe_crop, draw_str

# import os
# os.environ['CUDA_VISIBLE_DEVICES']='0'


def composite4(fg, bg, a, w, h):
    fg = np.array(fg, np.float32)
    bg_h, bg_w = bg.shape[:2]
    x = 0
    if bg_w > w:
        x = np.random.randint(0, bg_w - w)
    y = 0
    if bg_h > h:
        y = np.random.randint(0, bg_h - h)
    bg = np.array(bg[y:y + h, x:x + w], np.float32)
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[:, :, 0] = a / 255.
    im = alpha * fg + (1 - alpha) * bg
    im = im.astype(np.uint8)
    return im, bg


if __name__ == '__main__':
    # img_rows, img_cols = 320, 320
    channel = 4
    # model path
    pretrained_path = 'models/ATNetmodel-0.0451.hdf5'

    # test data path
    out_test_path = 'data/merged_test/'
    test_images = [f for f in os.listdir(out_test_path) if
                   os.path.isfile(os.path.join(out_test_path, f)) and f.endswith('.png')]


    total_sad_loss = 0.0
    total_mse_loss = 0.0
    test_data_number = len(test_images)
    for i in range(test_data_number):
        filename = test_images[i]
        image_name = filename.split('.')[0]

        print('\nStart processing image: {}'.format(filename))

        bgr_img = cv.imread(os.path.join(out_test_path, filename))
        bg_h, bg_w = bgr_img.shape[:2]
        print('bg_h, bg_w: ' + str((bg_h, bg_w)))

        a = get_alpha_test(image_name)
        a_h, a_w = a.shape[:2]
        print('a_h, a_w: ' + str((a_h, a_w)))

        alpha = np.zeros((bg_h, bg_w), np.float32)
        alpha[0:a_h, 0:a_w] = a
        trimap = generate_trimap(alpha)

        pd_h = 0
        pd_w = 0
        if bg_h % 64 != 0:
            pd_h = 64 - bg_h % 64
        if bg_w % 64 != 0:
            pd_w = 64 - bg_w % 64

        channel_one = bgr_img[:, :, 0]
        channel_two = bgr_img[:, :, 1]
        channel_three = bgr_img[:, :, 2]

        channel_one = np.pad(channel_one, ((0, pd_h), (0, pd_w)), 'constant', constant_values=(0, 0))
        channel_two = np.pad(channel_two, ((0, pd_h), (0, pd_w)), 'constant', constant_values=(0, 0))
        channel_three = np.pad(channel_three, ((0, pd_h), (0, pd_w)), 'constant', constant_values=(0, 0))
        pd_image = np.dstack((channel_one, channel_two, channel_three))

        pd_trimap = np.pad(trimap, ((0, pd_h), (0, pd_w)), 'constant', constant_values=(0, 0))

        hight, wight = pd_image.shape[:2]

        encoder_decoder = build_encoder_decoder(hight, wight)
        final = build_refinement(encoder_decoder)
        #final = encoder_decoder
        final.load_weights(pretrained_path)
        print(final.summary())


        cv.imwrite('images/adobe_data/{}_image.png'.format(i), np.array(bgr_img).astype(np.uint8))
        cv.imwrite('images/adobe_data/{}_trimap.png'.format(i), np.array(trimap).astype(np.uint8))
        cv.imwrite('images/adobe_data/{}_alpha.png'.format(i), np.array(alpha).astype(np.uint8))

        x_test = np.empty((1,  hight, wight, 4), dtype=np.float32)
        x_test[0, :, :, 0:3] = pd_image / 255.
        x_test[0, :, :, 3] = pd_trimap / 255.

        y_pred = final.predict(x_test)
        print('predict has finished')

        y_pred = np.reshape(y_pred, (hight, wight))
        print(y_pred.shape)
        y_pred = y_pred * 255.0
        y_pred = get_final_output(y_pred, pd_trimap)
        y_pred = y_pred.astype(np.uint8)


        out_1 = y_pred.copy()
        all_out = np.zeros((bg_h, bg_w), np.float32)
        all_out[0:bg_h, 0:bg_w] = out_1[0:bg_h, 0:bg_w]

        all_out_for_loss = all_out.astype(np.uint8)

        sad_loss = compute_sad_loss(all_out_for_loss, alpha, trimap)
        total_sad_loss += sad_loss
        mse_loss = compute_mse_loss(all_out_for_loss, alpha, trimap)
        total_mse_loss += mse_loss
        str_msg = 'sad_loss: %.4f, mse_loss: %.4f' % (sad_loss, mse_loss)
        print(str_msg)

        # draw_str(out, (10, 20), str_msg)
        cv.imwrite('images/adobe_data/{}_out.png'.format(i), all_out)
        K.clear_session()
    mean_sad = total_sad_loss/(1.0*test_data_number)
    mean_mse = total_mse_loss / (1.0 * test_data_number)
    print(mean_sad )
    with open('images/test_result.txt', 'w') as f1:
        f1.write(str(mean_sad) + " "+ str(mean_mse))
