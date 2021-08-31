import os
import json
import cv2
import numpy as np
import tensorflow as tf
from tkinter.filedialog import askdirectory


# fact sheet
path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
dir = os.path.join(path, 'data_collection', 'data', 'facts.json')
print(dir)
with open(dir) as f:
    facts = json.load(f)


def single_prediction():
    h = Helper()
    loaded_model = h.load_model()
    path = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/02_demo/00_test/left_img.png"
    input_img = h.load_img(path)
    h.generate_img(loaded_model, input_img)
    cv2.imshow('single_prediction', h.phenotype)
    cv2.waitKey(0)


def get_warp_transformation(crop_idx):
    crop_idx = [crop_idx['0'], crop_idx['1'], crop_idx['2'], crop_idx['3']]
    pts_from = np.float32(crop_idx)
    pts_to = np.float32([[0, 0],
                        [facts['fig_size']['x'], 0],
                        [facts['fig_size']['x'], facts['fig_size']['y']],
                        [0, facts['fig_size']['y']]])
    M = cv2.getPerspectiveTransform(pts_from, pts_to)
    return M


def crop_feature(M, crop_idx, img):
    img_cropped = cv2.warpPerspective(img,
                                        M,
                                        (int(facts['fig_size']['x']),
                                        int(facts['fig_size']['x'])),
                                        flags=cv2.WARP_FILL_OUTLIERS,
                                        borderMode=cv2.BORDER_CONSTANT)
    return img_cropped


def inverse_fframe(M, crop_idx, feature, fframe):
    rows, cols, chs = feature.shape
    fframe_feature = cv2.warpPerspective(fframe, M, (cols, rows), flags=cv2.WARP_INVERSE_MAP)
    patched_img = overlay_fframe(feature, fframe_feature)
    return patched_img


def overlay_fframe(feature, fframe_feature):
    condition = (fframe_feature != 0)
    patched = np.where(condition, fframe_feature, feature)
    return patched


def custom_img_addition(height_img, toolpath_img):
    condition = np.zeros([256,256,3])
    condition[:,:,0] = (toolpath_img[:,:,2] > 0)
    condition[:,:,1] = (toolpath_img[:,:,2] > 0)
    condition[:,:,2] = (toolpath_img[:,:,2] > 0)
    arr = np.where(condition, toolpath_img, height_img)
    return arr


def load_model():
    dir_name = askdirectory()
    try:
        loaded_model = tf.keras.models.load_model(dir_name)
        print('model is loaded from {}\n'.format(dir_name))
        return loaded_model
    except FileNotFoundError:
        print('model, {}, does not exist\n'.format(dir_name))


def denormalize(img_to_denormalize):
    img_to_denormalize = np.add(img_to_denormalize, 1)
    img_to_denormalize = np.multiply(img_to_denormalize, 127.5)
    return img_to_denormalize


def encode(img_to_encode):
    img_to_encode = img_to_encode.astype(np.uint8)
    return img_to_encode


def generate_img(model, img_path):
    input_img = tf.io.read_file(img_path)
    input_img = tf.image.decode_png(input_img)
    input_img = tf.cast(input_img, tf.float32)
    input_img = (input_img / 127.5) - 1
    input_tensor = np.reshape(input_img, [1, 256, 256, 3])

    prediction = model(input_tensor, training=True)

    output_img = prediction[0].numpy()
    img_denormalized = denormalize(output_img)
    img_encoded = encode(img_denormalized)
    return img_encoded


def write_height2ascii(arr, path, cellsize=1.0):
    grid_data = arr[:,:,1]
    rows,cols = np.shape(grid_data)
    esri = EsriGrid(
                    ncols=cols,
                    nrows=rows,
                    xllcorner=facts['feature_bounds']['min_bound'][0],
                    yllcorner=facts['feature_bounds']['min_bound'][1],
                    cellsize=cellsize,
                    grid_data=grid_data,
                    filepath=path,
                    NODATA_VALUE=-9999)

    esri.write_file()
    return esri


class EsriGrid(object):
    def __init__(self, ncols, nrows, xllcorner, yllcorner, cellsize, grid_data, filepath, NODATA_VALUE=-9999):
        self.ncols = ncols
        self.nrows = nrows
        self.xllcorner = xllcorner
        self.yllcorner = yllcorner
        self.cellsize = cellsize
        self.NODATA_VALUE = NODATA_VALUE
        self.grid_data = grid_data
        self.filepath = filepath

    def read_file(self):
        f = open(self.filepath, "r")
        return f

    def write_file(self):
        f = open(self.filepath, "w")

        # create file header
        f.write("ncols {}\n".format(self.ncols))
        f.write("nrows {}\n".format(self.nrows))
        f.write("xllcorner     {}\n".format(self.xllcorner))
        f.write("yllcorner     {}\n".format(self.yllcorner))
        f.write("cellsize      {}\n".format(self.cellsize))
        f.write("NODATA_value  {}\n".format(self.NODATA_VALUE))

        # write data rows
        for row in range(self.nrows):
            for col in range(self.ncols):
                f.write("{} ".format(self.grid_data[row, col] if self.grid_data[row, col] != 0 else self.NODATA_VALUE ))
            # new row
            f.write("\n")

        # close file
        f.close()



if __name__ == "__main__":
    # scan sandbox
    hm_feature_name = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/00_data_collection/01_production/00000_2021-08-05/01_processed/00000_2021-08-05_height_feature.png"
    hm_feature = cv2.imread(hm_feature_name)
    # draw toolpath
    tp_feature_name = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/00_data_collection/01_production/00000_2021-08-05/01_processed/00000_2021-08-05_toolpath_feature_fix.png"
    tp_feature = cv2.imread(tp_feature_name)
    # get crop_idx
    tp_json_name = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/00_data_collection/01_production/00000_2021-08-05/00_RAW/00000_2021-08-05_toolpath.json"
    with open(tp_json_name) as f:
        tp_facts = json.load(f)
    crop_idx = tp_facts['frame_corner_pts']
    # warp
    M = get_warp_transformation(crop_idx)
    hm_fframe = crop_feature(M, crop_idx, hm_feature)
    tp_fframe = crop_feature(M, crop_idx, tp_feature)
    # prediction
    input_img = custom_img_addition(hm_fframe, tp_fframe)
    input_img_name = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/02_demo/00_test/input.png"
    cv2.imwrite(input_img_name, input_img)
    model = load_model()
    predicted_img = generate_img(model, input_img_name)
    # warp back
    inversed_img = inverse_fframe(M, crop_idx, hm_feature, predicted_img)

    # blur edges experiments
    size = inversed_img.shape
    black = np.zeros(size,dtype=np.uint8)
    black.fill(0)


    cv2.imshow('original',inversed_img)
    cv2.waitKey(0)
    # cv2.imshow('feature',hm_feature)
    # cv2.waitKey(0)
    # create mask
    crop_idx = [crop_idx['0'], crop_idx['1'], crop_idx['2'], crop_idx['3']]
    pts_from = np.int32(crop_idx)
    mask = cv2.fillConvexPoly(black, pts_from, (255,255,255))
    mask_blur  = cv2.GaussianBlur(mask,(99,99),0).astype('float') / 255.
    cv2.imshow('mask',mask_blur)
    cv2.waitKey(0)
    img = inversed_img.astype('float') / 255.
    bg = hm_feature.astype('float') / 255.
    out  = bg * (1 - mask_blur)  + img * mask_blur
    out = (out * 255).astype('uint8')
    cv2.imshow('result',out)
    cv2.waitKey(0)


    # saving terrain
    # save_dir = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/02_demo/00_test/"
    # ascii_path = save_dir + "ascii_{}.txt".format(0)
    # write_height2ascii(inversed_img, ascii_path)
