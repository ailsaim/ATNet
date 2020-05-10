img_rows, img_cols = 320,320
# img_rows_half, img_cols_half = 160, 160
channel = 4
batch_size = 16
epochs = 100
patience = 50
num_samples = 43100
num_train_samples = 34480
# num_samples - num_trai_samples
num_valid_samples = 8620
unknown_code = 128
epsilon = 1e-6
epsilon_sqr = epsilon ** 2

##############################################################
# Set your paths here

# path to provided foreground images
fg_path = 'data/fg/'
valid_fg_path = 'data/fg_test/'

# path to provided alpha mattes
a_path = 'data/mask/'
valid_a_path = 'data/mask_test/'

# Path to background images (MSCOCO)
bg_path = 'data/bg/'
valid_bg_path = 'data/bg_test/'

# Path to folder where you want the composited images to go
out_path = 'data/merged/'
valid_out_path = 'data/merged_test/'

##############################################################
