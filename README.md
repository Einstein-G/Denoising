Hello everyone
Clone the repository (as seen in presentation)

The scripts are setup so you can configure all parameters from the config files in that folder so you don't really need to edit scripts unless you want to tweak features for yourself.
Here's an explanation about what each file in there does, in the order I usually use them.

_____________________________________________________________________________________

Optional: average_stacks.py 
-Takes a large stack that has multiple timeframes in order and produces an average stack. Example: 366x1024x1024 stack consisting of 6 sets of 61 images. Basically image 0,61,122, ... have different 1x frames of the same slice. Resulting Stack is 61x1024x1024
 
_____________________________________________________________________________________

generate_2D_patches.py T
-This will generate patches from your data. If your data is 2D (i.e. no depth) the patch_size parameter must be 2D as well.  You can edit the sample_data_config.json file to setup the parameters for your application. Here is the description of the parameters.

Basepath: Path to a base folder in which you can find the source and target dirs
source_dir: Folder names for source directory. Basically, the directory of your inputs for training. The directory should look like this: 
SOURCE_DIR/T01 contains img1.tif,img2.tif …
SOURCE_DIR/T02 contains img1.tif,img2.tif …
SOURCE_DIR/T03 contains img1.tif,img2.tif …
target_dir: Name of directory of targets. Can be other noisy frames (n2n) or averages. Structure should be:
TARGET_DIR contains img1.tif,img2.tif …
It’s important that corresponding images have the same file name. so a 1x frame of ROI1 should have the same name as any other 1x frame of the same ROI and as the average of that ROI. That’s why we place them in different folders. This is at times frustrating but works.
 
axes: Ignore
patch_size: the size of your patches. Smaller size means lighter model that trains faster/with fewer data but has more “edge effects” (where pixels on the edge of patches don’t have full context) 
n_patches_per_image: Number of patches to create per image for training. These are made by looking for bright areas to avoid sampling empty parts. You can just set it to 1024 and forget it, I will let you know if I think it should be changed
transforms: Controls the type of transforms that will be made for each patch. Rotate 3 will produce 4 images per image that are rotated by 0, 90, 180, 270 degrees. I will implement more transforms like mirroring but that’s good for now. Check the script for the exact syntax, it’s a little bit weird. 
save_file: Where to save the resulting data file. This will be a single file containing all your patches in a fast to read format, needs to be the input for train.py
_____________________________________________________________________________________
 
train.py Trains a model. Will produces patch images and a training history graph as outputs in the directory its is run in. Edit sample_train_config with your parameters
parameters:
data_path: path to the file created by generate_2D_patches
model_name: what to name the directory in which your model will be saved
model_save_path: where your model should be saved
train_epochs: how many training epochs to do.
train_setps_per_epoch: how many steps per epoch. Can start as 100 but should really be bumped to 1000. 
To determine how long to train your networks, check the history graphs. If you are noticing that the loss function keeps trending downward when the model stops training, increase epochs or steps a bit. 
 
_____________________________________________________________________________________
 
care.py The script to use to denoise your test set. Parameters are self explanatory just point to a directory of images and to a directory to save the resulting denoised images. Don’t forget to set the model name and path accordingly. 
_____________________________________________________________________________________



compare_stacks.py will compare stacks of images against a “clean” ground truth and produce graphs of the metric as a function of imaging depth.  As usual, use sample_compare_config.json to set parameters. 
Parameters are again self explanatory but here are a couple notes on them:
Order matters. The labels and filenames should be in the same order.
In file labels, use the syntax as is. There must always be an image labeled ‘clean’ which is used as the ground truth internally. 
In metrics, again use the syntax as is. Currently the options are mse, psnr and mae. If you want to implement more metrics let me know and we can make it work. 


Upcoming features:
	crossvalidate.py  Does crossvalidation on a dataset. Should take a long time to run because it trains multiple models. 
	Beta analysis incorporated into compare_stacks.py


