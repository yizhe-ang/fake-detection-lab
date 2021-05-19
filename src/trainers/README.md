# EXIF-SC Training
The code implements a mock of the self-consistency training algorithm, which comprises two-stages:

# First Stage
The first stage trains the network to predict EXIF attribute consistency (multi-label classification) from a pair of image patches.

The sampling process for a training batch is as follows:

1. We'll select a specific EXIF attribute value. To do this, we'll first randomly sample an EXIF attribute, and then randomly sample a value from it.
2. The first half of the batch will be consistent, i.e. pairs will both have that specific attribute value. We randomly sample from the set of images with that attribute value.
3. The second half of the batch will be inconsistent, i.e. the first image will that specific value, but the second image to be compared to will have a different value. We sample from the rest of the images to form those second images.

# Second Stage
The second stage attaches another MLP on top of the EXIF attribute predictions in order to train it to predict whether the image patches come from the same image (binary classification).

The rest of the network weights are frozen, and only this MLP is trained.

A training batch is constructed by ensuring that the first half of the batch is consistent (pairs are patches that come from the same image), and the second half of the batch is inconsistent (pairs are patches that come from different images).

# Implementation Notes
- The EXIF attributes to predict are chosen dynamically based on the dataset, for e.g. the top 80 EXIF attributes with the least missing values.
- If an attribute is missing for either image, it is immediately assigned a target of 0.
- Binary cross-entropy is used as the loss function.
- The incorporation of post-processing consistency attributes is yet to be implemented.