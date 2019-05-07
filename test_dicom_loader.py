"""
Unit tests that ensure that the DICOMLoader class is functioning correctly
"""
import unittest
import numpy as np
from scipy.misc import imsave
import os
from dicom_loader import DICOMLoader

class TestDICOMLoader(unittest.TestCase):

    def test_num_data(self):
        """
        Tests the various properties of a DICOMLoader object
        and if the number of elements loaded in match the example
        dataset provided.  There are 96 i-contour files overall,
        so the total dataset should be 96 elements.
        """

        dl = DICOMLoader('data')
        self.assertTrue(len(dl) == 96)
        self.assertFalse(dl.debug_mode)
        self.assertFalse(dl.return_filenames_with_batch)
        self.assertTrue(dl.seed is None)
        self.assertTrue(dl.batch_size == 8)
        self.assertTrue(dl.base_dir == 'data')

        # Change some things now
        dl.return_filenames_with_batch = True
        dl.batch_size = 16
        dl.seed = 42
        self.assertTrue(dl.return_filenames_with_batch)
        self.assertTrue(dl.seed == 42)
        self.assertTrue(dl.batch_size == 16)

    def test_next(self):
        """
        Tests the next() method with various batch sizes to ensure that
        the data is being loaded properly using the dataset provided.
        """

        dl = DICOMLoader('data')
        dl.return_filenames_with_batch = True # for debugging

        # Default batch size is 8
        (imgs, masks, filenames) = dl.next()

        # Ensure the shapes of what are returned all match
        self.assertTrue(imgs.shape == (8, 256, 256))
        self.assertTrue(masks.shape == (8, 256, 256))
        self.assertTrue(len(filenames) == 8)

        # Ensure all filenames are unique
        dicom_names = [f[0] for f in filenames]
        contour_names = [f[1] for f in filenames]
        dicom_set = set(dicom_names)
        contour_set = set(contour_names)
        self.assertTrue(len(dicom_set) == 8)
        self.assertTrue(len(contour_set) == 8)

        # Change the batch size to 48
        dl.batch_size = 48

        # Try again
        (imgs, masks, filenames) = dl.next()
        self.assertTrue(imgs.shape == (48, 256, 256))
        self.assertTrue(masks.shape == (48, 256, 256))
        self.assertTrue(len(filenames) == 48)
        dicom_names = [f[0] for f in filenames]
        contour_names = [f[1] for f in filenames]
        dicom_set = set(dicom_names)
        contour_set = set(contour_names)
        self.assertTrue(len(dicom_set) == 48)
        self.assertTrue(len(contour_set) == 48)

        # Change batch size to 100
        # Should default to 96
        dl.batch_size = 100
        self.assertTrue(dl.batch_size == 96)

        # Try getting in a batch
        # This should return None because we have now
        # exceeded the number of examples to return as we've
        # already gone through 48 + 8 = 56 examples and so
        # it's impossible to get 96 now
        tmp = dl.next()
        self.assertTrue(tmp is None)

    def test_seed_reset(self):
        """
        Tests whether the seeding and reset mechanisms work.  By
        setting a random seed, we generate an image batch and mask
        data.  By resetting the seed and generating the data again
        with the same batch size, the difference should be small.
        """

        dl = DICOMLoader('data')
        dl.return_filenames_with_batch = True
        dl.batch_size = 16
        dl.seed = 10101

        # Load in the batch
        (imgs_before, masks_before, filenames_before) = dl.next()

        # Reset
        dl.reset()

        # Try it again
        (imgs_after, masks_after, filenames_after) = dl.next()

        # Ensure it's all the same
        self.assertTrue(imgs_before.shape == imgs_after.shape)
        self.assertTrue(masks_before.shape == masks_after.shape)
        self.assertTrue(len(filenames_before) == len(filenames_before))

        self.assertTrue(np.all(imgs_before == imgs_after))
        self.assertTrue(np.all(masks_before == masks_after))
        for (i, j) in zip(filenames_before, filenames_after):
            self.assertTrue(i == j)

    def test_one_epoch(self):
        """
        Tests whether we can iterate through one epoch successfully.
        Also writes the original images and masked regions to file to
        visually assess correctness
        """

        # Set up DICOMLoader
        dl = DICOMLoader('data')
        count = 0
        num_iter = 0
        dl.seed = 42

        # Make the output directory if we need to
        if not os.path.exists('test_dicom_loader_output'):
            os.makedirs('test_dicom_loader_output')

        # For each batch of images and masks...
        for (imgs, masks) in dl:
            # Verify the shape
            self.assertTrue(imgs.shape == (8, 256, 256))
            self.assertTrue(masks.shape == (8, 256, 256))

            # Go through each pair of image and mask...
            for (img, mask) in zip(imgs, masks):
                # Write the input image and masked image to file
                input_img = np.dstack([img] * 3)
                output_img = input_img.copy()
                output_img[...,0][mask] = np.max(img)
                output_img[...,1][mask] = np.max(img)

                imsave(os.path.join('test_dicom_loader_output',
                                    'output{:04d}.png'.format(count + 1)),
                                    np.hstack((input_img, output_img)))
                count += 1

            num_iter += 1

        # Checks if we have gone through 96 examples and
        # 12 batches
        self.assertTrue(count == 96)
        self.assertTrue(num_iter == 12)

    def test_one_epoch_with_ocontours(self):
        """
        Tests whether we can iterate through one epoch successfully including
        loading in the o-contour files. Also writes the original images and
        masked regions to file to visually assess correctness.
        """

        # Set up DICOMLoader
        dl = DICOMLoader('data', load_ocontours=True)
        count = 0
        num_iter = 0
        dl.seed = 42

        # Make the output directory if we need to
        if not os.path.exists('test_dicom_loader_output_withocontours'):
            os.makedirs('test_dicom_loader_output_withocontours')

        # For each batch of images and masks...
        for (imgs, i_masks, o_masks) in dl:
            # Verify the shape
            self.assertTrue(imgs.shape == (8, 256, 256))
            self.assertTrue(i_masks.shape == (8, 256, 256))
            self.assertTrue(o_masks.shape == (8, 256, 256))

            # Go through each triple of image and masks...
            for (img, i_mask, o_mask) in zip(imgs, i_masks, o_masks):
                # Write the input image and masked i-contour image to file
                input_img = np.dstack([img] * 3)
                output_img_icontour = input_img.copy()
                output_img_icontour[...,0][i_mask] = np.max(img)
                output_img_icontour[...,1][i_mask] = np.max(img)

                # Repeat for o-contour mask
                output_img_ocontour = input_img.copy()
                output_img_ocontour[...,0][o_mask] = np.max(img)
                output_img_ocontour[...,1][o_mask] = np.max(img)

                imsave(os.path.join('test_dicom_loader_output_withocontours',
                                    'output{:04d}.png'.format(count + 1)),
                                    np.hstack((input_img, output_img_icontour, output_img_ocontour)))

                count += 1

            num_iter += 1

        # Checks if we have gone through 40 examples and
        # 5 batches since there are only 46 examples
        self.assertTrue(count == 40)
        self.assertTrue(num_iter == 5)

    def test_num_data_with_ocontour(self):
        """
        Tests the functionality of loading in the o-contour files in addition
        to the i-contour files with the associated images.
        """

        # Set up the DICOMLoader
        dl = DICOMLoader('data', load_ocontours=True)

        # There are 46 o-contour files, meaning that there should now
        # be just 46 files loaded in
        self.assertTrue(len(dl) == 46)

    def test_next_with_ocontour(self):
        """
        Same line of thinking with test_next, but now we want to explicitly
        load in o-contour data too
        """

        dl = DICOMLoader('data', load_ocontours=True)
        dl.return_filenames_with_batch = True # for debugging

        # Default batch size is 8
        (imgs, i_masks, o_masks, filenames) = dl.next()

        # Ensure the shapes of what are returned all match
        self.assertTrue(imgs.shape == (8, 256, 256))
        self.assertTrue(i_masks.shape == (8, 256, 256))
        self.assertTrue(o_masks.shape == (8, 256, 256))
        self.assertTrue(len(filenames) == 8)

        # Ensure all filenames are unique
        dicom_names = [f[0] for f in filenames]
        icontour_names = [f[1] for f in filenames]
        ocontour_names = [f[2] for f in filenames]
        dicom_set = set(dicom_names)
        icontour_set = set(icontour_names)
        ocontour_set = set(ocontour_names)
        self.assertTrue(len(dicom_set) == 8)
        self.assertTrue(len(icontour_set) == 8)
        self.assertTrue(len(ocontour_set) == 8)

        # Change the batch size to 24
        dl.batch_size = 24

        # Try again
        (imgs, i_masks, o_masks, filenames) = dl.next()
        self.assertTrue(imgs.shape == (24, 256, 256))
        self.assertTrue(i_masks.shape == (24, 256, 256))
        self.assertTrue(o_masks.shape == (24, 256, 256))
        self.assertTrue(len(filenames) == 24)
        dicom_names = [f[0] for f in filenames]
        icontour_names = [f[1] for f in filenames]
        ocontour_names = [f[2] for f in filenames]
        dicom_set = set(dicom_names)
        icontour_set = set(icontour_names)
        ocontour_set = set(ocontour_names)
        self.assertTrue(len(dicom_set) == 24)
        self.assertTrue(len(icontour_set) == 24)
        self.assertTrue(len(ocontour_set) == 24)

        # Change batch size to 100
        # Should default to 46
        dl.batch_size = 100
        self.assertTrue(dl.batch_size == 46)

        # Try getting in a batch
        # This should return None because we have now
        # exceeded the number of examples to return as we've
        # already gone through 48 + 8 = 56 examples and so
        # it's impossible to get 96 now
        tmp = dl.next()
        self.assertTrue(tmp is None)

    def test_loading_some_patients(self):
        """
        Tests to ensure that we can load in one or a few patients instead
        of all patients in the link.csv file
        """

        dl = DICOMLoader('data', patient_IDs=['SCD0000101'])
        # Should only contain 18 files
        self.assertTrue(len(dl) == 18)

        # Try again with o-contour files
        dl = DICOMLoader('data', patient_IDs=['SCD0000101'], load_ocontours=True)
        self.assertTrue(len(dl) == 9)

        # Try with a few more patients
        dl = DICOMLoader('data', patient_IDs=['SCD0000101', 'SCD0000301', 'SCD0000401'])
        # There are 18 + 20 + 18 = 56 files for i-contours
        self.assertTrue(len(dl) == 56)

        # Try again with o-contour files
        dl = DICOMLoader('data', patient_IDs=['SCD0000101', 'SCD0000301', 'SCD0000401'], load_ocontours=True)
        self.assertTrue(len(dl) == 28)

if __name__ == '__main__':
    unittest.main()