"""
This file contains a tool suite that loads in DICOM images and their
associated contour files which are a list of 2D polygon coordinates that
are closed and encompass an area inside the DICOM image.  The closed polygon
coordinates are used to create an associated mask where any locations internal
to the polygon are marked as unsigned 8-bit 255 and those locations external to
the polygon are marked unsigned 8-bit 0.  Additionally, once the data has been
loaded, the tool suite can be used to generate a batch of DICOM images and
associated binary masks that determine what is interesting to look at in each
DICOM image.

The tool suite assumes that a path to a data directory is provided and it
contains the following structure:

directory
|
| --> /contourfiles
| --> /dicoms
| --> link.csv

It contains two subdirectories: contourfiles and dicoms that contains
the following subdirectories:

---

contourfiles
|
| --> <directory #1 of contour data>
     |
     | --> i-contours
     | --> o-contours
| --> <directory #2 of contour data>
     |
     | --> i-contours
     | --> o-contours
| --> <directory #3 of contour data>
     |
     | --> i-contours
     | --> o-contours
...

Inside each i-contours and o-contours subdirectory are text files that contain
coordinates where each row is space delimited that has (x, y) coordinates
and the files are named in the following convention:

    IM-0001-xxxx-icontour-manual.txt
                or
    IM-0001-xxxx-ocontour-manual.txt

The xxxx is a DICOM image ID that contains the contour information demarcating
a closed polygon of the corresponding DICOM image enumerated by the ID (more on
this later).  Notice that the icontour and ocontour towards the end of the file
denote that the file is an i-contour or o-contour respectively.

---

dicoms
|
| --> <directory #1 of DICOM image data>
| --> <directory #2 of DICOM image data>
| --> <directory #3 of DICOM image data>
...

Each directory contains DICOM image files.  Each image is enumerated as
<ID>.dcm where <ID> contains an integer image ID that corresponds to the contour
points found in the contourfiles directory.  Notice that each directory inside
contourdata may not have the same order of appearance for the dicoms directory.
That is, directory #1 of the contour data may not correspond to directory #1
of the DICOM image data.

---

To mitigate this shortcoming, the link.csv file contains the linking
information that allows us to determine which directory of the contour data
matches the corresponding directory of the DICOM image data  The CSV file
contains the following information:

patient_id,original_id
<directory #1 inside dicoms containing image data>,<directory #1 inside contourfiles that corresponds>
<directory #2 inside dicoms containing image data>,<directory #2 inside contourfiles that corresponds>
<directory #3 inside dicoms containing image data>,<directory #3 inside contourfiles that corresponds>
...
...

The first column denotes a directory inside the dicoms directory that contains
DICOM images.  This is known as the patient ID and signifies the ID of the patient
that the DICOM images belong to.  The second column provides which corresponding
directory in the contoursfile directory that is being used to demarcate the images
in the dicoms subdirectory provided by the first column.  Also note that
there may be missing data where there is contour data defined for a particular
image, but the image may not be present in the corresponding directory of the
image data and vice-versa.  This is interpreted as missing data and will not be
loaded in.

In addition, to increase functionality and to provide multiple use cases, the
user has the option of loading in either just the i-contours or both the
i-contours and o-contours.  If the user desires to load both, then the data
loaded in will only consist of DICOM images such that both contours exist for
each DICOM image.  Specifically, if there are DICOM images that only have
an i-contour but not an o-contour, this pair of image and associated contours
are treated as missing and are not loaded in.  The default operation is to NOT
load in both contours so the user must specifically construct a DICOMLoader
object that loads in both sets of contour information.

Once the directory has been provided, one simply sets up the batch size
desired and the created instance of the class is treated as a Python iterator
where you simply iterate over the iterator until the number of examples have
been exhausted.  One can also reset the iterator to bring it back to the
beginning for reproducible batches or one can set a random seed.

One can toggle debug mode by outputting status messages to file
for debug purposes.  These messages are written at different stages
of the loading and processing.  Note that the messages are appended to the
debug file to retain history.

Finally, because it may be desirable to analyze a pair of DICOM and contour
directories individually for some analysis, one can also provide a list of
patient IDs
"""

import os
import numpy as np
import parsing
import csv
from numpy.random import RandomState
import logging

class DICOMLoader(object):
    """
    DICOM Image and Mask Loader for DICOM images and their associated i-contour
    files

    One provides a directory that contains the data in the agreed format as
    noted above.  After setting the batch_size property, one can simply use
    the instance and iterate through it until the examples are exhausted.
    You can also set the batch size and iterate through it again with the
    modified batch size. The state of the batch generation will reset if a new
    random seed is provided or if the reset method is explicitly set.

    As one iterates over the training data, the following behaviour occurs:

    1. For each epoch...
        a. Randomly shuffle the dataset
        b. For each batch in the shuffled dataset...
            i. Provide a batch size amount of DICOM images and associated masks

    If the number of images is not a multiple of the batch size, then the
    final batch is simply ignored and a batch is not created for this last part
    of the dataset.  However, the random shuffling at the beginning of each
    epoch mitigates this.

    The default batch size is 8.

    Example use:
        # Create instance of DICOMLoader
        dl = DICOMLoader('/path/to/data')

        # Retrieve base directory, but you can't set it
        # You need to create another DICOMLoader object to change
        # the data - safer that way
        base_dir = dl.base_dir

        # Set batch size to 16 and seed 42
        dl.batch_size = 16
        dl.seed = 42

        # Iterate until we run out of batches...
        for (imgs, i_masks) in dl:
            # imgs contains the DICOM images and should be of size batch_size x rows x columns
            # masks contains the associated masks for each DICOM image and should
            # be of size batch_size x rows x cols
            #
            # Do something with the data
            # ...

        # Reset the batch size
        dl.batch_size = 8 # This doesn't change or reset the random seed
        for (imgs, i_masks) in dl:
            # Rinse and repeat...
            #
            # Do something with the data
            # ...

        dl.reset() # Resets the generator to the beginning

        # Reset the seed
        dl.seed = 10 # Can also set to None which sets to the default
                     # random seed

        for (imgs, i_masks) in dl:
            # Rinse and repeat
            #
            # Do something with the data
            # ...

        # Also output the DICOM and contour filenames for debugging
        dl.return_filenames_with_batch = True
        for (imgs, i_masks, filenames) in dl:
            # Rinse and repeat...
            # filenames is a list of tuples where each element is
            # the full path to a DICOM image name and associated i-contour
            # file
            #
            # Do something with the data
            # ...

        # For the case of loading in both contours at the same time:
        dl = DICOMLoader('/path/to/data', load_ocontours=True)

        # Iterate over the batches of DICOM images and pairs of masks
        for (imgs, i_masks, o_masks) in dl:
            # imgs and i_masks are the same as before but o_masks now
            # contain the o-contours and are the same size as i_masks
            #
            # Do something with the data...

        # Also output the contour filenames for debugging
        dl.return_filenames_with_batch = True
        for (imgs, i_masks, o_masks, filenames) in dl:
            # Rinse and repeat...
            # Like before, but filenames is now a list of tuples with three
            # elements as opposed to two that contain the full path to a DICOM
            # image name and associated i-contour and o-contour files
            #
            # Do something with the data
            # ...

        # Can change the batch size and random seed like before
        dl.batch_size = 16
        dl.seed = 60060
        dl.return_filenames_with_batch = False

        for (imgs, i_masks, o_masks) in dl:
            # Same as before
            # ...
            # ...

        # Load in data that is for just one patient
        dl = DICOMLoader('/path/to/data', patient_IDs=['patient_ID1'])
        for (imgs, i_masks) in dl:
            # Rinse and repeat
            # ...

        # Do it again for some more patients and load in o-contours
        dl = DICOMLoader('/path/to/data', load_ocontours=True, patient_IDs=['patient_ID1', 'patient_ID2'])

        for (imgs, i_masks, o_masks) in dl:
            # Rinse and repeat
            # ...
    """

    def __init__(self, base_dir, load_ocontours=False, patient_IDs=None, debug_mode=False):
        """
        Constructor.  Given a base directory with the agreed format, find all
        pairs of files that have both an i-contour file and corresponding mask
        file.  This is the default operation.  You can additionally set a flag
        to load both i-contour and o-contour files as well, but only those
        images that have both an i-contour and o-contour file are loaded in.
        Finally, one can simply load a subset of the data by specifying which
        patient IDs to use by consulting the patient_id column in the link.csv
        file.

        Args:
            base_dir: A str containing the path to the base directory
            load_ocontours: Decides to load both i-contour and o-contour files
            patient_IDs: A list of strings such that each element is a patient
                         ID to load in.
            debug_mode: Turn on/off debug mode.  You can also set it via the
                        property

        Raises:
            ValueError: If the directory is not in the agreed format
            Exception: If the CSV file is not properly formatted
            Exception: If any of the files in any of the data directories is not
                       in the agreed format
            Exception: If a directory inside the link.csv file does not exist
            Exception: If the input directory is not a string
            Exception: If any of the flags are not Boolean
            Exception: If the patient IDs are not in a list
            Exception: If any patient ID is not a string
            Exception: If all patient IDs specified do not exist in the CSV file
            Exception: If the data loaded in is empty
        """

        # Basic debugging of inputs into constructor
        if not isinstance(base_dir, str):
            raise ValueError("The base directory {} is not a string".format(base_dir))

        if not isinstance(debug_mode, bool):
            raise ValueError("The debug mode flag is not True or False")

        if not isinstance(load_ocontours, bool):
            raise ValueError("The loading of the o-contours flag is not True or False")

        if patient_IDs is not None:
            if not isinstance(patient_IDs, list):
                raise ValueError("The patient IDs should be a list of strings")

            for st in patient_IDs:
                if not isinstance(st, str):
                    raise ValueError("{} is not a string in the patient IDs".format(st))

        # Debug mode flag - outputs a logging file just in case
        self._debug_mode = debug_mode

        # If we turn it on, create a logging instance
        if debug_mode:
            logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='debug.log')

        # List all files / directories in the base directory
        if debug_mode:
            logging.info('Begin construction of DICOMLoader')
        files_dir = os.listdir(base_dir)

        # Basic error checking for core contents
        # Should only contain 2 directories and 1 file
        if debug_mode:
            logging.info('Checking directory structure of {}'.format(files_dir))

        if len(files_dir) != 3:
            if debug_mode:
                logging.error("Should contain only 2 directories and 1 file")
            raise ValueError("Should contain only 2 directories and 1 file")

        # Make sure the directories and files align
        sorted_files_dir = tuple(sorted(files_dir))
        if sorted_files_dir != ('contourfiles', 'dicoms', 'link.csv'):
            if debug_mode:
                logging.error("Directory structure should be: contourfiles, dicoms, link.csv")
            raise ValueError("Directory structure should be: contourfiles, dicoms, link.csv")

        # Remember this for later
        self._base_dir = base_dir

        # Set default batch size and seed
        self._batch_size = 8
        self._seed = None

        # The recommended way to create a random object as we will create
        # independent Random streams and is the recommended practice for
        # development
        # See https://stackoverflow.com/a/5837352/3250829 for a good discussion
        self._prng = RandomState()

        # Default is to not output the filanmes for each batch
        self._return_filenames_with_batch = False

        # Iternally store the height and width of each image
        # for verification
        self._height = None
        self._width = None

        # Remember the other attributes now
        self._load_ocontours = load_ocontours
        self._patient_IDs = None # Default

        # Internally load in the contour data and DICOM image data as
        # a list of tuples such that the first element is the path to a
        # DICOM image and the second is the corresponding matching contour file
        if debug_mode:
            logging.info("Parsing link.csv file")
        try:
            with open(os.path.join(base_dir, 'link.csv'), 'r') as f:
                rdr = csv.DictReader(f)
                dirs = [(row['patient_id'], row['original_id']) for row in rdr]
        except Exception as e:
            if debug_mode:
                logging.error("CSV file is not formatted correctly")
            raise

        # Remembers all corresponding pairs of contour data and DICOM image data
        if debug_mode:
            logging.info("Loading in pairs of contour data and corresponding DICOM images")

        # Additional step
        # Remove out the patient IDs in the link.csv list that are not in the
        # patient_IDs list
        if patient_IDs is not None:
            # Create a set of patient IDs and also removing duplicates
            patient_set = set(patient_IDs)

            # Go through each element in the loaded list of tuples and only
            # remember those patients that matter
            dirs = [(p_ID, o_ID) for (p_ID, o_ID) in dirs if p_ID in patient_set]

            # Safe guard against patient IDs that don't exist
            if len(dirs) == 0:
                if debug_mode:
                    logging.error("No matching criteria for selecting out patient IDs")
                raise Exception("No matching criteria for selecting out patient IDs")

            # Save patient IDs list based on what is valid
            self._patient_IDs = [p_ID for (p_ID, _) in dirs]

        # Internal variable that stores the pairs
        self._data_pairs = self.__load_pairs(dirs)

        # Safe guard to ensure there is some data
        if len(self._data_pairs) == 0:
            if debug_mode:
                logging.error("No data loaded in due to improperly formatted filenames")
            raise Exception("No data loaded in due to improperly formatted filenames")

        # Remembers the number of examples
        self._num_examples = len(self._data_pairs)

        # Remembers where we are when iterating over the data
        self._index = 0

        # Remembers which batch number we're at - primarily for debugging
        self._batch_number = 0

        # Remembers the shuffled indices that allow us to create
        # our batches
        self._indices = None


    def __load_pairs(self, dirs):
        """
        Internal method that loads in all correct pairs of DICOM images and
        corresponding contour data.  Handles both i-contours and i-contours +
        o-contours.

        Args:
            dirs: List of tuples where each element is a pair of DICOM image
                  directory and contour files directory

        Returns:
            A list of tuples each with two elements. Each tuple contains a pair
            of paths to a DICOM image and associated i-contour file.  Each tuple
            could also be a triple of paths to a DICOM image and associated
            i-contour and o-contour files.

        Raises:
            Exception: If either a directory in the dicoms or contourfiles
                       directory doesn't exist for a provided tuple
        """

        # List to return
        data_pairs = []

        # Go through each pair...
        for (dicom_dir, contour_dir) in dirs:
            # Check if each pair of directory exists
            dicom_dir_full = os.path.join(self._base_dir, 'dicoms', dicom_dir)
            if not os.path.exists(dicom_dir_full):
                out_str = "{} does not exist in {}".format(dicom_dir,
                            os.path.join(self._base_dir, 'dicoms'))
                if self._debug_mode:
                    logging.error(out_str)
                raise Exception(out_str)

            contour_dir_full = os.path.join(self._base_dir, 'contourfiles',
                                            contour_dir)
            if not os.path.exists(contour_dir_full):
                out_str = "{} does not exist in {}".format(contour_dir,
                           os.path.join(self._base_dir, 'contourfiles'))
                if self._debug_mode:
                    logging.error(out_str)
                raise Exception(out_str)

            # Next obtain a list of contour files inside the i-contours
            # directory and simultaneously check the filename format
            icontour_dir = os.path.join(contour_dir_full, 'i-contours')
            files_icontour = [f for f in os.listdir(icontour_dir)
                              if os.path.isfile(os.path.join(icontour_dir, f))
                              and f.endswith('.txt')]

            # If there are no files in this i-contour directory, skip
            if len(files_icontour) == 0:
                if self._debug_mode:
                    logging.warning("No i-contour files exist in {}".format(icontour_dir))
                continue

            # Obtain a list of contour files inside the o-contours directory
            # and simultaneously check the filename format
            if self._load_ocontours:
                ocontour_dir = os.path.join(contour_dir_full, 'o-contours')
                files_ocontour = [f for f in os.listdir(ocontour_dir)
                              if os.path.isfile(os.path.join(ocontour_dir, f))
                              and f.endswith('.txt')]

                # If there are no files in this o-contour directory, skip
                if len(files_ocontour) == 0:
                    if self._debug_mode:
                        logging.warning("No o-contour files exist in {}".format(ocontour_dir))
                    continue

            # Go through each file and ensure proper formatting
            # Also remember all of the unique IDs
            contour_ids = set([])
            for file in files_icontour:
                # Split based on '-'
                tokens = file.split('-')

                # There should be 5 strings
                if len(tokens) != 5:
                    if self._debug_mode:
                        logging.warning("Contour file {} is improperly named".format(os.path.join(contour_dir_full, file)))
                    continue

                # Ensure that the first two tokens are IM and 0001
                if tokens[0] != "IM":
                    if self._debug_mode:
                        logging.warning("Contour file {} is improperly named".format(os.path.join(contour_dir_full, file)))
                    continue

                if tokens[1] != "0001":
                    if self._debug_mode:
                        logging.warning("Contour file {} is improperly named".format(os.path.join(contour_dir_full, file)))
                    continue

                # Get the ID of the file and add it in
                # Should be the third token
                try:
                    contour_ids.add(int(tokens[2].lstrip('0')))
                except Exception as e:
                    # Handles the case where the third token isn't an integer
                    if self._debug_mode:
                        logging.warning("ID in {} is not an integer: {}".format(os.path.join(contour_dir_full, file), tokens[2]))

            # Repeat the logic but for the o-contours
            # contour_ids is a set of IDs that have i-contours
            # Simply create a second set and intersect it with the first
            # set.  What is common between them are the IDs where they both
            # contain i-contour and o-contour data
            if self._load_ocontours:
                contour_ids_outer = set([])
                for file in files_ocontour:
                    # Repeat the same logic
                    tokens = file.split('-')
                    if len(tokens) != 5:
                        if self._debug_mode:
                            logging.warning("Contour file {} is improperly named".format(os.path.join(contour_dir_full, file)))
                        continue

                    if tokens[0] != "IM":
                        if self._debug_mode:
                            logging.warning("Contour file {} is improperly named".format(os.path.join(contour_dir_full, file)))
                        continue

                    if tokens[1] != "0001":
                        if self._debug_mode:
                            logging.warning("Contour file {} is improperly named".format(os.path.join(contour_dir_full, file)))
                        continue

                    try:
                        contour_ids_outer.add(int(tokens[2].lstrip('0')))
                    except Exception as e:
                        if self._debug_mode:
                            logging.warning("ID in {} is not an integer: {}".format(os.path.join(contour_dir_full, file), tokens[2]))

                # The final set is just the intersection
                contour_ids = contour_ids_outer.intersection(contour_ids)

            # If there are no valid contour files, skip
            if len(contour_ids) == 0:
                if self._debug_mode:
                    logging.warning("No valid contour files for {}".format(contour_dir_full))
                continue

            # Go through each DICOM image and see if there is a corresponding
            # contour file that matches this ID

            # First filter out only the files that are dicom images
            files_dicom = [f for f in os.listdir(dicom_dir_full)
                           if os.path.isfile(os.path.join(dicom_dir_full, f))
                           and f.endswith('.dcm')]

            # Safe guard against no images
            if len(files_dicom) == 0:
                if self._debug_mode:
                    logging.warning("No DICOM files in {}".format(dicom_dir_full))
                continue

            # Go through each DICOM image file now...
            for file in files_dicom:
                full_filename = os.path.join(dicom_dir_full, file)
                # Obtain the ID
                try:
                    ID, _ = os.path.splitext(file)
                    ID = int(ID)
                except Exception as e:
                    if self._debug_mode:
                        logging.error("DICOM file {} is improperly named".format(full_filename))
                    continue

                # See if we can find the same value
                if ID not in contour_ids:
                    if self._debug_mode:
                        if self._load_ocontours:
                            out_str = "{} does not have associated i- and o-contour data".format(full_filename)
                        else:
                            out_str = "{} does not have associated i-contour data".format(full_filename)
                        logging.warning(out_str)
                    continue

                # Get the full path to the DICOM file
                file_dicom = os.path.join(dicom_dir_full, file)

                # Get the full path to the associated contour file
                file_icontour = os.path.join(icontour_dir, 'IM-0001-{:04d}-icontour-manual.txt'.format(ID))

                # Add to the list
                if self._load_ocontours:
                    # If we need to load in o-contours, make the proper path to the
                    # associated file and add both i-contour and o-contour file in
                    file_ocontour = os.path.join(ocontour_dir, 'IM-0001-{:04d}-ocontour-manual.txt'.format(ID))
                    data_pairs.append((file_dicom, file_icontour, file_ocontour))
                else:
                    # Add just the i-contour file.  We add None as a third
                    # element of the tuple to make loading in the masks easier
                    data_pairs.append((file_dicom, file_icontour, None))

        return data_pairs

    def __load_image_and_mask(self, dicom_file, i_contour_file, o_contour_file=None):
        """
        Internal method.   Loads in the DICOM image and contour files at the
        given DICOM file and contour file paths

        Args:
            dicom_file: String containing a path to the DICOM file
            i_contour_file: String containing a path to a i-contour file
            o_contour_file: String containing a path to a o-contour file.
                            Optional.

        Returns:
            A tuple of DICOM image and the associated masked images with the contour
            data.  If no path to the o-contour file exists, then the returned
            mask in this case is set to None.  What is returned is a tuple
            of three elements with the DICOM image and i-contour mask being
            the first two elements and the third element can be None or a
            mask that contains the o-contour mask.
        """

        # Load in contour data for i-contour file
        coords = parsing.parse_contour_file(i_contour_file)

        # Return None, None, None if i-contour data is empty
        if not coords:
            if self._debug_mode:
                logging.warning("Coordinates for file {} are empty".format(i_contour_file))
            return None, None, None

        # Load in the image data
        dct = parsing.parse_dicom_file(dicom_file)

        # Return None, None, None if the image is invalid
        if dct is None:
            if self._debug_mode:
                logging.warning("Image data for file {} is invalid".format(dicom_file))
            return None, None, None

        # Get the actual image data a create the actual mask
        img = dct['pixel_data']

        # Make sure that the data is grayscale
        if len(img.shape) != 2:
            if self._debug_mode:
                logging.warning("Image data for file {} is not grayscale".format(dicom_file))
            return None, None, None

        # Get the mask data now
        (height, width) = img.shape
        i_mask = parsing.poly_to_mask(coords, width, height)

        # Repeat for o-contour data
        if o_contour_file is not None:
            coords = parsing.parse_contour_file(o_contour_file)

            if not coords:
                if self._debug_mode:
                    logging.warning("Coordinates for file {} are empty".format(o_contour_file))
                return None, None, None

            o_mask = parsing.poly_to_mask(coords, width, height)

        else:
            o_mask = None

        # Return the information now
        return img, i_mask, o_mask

    def __str__(self):
        """
        Pretty prints the object.  Doing print() on this object displays
        the information about the object.

        Returns:
            The string representation of the current status of the object
        """
        s = "DICOMLoader Properties:\n"
        s += "Base directory: {}\n".format(self._base_dir)
        s += "Size of dataset: {}\n".format(self._num_examples)
        s += "Batch Size: {}\n".format(self._batch_size)
        s += "Seed: {}\n".format("None" if self._seed is None else self._seed)
        s += "Return filenames with batch: {}\n".format("Yes" if self._return_filenames_with_batch else "No")
        s += "Load in o-contours: {}\n".format("Yes" if self._load_ocontours else "No")
        s += "Batch Number: {}\n".format(self._batch_number)
        if self._patient_IDs is not None:
            st = "Patient IDs: " + ", ".join(self._patient_IDs)
            s += st
        return s

    def __del__(self):
        """
        When the object goes out of context, we need to make sure
        we free all logging handlers
        """
        if self._debug_mode:
            logging.shutdown()

    def __iter__(self):
        """
        Override to allow for iterating over an instance

        Returns:
            An iterator to go over each batch of images and masks
        """

        # Provide an iterator and stop when None is returned
        return iter(self.next, None)

    def __len__(self):
        """
        Override to describe the total number of examples loaded in

        Returns:
            The size of the dataset.  Note that reflects potentially
            invalid data.  This contains the total number of correctly
            matched pairs of files between the DICOM images and contour
            data.
        """

        return len(self._data_pairs)

    def next(self):
        """
        Override to allow iteration over the instance, or you can manually
        call this method to get the next batch of examples

        Returns:
            A tuple that contains two, three or four elements depending on
            the situation:

            1. If the `return_filenames_with_batch` property is set to False
            (default), then the first element of the tuple is a 3D numpy array
            of size `batch_size` x `height` x `width` where `batch_size` is the set
            batch size and the `height` and `width` are the height and width
            of a single DICOM image.  The second element contains the corresponding
            masks demarcated by the i-contours that is the same size as the
            first element of the tuple but as of type `bool`.
            2. If the `return_filenames_with_batch` property is set to True, then
            the format is the same as scenario #1 but there is an additional
            element to the tuple at the end that contains a list of filenames.
            This list is a list of tuples that contain the corresponding filenames
            to the DICOM images and i-contour files in question for the batch.
            3. If the `return_filenames_with_batch` property is set to False
            (default) but it is desired to load in o-contours, then the first
            element of the tuple is a 3D numpy array of size `batch_size` x `height`
            x `width` containing the DICOM image data, the second being the
            masks for the i-contours and the third being the masks for the o-contours.
            The dimensions and type of the o-contours match the i-contours.
            4. If the `return_filenames_with_batch` property is set to True and
            the o-contours are desired to be loaded in, then similarly with scenario
            #2 an extra element is added at the end of the tuple of scenario #3
            where a list of tuples filenames is provided such that each tuple
            now contains a path to a DICOM image, i-contour and o-contour file.

        Raises:
            Exception: If any of the images do not share the same height and
                       width compared to the other images in the batch.
        """

        # First check if we have exhausted the examples
        if self._index >= self._num_examples:
            if self._debug_mode:
                logging.info("Reached the end of the dataset")
            # If we have, signal to stop iterating and
            # reset for next iteration
            self._index = 0
            return None

        # If we're at the beginning, produce a random permutation
        # of indices
        elif self._index == 0:
            # Produce a random permutation of indices
            # Don't bother shuffling if the number of examples is the batch size
            if self._num_examples == self._batch_size:
                self._indices = np.arange(self._num_examples).astype(np.int)
            else:
                self._indices = self._prng.permutation(self._num_examples)

            self._batch_number = 0

        if self._debug_mode:
            logging.info('Loading in batch number {}'.format(self._batch_number + 1))

        # Get the batch size as it may change per call
        batch_size = self._batch_size

        # Loop through each index, obtain the DICOM and
        # contour pair, load the data in and place
        # onto their corresponding lists
        imgs = []
        i_masks = []
        o_masks = [] # For o-contours

        filenames = []

        # Remembers how many valid examples we've seen so far
        count = 0

        # Add examples to the lists
        # Loop until we either hit the batch size or if we're
        # at the end of the examples
        while count < batch_size and self._index != self._num_examples:
            # Get a random index
            ind = self._indices[self._index]
            if self._debug_mode:
                logging.info('Loading in image: {}'.format(self._data_pairs[ind][0]))
                logging.info('Loading in i-contour: {}'.format(self._data_pairs[ind][1]))

                if self._load_ocontours:
                    logging.info('Loading in o-contour: {}'.format(self._data_pairs[ind][2]))

            # Load the pair of data
            img, i_mask, o_mask = self.__load_image_and_mask(*self._data_pairs[ind])

            # If nothing's wrong with it,try adding it in
            if img is not None and i_mask is not None:

                # First check to make sure all images are the same
                # height and width

                # Remember the height and width the first time we ever
                # load in an image
                if self._height is None:
                    self._height = img.shape[0]
                if self._width is None:
                    self._width = img.shape[1]

                # Keep comparing to this first one
                # If the images are all the same size, this exception
                # will never be thrown
                if (self._height, self._width) != img.shape:
                    out_str = "The DICOM images must all have the same height and width"
                    if self._debug_mode:
                        logging.error(out_str)
                    raise Exception(out_str)

                # Situation with o-contours
                if self._load_ocontours:
                    # If there's something wrong with the o-contours,
                    # then we skip this image and pair of contours
                    # all together
                    if o_mask is None:
                        if self._debug_mode:
                            logging.warning('The o-contour data is invalid')
                        continue

                    # We can finally add it in
                    imgs.append(img)
                    i_masks.append(i_mask)
                    o_masks.append(o_mask)
                else:
                    imgs.append(img)
                    i_masks.append(i_mask)

                # Also get the filenames for debugging
                if self._return_filenames_with_batch:
                    filenames.append(self._data_pairs[ind])
                count += 1 # Valid data count gets upped by 1
            else:
                if self._debug_mode:
                    logging.warning('The image and/or i-contour data is invalid')
            self._index += 1 # Number of examples seen gets upped by 1

        self._batch_number += 1 # Increase number of batches seen by 1
        if self._debug_mode:
            logging.info('Image batch size: {}'.format((count, self._height, self._width)))
            logging.info('Mask batch size: {}'.format((count, self._height, self._width)))

        # Return if the mininum batch size is met
        if count == batch_size:
            if self._return_filenames_with_batch:
                if self._load_ocontours:
                    return (np.array(imgs), np.array(i_masks), np.array(o_masks), filenames)
                else:
                    return (np.array(imgs), np.array(i_masks), filenames)
            else:
                if self._load_ocontours:
                    return (np.array(imgs), np.array(i_masks), np.array(o_masks))
                else:
                    return (np.array(imgs), np.array(i_masks))

        else: # Happens if we don't hit the batch size amount
            # Reshuffle for the next time and stop iterating
            if self._debug_mode:
                logging.warning('Insufficient number of examples for this batch. {} out of {}'.format(count, batch_size))
            self._index = 0
            return None

    def reset(self):
        """
        Resets the state of the random generator back to what it
        was before.  If a seed was specified, it will reset the
        random generator with this seed.  If not, then it will
        reset it back to the default seed.  It also resets the batch
        generator to what it was before it all started.
        """

        if self._debug_mode:
            logging.info("Resetting DICOM Loader")

        if self._seed is None:
            self._prng = RandomState()
        else:
            self._prng = RandomState(self._seed)

        # Ensures we reset index generation
        self._index = 0

    # Provide setters and getters
    @property
    def base_dir(self):
        """
        Contains base directory

        Returns:
            The base directory
        """
        return self._base_dir

    @property
    def batch_size(self):
        """
        Contains batch size

        Returns:
            The batch size
        """
        return self._batch_size

    @property
    def seed(self):
        """
        Contains the seed

        Returns:
            The seed
        """
        return self._seed

    @property
    def debug_mode(self):
        """
        Contains whether we want to turn on debug mode.  Turning on debug mode
        saves logging information into an output file called debug.log.  In order
        to write the contents to file, you must turn OFF debug mode prior to reading
        in the debug file contents.

        Returns:
            The debug mode status
        """
        return self._debug_mode

    @property
    def return_filenames_with_batch(self):
        """
        Contains whether to return the DICOM and contour filenames in
        addition to the batch of images and masks.  Primarily for
        debugging.  When iterating over the instance or using the
        next() method, a third element is attached to each tuple
        in the list which contains a pair of filenames that are the
        DICOM image and i-contour file respectively

        Returns:
            The status of returning the filenames with the batch
        """
        return self._return_filenames_with_batch

    @batch_size.setter
    def batch_size(self, value):
        """
        Sets the batch size

        Args:
            value: The batch size value as an int

        Raises:
            ValueError: If the batch size is 0 or negative
        """

        if self._debug_mode:
            logging.info('Setting batch size to {}'.format(value))

        if value <= 0:
            if self._debug_mode:
                logging.error("Batch size is 0 or negative")
            raise ValueError("Batch size is 0 or negative")

        self._batch_size = int(value)

        # Safe guard
        # If the batch size exceeds the number of examples
        # cap to the number of examples
        if self._batch_size > self._num_examples:
            self._batch_size = self._num_examples

    @seed.setter
    def seed(self, value):
        """
        Sets the seed

        Args:
            value: The seed as an int or set to None to set to default

        Raises:
            ValueError: If the seed is negative
        """

        if self._debug_mode:
            logging.info('Setting seed to {}'.format(value))

        # If the seed to set is not None...
        if value is not None:
            try: # Try converting to integer
                # If already an integer, great!
                value = int(value)
                # Ensure positive seed
                if value < 0:
                    if self._debug_mode:
                        logging.info("Seed is negative")

                    raise ValueError("Seed is negative")
            # Throw exception if not an integer
            except ValueError as e:
                raise ValueError("Improper seed value: {}".format(value))

        # Finally set the seed
        self._seed = value

        # Does the actual reset
        self.reset()

    @return_filenames_with_batch.setter
    def return_filenames_with_batch(self, value):
        """
        Sets whether to return the DICOM and contour filenames in
        addition to the batch of images and masks.  Primarily for
        debugging.  When iterating over the instance or using the
        next() method, a third element is attached to each tuple
        in the list which contains a pair of filenames that are the
        DICOM image and i-contour file respectively.

        Args:
            value: Set to True or False to enable/disable

        Raises:
            ValueError: If the input is not bool
        """

        if self._debug_mode:
            logging.info('Setting return_filenames_with_batch to {}'.format(value))

        # Check if the input is a bool
        if not isinstance(value, bool):
            if self._debug_mode:
                logging.error("{} is not of type bool".format(value))
            raise ValueError("{} is not of type bool".format(value))

        self._return_filenames_with_batch = value

    @debug_mode.setter
    def debug_mode(self, value):
        """
        Determines whether we want to turn on debug mode.  Turning on debug mode
        saves logging information into an output file called debug.log.  In order
        to write the contents to file, you must turn OFF debug mode prior to reading
        in the debug file contents.  However, if the DICOMLoader instance goes out
        of scope or if there is nothing that is retaining a reference to the
        instance, the debug file is automatically written and closed.

        Args:
            value: Set to True or False to enable/disable
        Returns:
            ValueError: The debug mode status

        Raises:
            If the input is not bool
        """

        if not isinstance(value, bool):
            raise ValueError("{} is not of type bool".format(value))

        # Case #1 - If the present flag is True and incoming is False
        if self._debug_mode and not value:
            logging.info('Turning off debug mode')
            logging.shutdown()

        # Case #2 - If the present flag is False and incoming is True
        elif not self._debug_mode and value:
            # Note: Default behaviour is to append to the file
            logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='debug.log')
            logging.info('Turning on debug mode')

        # Anything else, don't bother doing anything
        self._debug_mode = value

if __name__ == '__main__':
    # Tester code
    dl = DICOMLoader('data', debug_mode=True)
    dl.seed = 42
    dl.batch_size = 16

    # Read first batch
    (imgs, i_masks) = dl.next()

    # Pretty print
    print(dl)

    print("\nSize of first image batch: {}".format(imgs.shape))
    print("Size of first i-contour mask batch: {}\n".format(i_masks.shape))

    print("Iterate through the rest of the batches...")
    for (imgs, i_masks) in dl:
        print("Size of image batch: {}".format(imgs.shape))
        print("Size of i-contour mask batch: {}".format(i_masks.shape))
        # Show the status at each iteration
        print(dl) # Batch number should keep increasing
        print()

    # Turn off debugging now
    dl.debug_mode = False

    # Create new object that contains o-contour data
    dl = DICOMLoader('data', load_ocontours=True, debug_mode=True)
    dl.seed = 60600
    dl.batch_size = 10

    # Read first batch
    (imgs, i_masks, o_masks) = dl.next()

    # Pretty print
    print(dl)

    print("\nSize of first image batch: {}".format(imgs.shape))
    print("Size of first i-contour mask batch: {}\n".format(i_masks.shape))
    print("Size of first o-contour mask batch: {}\n".format(o_masks.shape))

    print("Iterate through the rest of the batches...")
    for (imgs, i_masks, o_masks) in dl:
        print("Size of image batch: {}".format(imgs.shape))
        print("Size of i-contour mask batch: {}".format(i_masks.shape))
        print("Size of o-contour mask batch: {}".format(o_masks.shape))
        # Show the status at each iteration
        print(dl) # Batch number should keep increasing
        print()

    # Turn off debugging now
    dl.debug_mode = False
