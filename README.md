1. Make sure to save all these scripts in a parent folder containing another folder named "training_data"
2. You will need the required libraries and the endfb-viii.0-hdf5 material data.
3. Generate data in batches and save to a "batch folder" using the BEAVRScopy.py. You will have to change the batch folder directory to exist within the training_data folder
4. Once all the data is generated in batches, the consolidate_training.py will create final .npy files with consolidated data from all the batches (this is nice if certain batches become corrupt)
5. Now run the FM_cleaner.py, this will remove all zeros and store the masks within training_data (Note: you will also have to change the assignment of "data_dir" to the training_data folder within your system)
6. With a cleaned dataset and a stored mask, run the nn1HiddenInterpolClean.py script. You will also have to change the directories within this file to match wherever your data is stored. I would suggest keeping all data in a single "training_data"

