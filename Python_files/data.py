data_te = ImageDataGenerator(rescale = 1./255).flow_from_directory('DATASET_2/TEST', 
                                                                   target_size = (224, 224), 
                                                                   batch_size = 2000,
                                                                   seed = 123)

data_tr = ImageDataGenerator(rescale = 1./255).flow_from_directory('DATASET_2/TRAIN', 
                                                                   target_size = (224, 224),  
                                                                   batch_size = 4000,
                                                                   seed = 123)

mod_te = ImageDataGenerator(rescale = 1./255, 
                            rotation_range = 360, 
                            width_shift_range = 0.3, 
                            height_shift_range = 0.3, 
                            brightness_range = [0.2, 1.0], 
                            horizontal_flip = True, 
                            vertical_flip = True, 
                            zoom_range = [0.5, 1.5]).flow_from_directory('DATASET_2/TEST', 
                                                                   target_size = (224, 224), 
                                                                   batch_size = 1000,
                                                                   seed = 123)

mod_tr = ImageDataGenerator(rescale = 1./255, 
                            rotation_range = 360, 
                            width_shift_range = 0.3, 
                            height_shift_range = 0.3, 
                            brightness_range = [0.2, 1.0], 
                            horizontal_flip = True, 
                            vertical_flip = True, 
                            zoom_range = [0.5, 1.5]).flow_from_directory('DATASET_2/TRAIN', 
                                                                   target_size = (224, 224), 
                                                                   batch_size = 2000,
                                                                   seed = 123)

images_te, labels_te = next(data_te)
images_tr, labels_tr = next(data_tr)

images_mod_te, labels_mod_te = next(mod_te)
images_mod_tr, labels_mod_tr = next(mod_tr)

