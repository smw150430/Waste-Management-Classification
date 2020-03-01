images = np.concatenate((images_tr, images_te, images_mod_tr, images_mod_te))
labels = np.concatenate((labels_tr[:,0], labels_te[:,0], labels_mod_tr[:,0], labels_mod_te[:, 0]))

X_model, X_test, y_model, y_test = train_test_split(images, labels, test_size = 0.2, random_state = 123)
X_train, X_val, y_train, y_val = train_test_split(X_model, y_model, test_size = 0.2, random_state = 123)

