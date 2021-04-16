from tensorflow.keras.preprocessing.image import ImageDataGenerator

gen = ImageDataGenerator(height_shift_range=0.2, width_shift_range=0.2, zoom_range=0.2)
gen.flow_from_directory("test", target_size=(150, 150), save_to_dir="testsave")