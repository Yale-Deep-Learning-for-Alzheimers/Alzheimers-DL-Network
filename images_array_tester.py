import pickle

MRI_images_list = pickle.load(open("./Data/Combined_MRI_List.pkl", "rb"))
sample = MRI_images_list[0]
print(sample)