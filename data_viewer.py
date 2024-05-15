from PIL import Image
import os
import matplotlib.pyplot as plt
import pandas as pd

def look_at_df(df, image_folder, sort_by=None, ascending=True):

    if sort_by not in ["perspective_score_hood", "perspective_score_backdoor_left"]:
        raise KeyError("No column with spicified name")
    elif sort_by:
        df = df.sort_values(sort_by, ascending=ascending)
    #todo: check if dataset is equally distributed
    df["tot"] =  df["perspective_score_hood"]+ df["perspective_score_backdoor_left"]
    print(df["perspective_score_hood"].mean(), df["perspective_score_backdoor_left"].mean(),  df["tot"].mean())
    print(df["perspective_score_hood"].std(), df["perspective_score_backdoor_left"].std(), df["tot"].std())
    print(df["perspective_score_hood"].max(), df["perspective_score_backdoor_left"].max(),  df["tot"].max())
    print(df["perspective_score_hood"].min(), df["perspective_score_backdoor_left"].min(),  df["tot"].max())

    #no probability distribution possible can depict backdoor and hood at once
    #is more or less equally distributed and shouldnt introduce a strong bias
    print(df.head())
    for i,row in df.iterrows():
        print(row)
        plt.imshow(Image.open(os.path.join(image_folder, row["filename"])))
        plt.show()


if __name__ == '__main__':
    df = pd.read_csv("data/car_imgs_4000.csv")
    look_at_df(df, "data/imgs", "perspective_score_hood")