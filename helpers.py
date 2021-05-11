# TODO:
# - make a get_df_info(): which use all the normal key stats
# 		--> info(), descrive(), head(), unqiue()
# - make a "put into bins" panda function
# count, division = np.histogram(df["target"], bins=20)
# df["group"] = np.digitize(df["target"], division)
# df.group.hist(bins=20)
# 
# Make a function that can plot the average and uncertainty for e.g. CV
# Fix save_pickle such that it just takes a path


from .__code._helpers import *

