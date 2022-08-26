import os
import glob
files_1 = glob.glob('/home/rutu/WPI/Directed_Research/ReID_Datasets/VeRi/Dsl/*')
files_2 = glob.glob('/home/rutu/WPI/Directed_Research/ReID_Datasets/VeRi/Dsl_test/*')
files_3 = glob.glob('/home/rutu/WPI/Directed_Research/ReID_Datasets/VeRi/Dsl_query/*')
files_4 = glob.glob('/home/rutu/WPI/Directed_Research/ReID_Datasets/VeRi/Dsl2/*')
files_5 = glob.glob('/home/rutu/WPI/Directed_Research/ReID_Datasets/VeRi/Dsl2_test/*')
files_6 = glob.glob('/home/rutu/WPI/Directed_Research/ReID_Datasets/VeRi/Dsl2_query/*')

for f in files_1:
    os.remove(f)
for g in files_2:
    os.remove(g)
for h in files_3:
    os.remove(h)
for i in files_4:
    os.remove(i)
for j in files_5:
    os.remove(j)
for k in files_6:
    os.remove(k)