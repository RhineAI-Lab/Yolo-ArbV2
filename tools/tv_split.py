import os

tv_rate = 9 # 训练集是测试集的X倍
target_path = "E:\\dota_yolo"

f_test  = open(target_path+'\\val.txt', 'w')
f_train = open(target_path+'\\train.txt', 'w')
train_num = 0
test_num = 0
for i,name in enumerate(os.listdir(target_path+"\\images")):
    if i%tv_rate==0:
        f_test.write(target_path+'\\images\\' + name + '\n')
        test_num+=1
    else:
        f_train.write(target_path+'\\images\\' + name + '\n')
        train_num+=1
f_train.close()
f_test.close()
print("Classify complete. Train:"+str(train_num)+" Test:"+str(test_num))