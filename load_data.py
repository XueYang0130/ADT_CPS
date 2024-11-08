import numpy as np
import pandas as pd
from sklearn import preprocessing

if __name__ == '__main__':
    # load datafile that contains only normal
    print("============================================normal===================================")
    f1 = pd.read_excel("..//dataset//SWaT//SWaT_Dataset_Normal_v1.xlsx")
    normal = pd.DataFrame(f1)
    # drop the 1st column (timesteps) and the last column (Normal/Attack).
    normal=normal.drop(normal.columns[[0,-1]], axis=1)
    # Transform all columns into float64
    normal = normal.astype(float)
    print("normal shape:", pd.DataFrame(normal).shape)
    print(pd.DataFrame(normal).head(2))

    # load datafile that contains both normal and abnormal
    print("===========================================attack===========================================")
    f2 = pd.read_excel("..//dataset//SWaT//SWaT_Dataset_Attack_v0.xlsx")
    attack = pd.DataFrame(f2)
    # get labels (0-normal; 1-attack)
    labels = attack[attack.columns[-1]].values.tolist()
    y_all = np.array([0 if i == "Normal" else 1 for i in labels])
    # drop the 1st column (timesteps) and the last column (Normal/Attack).
    attack = attack.drop(attack.columns[[0, -1]], axis=1)
    # Transform all columns into float64
    attack = attack.astype(float)
    print("attack shape",attack.shape)
    print("label len",len(y_all))
    print(pd.DataFrame(attack).head(2))

    #normalized all data together and then divide them
    print("======================================combine==>normalize==>divide========================================")
    all_data = pd.DataFrame(np.vstack((normal, attack)))
    #all_data = pd.concat([normal, attack])
    print(all_data.shape)
    # normalize
    min_max_scaler = preprocessing.MinMaxScaler()
    all_data_scaled = min_max_scaler.fit_transform(all_data.values)
    #divide
    x_normal_scaled=pd.DataFrame(all_data_scaled[:495000,:])
    x_attack_scaled=pd.DataFrame(all_data_scaled[495000:,:])
    print("x_normal_scaled shape", x_normal_scaled.shape)
    print("x_attack_scaled shape", x_attack_scaled.shape)
    print("label len:",len(y_all),"anomaly num:",sum(y_all))

    # get sliding windows with length=12
    window_size = 12
    input_size = 51
    windows_normal = x_normal_scaled.values[
        np.arange(window_size)[None, :] + np.arange(x_normal_scaled.shape[0] - window_size)[:, None]]
    windows_attack = x_attack_scaled.values[
        np.arange(window_size)[None, :] + np.arange(x_attack_scaled.shape[0] - window_size)[:, None]]
    y_windows =y_all[np.arange(window_size)[None, :] + np.arange(len(y_all) - window_size)[:, None]]
    y_labels=[1 if np.sum(i)>0 else 0 for i in y_windows]
    print("normal windows shape:", windows_normal.shape, "all windows shape:", windows_attack.shape, "label",len(y_labels), "anomalies:", np.sum(y_labels))

    #flatten
    windows_normal_flatten=windows_normal.reshape(windows_normal.shape[0], 12*51)
    windows_attack_flatten = windows_attack.reshape(windows_attack.shape[0], 12 * 51)
    print("normal windows shape:", windows_normal_flatten.shape, "all windows shape:", windows_attack_flatten.shape)

    # save datasets
    np.save("datasets//x_attack_scaled.npy", x_attack_scaled)
    np.save("datasets//x_normal_scaled.npy", x_normal_scaled)
    np.save("datasets//windows_normal_flatten.npy", windows_normal_flatten)
    np.save("datasets//windows_attack_flatten.npy", windows_attack_flatten)
    np.save("datasets//windows_attack_labels.npy", y_labels)

