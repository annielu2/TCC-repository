import pandas as pd
import glob

def get_data():
    csv_files = glob.glob('CachacaNER/csv/particao_*.csv')

    df_train = []
    df_test = []
    df_val = []

    partition = 1

    for file in csv_files:
        if(partition <= 7):
            df_train.append(pd.read_csv(file))
        elif(partition == 8):
            df_val.append(pd.read_csv(file))
        else:
            df_test.append(pd.read_csv(file))
        
        partition+=1

    train_dataset = pd.concat(df_train, ignore_index=True)
    val_dataset = pd.concat(df_val, ignore_index=True)
    test_dataset = pd.concat(df_test, ignore_index=True)

    return train_dataset, val_dataset, test_dataset

train_dataset, val_dataset, test_dataset = get_data()
print(train_dataset)
