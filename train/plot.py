import pandas as pd
import matplotlib.pyplot as plt

def plot_loss(path:str):
    file_path = path+"loss.csv"
    df = pd.read_csv(file_path)
    loss = pd.concat([df.iloc[i, :] for i in range(len(df))],0)
    idx = range(loss.shape[0])
    plt.plot(idx,loss)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    # plt.show()
    plt.savefig('train/add_mask.png')

if __name__ == "__main__":
    plot_loss('train\\add_mask\\')
