import matplotlib.pyplot as plt
import seaborn as sns

def plot_target_distribution(df, target_column):
    sns.countplot(data=df, x=target_column)
    plt.title("Target Variable Distribution")
    plt.show()
