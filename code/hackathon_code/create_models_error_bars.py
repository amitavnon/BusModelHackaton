from matplotlib import pyplot as plt
import numpy as np


def plot_error_bars(train_scores, test_scores):
    train_scores = {k: round(v, 2) for k, v in train_scores.items()}
    test_scores = {k: round(v, 2) for k, v in test_scores.items()}

    models = list(train_scores.keys())
    train_values = list(train_scores.values())
    test_values = list(test_scores.values())

    x = np.arange(len(models))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size as needed
    rects1 = ax.bar(x - width/2, train_values, width, label='Train Error')
    rects2 = ax.bar(x + width/2, test_values, width, label='Test Error')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Models')
    ax.set_ylabel('Scores')
    ax.set_title('Train and Test Errors by Models')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')  # Rotate labels by 45 degrees
    ax.legend()

    # Function to add labels on the bars
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.show()