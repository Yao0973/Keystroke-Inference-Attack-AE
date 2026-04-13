import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# confusion matrix data（rows:true labels.*predicted labels）
conf_matrix = np.array([
    [186,   0,   0,   0,   0,   1,   0,   0,  59,  54],
    [  0, 246,  23,   3,   0,   0,  26,   2,   0,   0],
    [  0,   5, 214,   5,   0,   1,   0,   0,  59,  16],
    [  0,   1,  39, 252,   0,   0,   0,   0,   6,   2],
    [  0,   0,   3,   0, 230,   0,   7,  60,   0,   0],
    [  9,   6,  37,   1,   2, 167,   0,   0,  78,   0],
    [  0,  12,   0,   0,   5,   0, 227,  56,   0,   0],
    [  0,   0,   0,   0,  23,   0,  26, 251,   0,   0],
    [ 26,   0,  24,   1,   0,  11,   0,   0, 187,  51],
    [ 12,   0,   4,   0,   0,   0,   0,   0,  52, 232]
])



# class labels（0 to 9）
labels = [str(i) for i in range(10)]

# calculate row totals
row_totals = conf_matrix.sum(axis=1)

# calculate recall for each class（accuracy）and average
diag_elements = np.diag(conf_matrix)  # extract diagonal elements（correct predictions count）
recall_rates = diag_elements / row_totals  # calculate recall for each class
average_accuracy = np.mean(recall_rates)  # calculate average accuracy（macro average）
print(f"Average accuracy: {average_accuracy:.4f}")  # print results，keep four decimal places

# create new array for percentages
conf_matrix_percent = np.zeros_like(conf_matrix, dtype=float)
for i in range(len(row_totals)):
    if row_totals[i] > 0:
        conf_matrix_percent[i, :] = conf_matrix[i, :] / row_totals[i] * 100

# set figure size（flatter ratio suitable for single column）
plt.figure(figsize=(8, 4))  # modify this line

# use.*to plot heatmap
sns.set(font_scale=0.9)  # appropriately reduce font size
# set global fontto Times New Roman
plt.rcParams["font.family"] = ["Times New Roman", "serif"]
ax = sns.heatmap(
    conf_matrix_percent,  # use percentage matrix directly
    annot=True, 
    fmt='.1f',  # only show percentages，keep one decimal place - format specifier already fixed
    cmap='Blues', 
    xticklabels=labels, 
    yticklabels=labels, 
    cbar_kws={'label': 'Percentage (%)'},  # update colorbar label
    linewidths=0.3,  # set cell border widthto1.0
    linecolor='black',  # set cell border colorto black
    annot_kws={'size': 13}  # increase annotation text font size(default about 10)
)

# adjust colorbar label font size（increase by one）
cbar = ax.collections[0].colorbar
cbar.set_label('Percentage (%)', fontsize=14)  # set font size to14(larger than default)
# increase colorbar tick font sizeset (14, consistent with axis tick font)
cbar.ax.tick_params(labelsize=13)

# add outer border
for _, spine in ax.spines.items():
    spine.set_visible(True)
    spine.set_linewidth(1.5)
    spine.set_color('black')

# set title and axis labels
# plt.title('Confusion Matrix with Percentages', fontsize=16)
plt.xlabel('Predicted Digit', fontsize=16)
plt.ylabel('Actual Digit', fontsize=16)

# rotate y-axis numbers90°
plt.yticks(rotation=0, fontsize=14)  # adjust tick font sizeadjust tick font size
plt.xticks(fontsize=14)  # add this lineset x-axis tick font size

# adjust layout and display image
plt.tight_layout()
plt.show()