import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Example data: percentages for each age group per year
data = {
    'Year': ["2007", "2017", "2024", "2030", "2050"],
    'Under 5':   [7, 6, 5, 4, 3],
    '5 to 14':   [10, 9, 8, 7, 6],
    '15 to 49':  [60, 58, 56, 50, 40],
    '50 to 70':  [20, 22, 25, 30, 35],
    'Over 70':   [3, 5, 6, 9, 16],
}
df = pd.DataFrame(data)
df.set_index('Year', inplace=True)

age_groups = ['Over 70', '50 to 70', '15 to 49', '5 to 14', 'Under 5']  # Stacked order (bottom to top)
colors = ['#08306b', '#2171b3', '#6baed6', '#9ecae1', '#c6dbef']  # Adjust to your palette

fig, ax = plt.subplots(figsize=(8, 6))

# Start at zero, stack up
bottom = np.zeros(len(df))
bars = []
for idx, group in enumerate(age_groups):
    bar = ax.bar(df.index, df[group], bottom=bottom, label=group, color=colors[idx], width=0.7, edgecolor="white")
    bars.append(bar)
    # Add label inside bar
    for rect in bar:
        height = rect.get_height()
        if height > 0:
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_y() + height / 2,
                f"{height:.2f}%" if height < 10 else f"{height:.2f}%" if height < 100 else f"{int(height)}%",
                ha='center', va='center',
                fontsize=11, color='black' if idx < 2 else 'white', fontweight='bold'
            )
    bottom += df[group].values

# Styling
ax.set_title('T2D Cases Among PLHIV by Age Group', fontsize=16, fontweight='bold')
ax.set_ylabel('Percentage of T2D Cases')
ax.set_xlabel('')
ax.set_ylim(0, 100)
ax.legend(title='Age Group', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
ax.set_xticks(df.index)
ax.set_xticklabels(df.index, fontsize=12)
ax.tick_params(left=False, bottom=False)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.show()