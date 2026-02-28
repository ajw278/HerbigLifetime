import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Bitstream Vera Serif']
plt.rcParams['text.usetex'] = True

plt.rcParams['font.size'] = 12

# Ensure ticks appear on all four sides of the axes
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True

# Set ticks to be inside the plot
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# Set tick size
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['ytick.minor.size'] = 3

# Add minor ticks 
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True

# Figure default sizes
plt.rcParams['figure.figsize'] = [6.4, 4.8]  # Default figure size
plt.rcParams['axes.labelsize'] = 14  # Larger labels for axes
plt.rcParams['axes.titlesize'] = 16  # Larger title size
plt.rcParams['axes.linewidth'] = 1.5  # Thicker axes lines

# Set grid style 
plt.rcParams['grid.alpha'] = 0.7  # Slight transparency for grid lines
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.7

# Remove grid by default
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.grid.which'] = 'major'

# Colour and line style customization
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['lines.markersize'] = 6

# Set default legend font size to 10
plt.rcParams['legend.fontsize'] = 10


CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

