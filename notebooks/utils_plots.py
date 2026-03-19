import matplotlib.pyplot as plt
import matplotlib as mpl

def set_publication_style():
    try:
        plt.style.use('seaborn-v0_8-paper')
    except OSError:
        pass  # Fallback to default

    params = {
        'font.family': 'serif',
        'font.serif': ['Palatino Linotype', 'Palatino', 'Times New Roman', 'DejaVu Serif'],
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        
        'text.usetex': False,  
        'mathtext.fontset': 'stix', 
        
        'figure.figsize': (10, 6),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'lines.linewidth': 1.5,
        
        'axes.prop_cycle': mpl.cycler(color=['000000', 'E24A33', '348ABD', '988ED5', '777777', 'FBC15E', '8EBA42', 'FFB5B8']),
    }
    plt.rcParams.update(params)

def setup_notebook():
    """configures notebook display settings"""
    set_publication_style()
    # %config InlineBackend.figure_format = 'retina' 
