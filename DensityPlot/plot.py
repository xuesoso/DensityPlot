import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits import axes_grid1
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as mticker
from scipy import ndimage

class FormatScalarFormatter(mticker.ScalarFormatter):
    def __init__(self, fformat="%1.1f", offset=True, mathText=True):
        self.fformat = fformat
        mticker.ScalarFormatter.__init__(self, useOffset=offset,
                                            useMathText=mathText)

    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = '$%s$' % mticker._mathdefault(self.format)

class MathTextSciFormatter(mticker.Formatter):
    '''
    This formatter can be fed to set ticklabels in scientific notation without
    the annoying "1e" notation (why would anyone do that?).
    Instead, it formats ticks in proper notation with "10^x".

    fmt: the decimal point to keep
    Usage = ax.yaxis.set_major_formatter(MathTextSciFormatter("%1.2e"))
    '''

    def __init__(self, fmt="%1.2e"):
        self.fmt = fmt
    def __call__(self, x, pos=None):
        s = self.fmt % x
        decimal_point = '.'
        positive_sign = '+'
        tup = s.split('e')
        significand = tup[0].rstrip(decimal_point)
        sign = tup[1][0].replace(positive_sign, '')
        exponent = tup[1][1:].lstrip('0')
        if exponent:
            exponent = '10^{%s%s}' % (sign, exponent)
        if significand and exponent:
            s =  r'%s{\times}%s' % (significand, exponent)
        else:
            s =  r'%s%s' % (significand, exponent)
        return "${}$".format(s)

def density2d(data_plot,
              ax=None,
              bins=300,
              mode='scatter_mesh',
              normed=False,
              smooth=True,
              logz=True,
              sigma=10.0,
              colorbar=False,
              xscale='linear',
              yscale='linear',
              xlabel=None,
              ylabel=None,
              xlim=None,
              ylim=None,
              title=None,
              filename=None,
              alpha=.1,
              cmap='Reds',
              min_threshold=0.01,
              figsize=(3, 3),
              transparent=False,
              colorbar_fmt='math',
              mesh_order='top',
              dot_alpha=1,
              dpi=200,
              **kwargs):
    """
    ## Modified based on density2d from FlowCal package.
    Plot a 2D density plot from two channels of a flow cytometry data set.

    `density2d` has three plotting modes which are selected using the `mode`
    argument. With ``mode=='mesh'``, this function plots the data as a true
    2D histogram, in which a plane is divided into bins and the color of
    each bin is directly related to the number of elements therein. With
    ``mode=='scatter'``, this function also calculates a 2D histogram,
    but it plots a 2D scatter plot in which each dot corresponds to a bin,
    colored according to the number elements therein. The most important
    difference is that the ``scatter`` mode does not color regions
    corresponding to empty bins. This allows for easy identification of
    regions with low number of events. With ``mode=='scatter_mesh'``, this
    function will first generate 2D histogram and plot it on a mesh, then will
    plot all the data as scatter dots underneath. For all modes, the calculated
    histogram can be smoothed using a Gaussian kernel by specifying
    ``smooth=True``. The width of the kernel is, in this case, given by
    `sigma`. By default, bin with z-value below the minimal threshold will not
    be plotted with a color in order to prevent obscuring of the data. The
    threshold can be adjusted by changing the value of ``min_threshold``.

    Parameters
    ----------
    data : numpy array
        A N x 2 data array to plot.
    bins : int, optional
        Default value set to 300. Should adjust based on density of the data
        over the entire plane.
    mode : {'mesh', 'scatter', 'scatter_mesh'}, str, optional
        Plotting mode. 'mesh' produces a 2D-histogram whereas 'scatter'
        produces a scatterplot colored by histogram bin value. 'scatter_mesh'
        produces a 2D-histogram with scatter dots of the data on top.
    normed : bool, optional
        Flag indicating whether to plot a normed histogram (probability
        mass function instead of a counts-based histogram).
    smooth : bool, optional
        Flag indicating whether to apply Gaussian smoothing to the
        histogram.
    logz : bool, optional
        Flag indicating whether to log transform the z axis values of 2D-
        histogram.
    colorbar : bool, optional
        Flag indicating whether to add a colorbar to the plot.
    filename : str, optional
        The name of the file to save the figure to. If None, do not save.

    Other parameters
    ----------------
    sigma : float, optional
        The sigma parameter for the Gaussian kernel to use when smoothing.
    xscale : str, optional
        Scale of the x axis, either ``linear``, ``log``, ```symlog```,
        ```logit```
    yscale : str, optional
        Scale of the y axis, either ``linear``, ``log``, ```symlog```,
        ```logit```
    xlabel : str, optional
        Label to use on the x axis. If None, attempts to extract channel
        name from `data`.
    ylabel : str, optional
        Label to use on the y axis. If None, attempts to extract channel
        name from `data`.
    xlim : tuple, optional
        Limits for the x axis. If not specified and `bins` exists, use
        the lowest and highest values of `bins`.
    ylim : tuple, optional
        Limits for the y axis. If not specified and `bins` exists, use
        the lowest and highest values of `bins`.
    title : str, optional
        Plot title.
    colorbar_fmt : str, optional
        The formatting of the tick mark. Available options are either
        ```math``` or ```scalar```
    mesh_order : str, optional
        The order to plot mesh for ```scatter_mesh```. Either ```top``` or
        ```bottom```
    dot_alpha : float, optional
        Opacity value for dots in ```scatter_mesh```
    dpi : int, optional
        Resolution of saved figure.
    kwargs : dict, optional
        Additional parameters passed directly to the underlying matplotlib
        functions: ``plt.scatter`` if ``mode==scatter``, and
        ``plt.pcolormesh`` if ``mode==mesh``.

    """
    import warnings
    warnings.filterwarnings("ignore")
    if not ax:
        existing_plot = False
        fig, ax = plt.subplots(figsize=figsize)
    else:
        existing_plot = True

    # Calculate histogram
    H,xe,ye = np.histogram2d(data_plot[:,0], data_plot[:,1], bins=bins)

    # Smooth
    if smooth:
        sH = ndimage.filters.gaussian_filter(
            H,
            sigma=sigma,
            order=0,
            mode='constant',
            cval=0.0)
    else:
        sH = None

    # Normalize
    if normed:
        H = H / np.sum(H)
        sH = sH / np.sum(sH) if sH is not None else None

    ###
    # Plot
    ###

    # numpy histograms are organized such that the 1st dimension (eg. FSC) =
    # rows (1st index) and the 2nd dimension (eg. SSC) = columns (2nd index).
    # Visualized as is, this results in x-axis = SSC and y-axis = FSC, which
    # is not what we're used to. Transpose the histogram array to fix the
    # axes.
    H = H.T
    sH = sH.T if sH is not None else None

    if mode == 'scatter':
        Hind = np.ravel(H)
        xc = (xe[:-1] + xe[1:]) / 2.0   # x-axis bin centers
        yc = (ye[:-1] + ye[1:]) / 2.0   # y-axis bin centers
        xv, yv = np.meshgrid(xc, yc)
        x = np.ravel(xv)[Hind != 0]
        y = np.ravel(yv)[Hind != 0]
        z = np.ravel(H if sH is None else sH)[Hind != 0]
        artist = ax.scatter(x, y, edgecolor='none', c=z, **kwargs)
    elif mode == 'mesh':
        if logz:
            sH = np.log10(sH)
            min_mesh_val = np.max(sH) + np.log10(min_threshold)
        else:
            min_mesh_val = np.max(sH)*min_threshold
        sH[sH <= min_mesh_val] = np.nan
        artist = ax.pcolormesh(xe, ye, H if sH is None else sH, alpha=alpha,
                      edgecolors='face', cmap=cmap)
    elif mode == 'scatter_mesh':
        if logz:
            sH = np.log10(sH)
            min_mesh_val = np.max(sH) + np.log10(min_threshold)
        else:
            min_mesh_val = np.max(sH)*min_threshold
        sH[sH <= min_mesh_val] = np.nan
        if mesh_order == 'top':
            zorder = np.inf
        else:
            zorder = 0
        artist = ax.pcolormesh(xe, ye, H if sH is None else sH, alpha=alpha,
                               zorder=zorder, edgecolors='face', cmap=cmap)
        ax.scatter(data_plot[:, 0], data_plot[:, 1], edgecolor='none',
                   alpha=dot_alpha, **kwargs)
    else:
        raise ValueError("mode {} not recognized".format(mode))

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(artist, cax=cax, )
        if normed:
            if logz:
                cbar.ax.set_title(r'Probability (log$_{10}$)', ha='left')
            else:
                cbar.ax.set_title('Probability', ha='left')
        else:
            cbar.ax.set_title('Counts', ha='left')
        if colorbar_fmt == 'math':
            cbar.ax.yaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))
        elif colorbar_fmt == 'scalar':
            cbar.ax.yaxis.set_major_formatter(FormatScalarFormatter("%.2f"))

    # x and y limits
    if xlim is not None:
        # Highest priority is user-provided limits
        ax.set_xlim(xlim)
    else:
        if existing_plot is False:
            # Use histogram edges
            ax.set_xlim((xe[0], xe[-1]))

    if ylim is not None:
        # Highest priority is user-provided limits
        ax.set_ylim(ylim)
    else:
        if existing_plot is False:
            # Use histogram edges
            ax.set_ylim((ye[0], ye[-1]))

    if title is not None:
        ax.set_title(title)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if xlabel is not None:
        ax.set_ylabel(ylabel)

    ax.set_yscale(yscale)
    ax.set_yscale(xscale)

    if filename:
        fig.savefig(filename, bbox_inches='tight', dpi=dpi,
                    transparent=transparent)
    if existing_plot is False:
        return (fig, ax)
