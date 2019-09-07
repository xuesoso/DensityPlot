import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits import axes_grid1
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as mticker
from scipy import ndimage
import scipy.optimize

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

def density2d(data_plot=None,
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
              cmap='viridis',
              min_threshold=0.01,
              figsize=(3, 3),
              transparent=False,
              colorbar_fmt='math',
              mesh_order='top',
              dot_alpha=1,
              dpi=200,
              x=None,
              y=None,
              **kwargs):
    """
    ## Modified based on density2d from FlowCal package.
    Plot a 2D density plot from two channels of a flow cytometry data set.

    `density2d` has four plotting modes which are selected using the `mode`
    argument. With ``mode=='mesh'``, this function plots the data as a true
    2D histogram, in which a plane is divided into bins and the color of
    each bin is directly related to the number of elements therein. With
    ``mode=='contour'``, this function plots the data as a contour map based on
    the 2D histogram of the data values. With ``mode=='scatter'``, this
    function also calculates a 2D histogram, but it plots a 2D scatter plot in
    which each dot corresponds to a bin, colored according to the number
    elements therein. The most important difference is that the ``scatter``
    mode does not color regions corresponding to empty bins. This allows for
    easy identification of regions with low number of events. With
    ``mode=='scatter_mesh'``, this function will first generate 2D histogram
    and plot it on a mesh, then will plot all the data as scatter dots
    underneath. For all modes, the calculated histogram can be smoothed using
    a Gaussian kernel by specifying ``smooth=True``. The width of the kernel
    is, in this case, given by `sigma`. By default, bin with z-value below the
    minimal threshold will not be plotted with a color in order to prevent
    obscuring of the data. The threshold can be adjusted by changing the value
    of ``min_threshold``.

    Parameters
    ----------
    data : numpy array
        A N x 2 data array to plot. An alternative acceptable input is to
        define ```x``` and ```y``` values.
    bins : int, optional
        Default value set to 300. Should adjust based on density of the data
        over the entire plane.
    mode : {'mesh', 'scatter', 'scatter_mesh'}, str, optional
        Plotting mode. 'mesh' produces a 2D-histogram whereas 'scatter'
        produces a scatterplot colored by histogram bin value. 'scatter_mesh'
        produces a 2D-histogram with scatter dots of the data on top. 'contour'
        produces a contour map based on the 2D-histogram.
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
        ```logit```, ```loigcle``` or ```biexp```
    yscale : str, optional
        Scale of the y axis, either ``linear``, ``log``, ```symlog```,
        ```logit```, ```loigcle``` or ```biexp```
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
    x : list or array, optional
        values for x axis. Ignored if ```data_plot``` is supplied
    y : list or array, optional
        values for y axis. Ignored if ```data_plot``` is supplied
    kwargs : dict, optional
        Additional parameters passed directly to the underlying mpl
        functions: ``plt.scatter`` if ``mode==scatter``, and
        ``plt.pcolormesh`` if ``mode==mesh``.

    """
    import warnings
    warnings.filterwarnings("ignore")

    if xscale == 'biexp':
        xscale = 'logicle'
    if yscale == 'biexp':
        yscale = 'logicle'

    if not ax:
        existing_plot = False
        fig, ax = plt.subplots(figsize=figsize)
    else:
        existing_plot = True
        if xscale == 'logicle' or yscale == 'logicle':
            new_xlim = [ax.get_xlim()[0], ax.get_xlim()[1]]
            new_ylim = [ax.get_ylim()[0], ax.get_ylim()[1]]
            add_range = np.array([new_xlim, new_ylim]).T

    if data_plot is None:
        try:
            data_plot = np.array([x, y]).T
        except:
            raise ValueError('You must provide values for either "data_plot" or "x" and "y"')

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
        order = np.argsort(z)
        artist = ax.scatter(x, y, edgecolor='none', c=z, cmap=cmap,
                            **kwargs)

    elif mode == 'contour' or mode == 'scatter_mesh' or mode == 'mesh':
        if smooth:
            Z = sH
        else:
            Z = H

        if logz:
            Z = np.log10(Z)
            min_mesh_val = np.max(Z) + np.log10(min_threshold)
        else:
            min_mesh_val = np.max(Z)*min_threshold

        Z[Z <= min_mesh_val] = np.nan

        if mode == 'contour':
            xc = (xe[:-1] + xe[1:]) / 2.0   # x-axis bin centers
            yc = (ye[:-1] + ye[1:]) / 2.0   # y-axis bin centers
            xx, yy = np.meshgrid(xc, yc)
            artist = ax.contour(xx, yy, Z, cmap=cmap, **kwargs)

        elif mode == 'scatter_mesh':
            if mesh_order == 'top':
                zorder = np.inf
            else:
                zorder = 0
            artist = ax.pcolormesh(xe, ye, H if Z is None else Z, alpha=alpha,
                                   zorder=zorder, edgecolors='face', cmap=cmap)
            ax.scatter(data_plot[:, 0], data_plot[:, 1], edgecolor='none',
                       alpha=dot_alpha, **kwargs)

        elif mode == 'mesh':
            Z[Z <= min_mesh_val] = np.nan
            artist = ax.pcolormesh(xe, ye, Z, alpha=alpha,
                          edgecolors='face', cmap=cmap, **kwargs)

        else:
            raise ValueError("mode {} not recognized".format(mode))

    else:
        raise ValueError("mode {} not recognized".format(mode))

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(artist, cax=cax, )
        if normed:
            if logz and mode != 'scatter':
                cbar.ax.set_title(r'Probability (log$_{10}$)', ha='left')
            else:
                cbar.ax.set_title('Probability', ha='left')
        else:
            cbar.ax.set_title('Counts', ha='left')
        if colorbar_fmt == 'math':
            cbar.ax.yaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))
        elif colorbar_fmt == 'scalar':
            cbar.ax.yaxis.set_major_formatter(FormatScalarFormatter("%.2f"))

    if title is not None:
        ax.set_title(title)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if xlabel is not None:
        ax.set_ylabel(ylabel)

    if xscale == 'logicle':
        if existing_plot:
            combined = np.concatenate([data_plot, add_range])
            ax.set_xscale(xscale, data=combined, channel=0)
        else:
            ax.set_xscale(xscale, data=data_plot, channel=0)
    else:
        ax.set_xscale(xscale)
    if yscale == 'logicle':
        if existing_plot:
            combined = np.concatenate([data_plot, add_range])
            ax.set_yscale(yscale, data=combined, channel=1)
        else:
            ax.set_yscale(yscale, data=data_plot, channel=1)
    else:
        ax.set_yscale(yscale, data=data_plot)

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

    if filename:
        fig.savefig(filename, bbox_inches='tight', dpi=dpi,
                    transparent=transparent)
    if existing_plot is False:
        return (fig, ax)

## Taken directly from FlowCal package (19/09/06)
class _LogicleTransform(mpl.transforms.Transform):
    """
    Class implementing the Logicle transform, from scale to data values.

    Relevant parameters can be specified manually, or calculated from
    a given FCSData object.

    Parameters
    ----------
    T : float
        Maximum range of data values. If `data` is None, `T` defaults to
        262144. If `data` is not None, specifying `T` overrides the
        default value that would be calculated from `data`.
    M : float
        (Asymptotic) number of decades in display scale units. If `data` is
        None, `M` defaults to 4.5. If `data` is not None, specifying `M`
        overrides the default value that would be calculated from `data`.
    W : float
        Width of linear range in display scale units. If `data` is None,
        `W` defaults to 0.5. If `data` is not None, specifying `W`
        overrides the default value that would be calculated from `data`.
    data : FCSData or numpy array or list of FCSData or numpy array
        Flow cytometry data from which a set of T, M, and W parameters will
        be generated.
    channel : str or int
        Channel of `data` from which a set of T, M, and W parameters will
        be generated. `channel` should be specified if `data` is not None.

    Methods
    -------
    transform_non_affine(s)
        Apply transformation to a Nx1 numpy array.

    Notes
    -----
    Logicle scaling combines the advantages of logarithmic and linear
    scaling. It is useful when data spans several orders of magnitude
    (when logarithmic scaling would be appropriate) and a significant
    number of datapoints are negative.

    Logicle scaling is implemented using the following equation::

        x = T * 10**(-(M-W)) * (10**(s-W) \
                - (p**2)*10**(-(s-W)/p) + p**2 - 1)

    This equation transforms data ``s`` expressed in "display scale" units
    into ``x`` in "data value" units. Parameters in this equation
    correspond to the class properties. ``p`` and ``W`` are related as
    follows::

        W = 2*p * log10(p) / (p + 1)

    If a FCSData object or list of FCSData objects is specified along with
    a channel, the following default logicle parameters are used: T is
    taken from the largest ``data[i].range(channel)[1]`` or the largest
    element in ``data[i]`` if ``data[i].range()`` is not available, M is
    set to the largest of 4.5 and ``4.5 / np.log10(262144) * np.log10(T)``,
    and W is taken from ``(M - log10(T / abs(r))) / 2``, where ``r`` is the
    minimum negative event. If no negative events are present, W is set to
    zero.

    References
    ----------
    .. [1] D.R. Parks, M. Roederer, W.A. Moore, "A New Logicle Display
    Method Avoids Deceptive Effects of Logarithmic Scaling for Low Signals
    and Compensated Data," Cytometry Part A 69A:541-551, 2006, PMID
    16604519.

    """
    # ``input_dims``, ``output_dims``, and ``is_separable`` are required by
    # mpl.
    input_dims = 1
    output_dims = 1
    is_separable = True
    # Locator objects need this object to store the logarithm base used as an
    # attribute.
    base = 10

    def __init__(self, T=None, M=None, W=None, data=None, channel=None):
        mpl.transforms.Transform.__init__(self)
        # If data is included, try to obtain T, M and W from it
        if data is not None:
            if channel is None:
                raise ValueError("if data is provided, a channel should be"
                    + " specified")
            # Convert to list if necessary
            if not isinstance(data, list):
                data = [data]
            # Obtain T, M, and W if not specified
            # If elements of data have ``.range()``, use it to determine the
            # max data value. Else, use the maximum value in the array.
            if T is None:
                T = 0
                for d in data:
                    # Extract channel
                    y = d[:, channel] if d.ndim > 1 else d
                    if hasattr(y, 'range') and hasattr(y.range, '__call__'):
                        Ti = y.range(0)[1]
                    else:
                        Ti = np.max(y)
                    T = Ti if Ti > T else T
            if M is None:
                M = max(4.5, 4.5 / np.log10(262144) * np.log10(T))
            if W is None:
                W = 0
                for d in data:
                    # Extract channel
                    y = d[:, channel] if d.ndim > 1 else d
                    # If negative events are present, use minimum.
                    if np.any(y < 0):
                        r = np.min(y)
                        Wi = (M - np.log10(T / abs(r))) / 2
                        W = Wi if Wi > W else W
        else:
            # Default parameter values
            if T is None:
                T = 262144
            if M is None:
                M = 4.5
            if W is None:
                W = 0.5
        # Check that property values are valid
        if T <= 0:
            raise ValueError("T should be positive")
        if M <= 0:
            raise ValueError("M should be positive")
        if W < 0:
            raise ValueError("W should not be negative")

        # Store parameters
        self._T = T
        self._M = M
        self._W = W

        # Calculate dependent parameter p
        # It is not possible to analytically obtain ``p`` as a function of W
        # only, so ``p`` is calculated numerically using a root finding
        # algorithm. The initial estimate provided to the algorithm is taken
        # from the asymptotic behavior of the equation as ``p -> inf``. This
        # results in ``W = 2*log10(p)``.
        p0 = 10**(W / 2.)
        # Functions to provide to the root finding algorithm
        def W_f(p):
            return 2*p / (p + 1) * np.log10(p)
        def W_root(p, W_target):
            return W_f(p) - W_target
        # Find solution
        sol = scipy.optimize.root(W_root, x0=p0, args=(W))
        # Solution should be unique
        assert sol.success
        assert len(sol.x) == 1
        # Store solution
        self._p = sol.x[0]

    @property
    def T(self):
        """
        Maximum range of data.

        """
        return self._T

    @property
    def M(self):
        """
        (Asymptotic) number of decades in display scale units.

        """
        return self._M

    @property
    def W(self):
        """
        Width of linear range in display scale units.

        """
        return self._W

    def transform_non_affine(self, s):
        """
        Apply transformation to a Nx1 numpy array.

        Parameters
        ----------
        s : array
            Data to be transformed in display scale units.

        Return
        ------
        array or masked array
            Transformed data, in data value units.

        """
        T = self._T
        M = self._M
        W = self._W
        p = self._p
        # Calculate x
        return T * 10**(-(M-W)) * (10**(s-W) - (p**2)*10**(-(s-W)/p) + p**2 - 1)

    def inverted(self):
        """
        Get an object implementing the inverse transformation.

        Return
        ------
        _InterpolatedInverseTransform
            Object implementing the reverse transformation.

        """
        return _InterpolatedInverseTransform(transform=self,
                                             smin=0,
                                             smax=self._M)

class _LogicleLocator(mpl.ticker.Locator):
    """
    Determine the tick locations for logicle axes.

    Parameters
    ----------
    transform : _LogicleTransform
        transform object
    subs : array, optional
        Subtick values, as multiples of the main ticks. If None, do not use
        subticks.

    """

    def __init__(self, transform, subs=None):
        self._transform = transform
        if subs is None:
            self._subs = [1.0]
        else:
            self._subs = subs
        self.numticks = 15

    def set_params(self, subs=None, numticks=None):
        """
        Set parameters within this locator.

        Parameters
        ----------
        subs : array, optional
            Subtick values, as multiples of the main ticks.
        numticks : array, optional
            Number of ticks.

        """
        if numticks is not None:
            self.numticks = numticks
        if subs is not None:
            self._subs = subs

    def __call__(self):
        """
        Return the locations of the ticks.

        """
        # Note, these are untransformed coordinates
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
        """
        Get a set of tick values properly spaced for logicle axis.

        """
        # Extract base from transform object
        b = self._transform.base
        # The logicle domain is divided into two regions: A "linear" region,
        # which may include negative numbers, and a "logarithmic" region, which
        # only includes positive numbers. These two regions are separated by a
        # value t, given by the logicle equations. An illustration is given
        # below.
        #
        # -t ==0== t ========>
        #     lin       log
        #
        # vmin and vmax can be anywhere in this domain, meaning that both should
        # be greater than -t.
        #
        # The logarithmic region will only have major ticks at integral log
        # positions. The linear region will have a major tick at zero, and one
        # major tick at the largest absolute  integral log value in screen
        # inside this region. Subticks will be added at multiples of the
        # integral log positions.

        # If the linear range is too small, create new transformation object
        # with slightly wider linear range. Otherwise, the number of decades
        # below will be infinite
        if self._transform.W == 0 or \
                self._transform.M / self._transform.W > self.numticks:
            self._transform = _LogicleTransform(
                T=self._transform.T,
                M=self._transform.M,
                W=self._transform.M / self.numticks)
        # Calculate t
        t = - self._transform.transform_non_affine(0)

        # Swap vmin and vmax if necessary
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        # Calculate minimum and maximum limits in scale units
        vmins = self._transform.inverted().transform_non_affine(vmin)
        vmaxs = self._transform.inverted().transform_non_affine(vmax)

        # Check whether linear or log regions are present
        has_linear = has_log = False
        if vmin <= t:
            has_linear = True
            if vmax > t:
                has_log = True
        else:
            has_log = True

        # Calculate number of ticks in linear and log regions
        # The number of ticks is distributed by the fraction that each region
        # occupies in scale units
        if has_linear:
            fraction_linear = (min(vmaxs, 2*self._transform.W) - vmins) / \
                (vmaxs - vmins)
            numticks_linear = np.round(self.numticks*fraction_linear)
        else:
            numticks_linear = 0
        if has_log:
            fraction_log = (vmaxs - max(vmins, 2*self._transform.W)) / \
                (vmaxs - vmins)
            numticks_log = np.round(self.numticks*fraction_log)
        else:
            numticks_log = 0

        # Calculate extended ranges and step size for tick location
        # Extended ranges take into account discretization.
        if has_log:
            # The logarithmic region's range will include from the decade
            # immediately below the lower end of the region to the decade
            # immediately above the upper end.
            # Note that this may extend the logarithmic region to the left.
            log_ext_range = [np.floor(np.log(max(vmin, t)) / np.log(b)),
                             np.ceil(np.log(vmax) / np.log(b))]
            # Since a major tick will be located at the lower end of the
            # extended range, make sure that it is not too close to zero.
            if vmin <= 0:
                zero_s = self._transform.inverted().\
                    transform_non_affine(0)
                min_tick_space = 1./self.numticks
                while True:
                    min_tick_s = self._transform.inverted().\
                        transform_non_affine(b**log_ext_range[0])
                    if (min_tick_s - zero_s)/(vmaxs - vmins) < min_tick_space \
                            and ((log_ext_range[0] + 1) < log_ext_range[1]):
                        log_ext_range[0] += 1
                    else:
                        break
            # Number of decades in the extended region
            log_decades = log_ext_range[1] - log_ext_range[0]
            # The step is at least one decade.
            if numticks_log > 1:
                log_step = max(np.floor(float(log_decades)/(numticks_log-1)), 1)
            else:
                log_step = 1
        else:
            # Linear region only
            linear_range = [vmin, vmax]
            # Initial step size will be one decade below the maximum whole
            # decade in the range
            linear_step = mpl.ticker.decade_down(
                linear_range[1] - linear_range[0], b) / b
            # Reduce the step size according to specified number of ticks
            while (linear_range[1] - linear_range[0])/linear_step > \
                    numticks_linear:
                linear_step *= b
            # Get extended range by discretizing the region limits
            vmin_ext = np.floor(linear_range[0]/linear_step)*linear_step
            vmax_ext = np.ceil(linear_range[1]/linear_step)*linear_step
            linear_range_ext = [vmin_ext, vmax_ext]

        # Calculate major tick positions
        major_ticklocs = []
        if has_log:
            # Logarithmic region present
            # If a linear region is present, add the negative of the lower limit
            # of the extended log region and zero. Then, add ticks for each
            # logarithmic step as calculated above.
            if has_linear:
                major_ticklocs.append(- b**log_ext_range[0])
                major_ticklocs.append(0)
            # Use nextafter to pick the next floating point number, and try to
            # include the upper limit in the generated range.
            major_ticklocs.extend(b ** (np.arange(
                log_ext_range[0],
                np.nextafter(log_ext_range[1], np.inf),
                log_step)))
        else:
            # Only linear region present
            # Draw ticks according to linear step calculated above.
            # Use nextafter to pick the next floating point number, and try to
            # include the upper limit in the generated range.
            major_ticklocs.extend(np.arange(
                linear_range_ext[0],
                np.nextafter(linear_range_ext[1], np.inf),
                linear_step))
        major_ticklocs = np.array(major_ticklocs)

        # Add subticks if requested
        subs = self._subs
        if (subs is not None) and (len(subs) > 1 or subs[0] != 1.0):
            ticklocs = []
            if has_log:
                # Subticks for each major tickloc present
                for major_tickloc in major_ticklocs:
                    ticklocs.extend(subs * major_tickloc)
                # Subticks from one decade below the lowest
                major_ticklocs_pos = major_ticklocs[major_ticklocs > 0]
                if len(major_ticklocs_pos):
                    tickloc_next_low = np.min(major_ticklocs_pos)/b
                    ticklocs.append(tickloc_next_low)
                    ticklocs.extend(subs * tickloc_next_low)
                # Subticks for the negative linear range
                if vmin < 0:
                    ticklocs.extend([(-ti) for ti in ticklocs if ti < -vmin ])
            else:
                ticklocs = list(major_ticklocs)
                # If zero is present, add ticks from a decade below the lowest
                if (vmin < 0) and (vmax > 0):
                    major_ticklocs_nonzero = major_ticklocs[
                        np.nonzero(major_ticklocs)]
                    tickloc_next_low = np.min(np.abs(major_ticklocs_nonzero))/b
                    ticklocs.append(tickloc_next_low)
                    ticklocs.extend(subs * tickloc_next_low)
                    ticklocs.append(-tickloc_next_low)
                    ticklocs.extend(subs * - tickloc_next_low)

        else:
            # Subticks not requested
            ticklocs = major_ticklocs

        return self.raise_if_exceeds(np.array(ticklocs))


    def view_limits(self, vmin, vmax):
        """
        Try to choose the view limits intelligently.

        """
        b = self._transform.base
        if vmax < vmin:
            vmin, vmax = vmax, vmin

        if not mpl.ticker.is_decade(abs(vmin), b):
            if vmin < 0:
                vmin = -mpl.ticker.decade_up(-vmin, b)
            else:
                vmin = mpl.ticker.decade_down(vmin, b)
        if not mpl.ticker.is_decade(abs(vmax), b):
            if vmax < 0:
                vmax = -mpl.ticker.decade_down(-vmax, b)
            else:
                vmax = mpl.ticker.decade_up(vmax, b)

        if vmin == vmax:
            if vmin < 0:
                vmin = -mpl.ticker.decade_up(-vmin, b)
                vmax = -mpl.ticker.decade_down(-vmax, b)
            else:
                vmin = mpl.ticker.decade_down(vmin, b)
                vmax = mpl.ticker.decade_up(vmax, b)
        result = mpl.transforms.nonsingular(vmin, vmax)
        return result

class _LogicleScale(mpl.scale.ScaleBase):
    """
    Class that implements the logicle axis scaling.

    To select this scale, an instruction similar to
    ``gca().set_yscale("logicle")`` should be used. Note that any keyword
    arguments passed to ``set_xscale`` and ``set_yscale`` are passed along
    to the scale's constructor.

    Parameters
    ----------
    T : float
        Maximum range of data values. If `data` is None, `T` defaults to
        262144. If `data` is not None, specifying `T` overrides the
        default value that would be calculated from `data`.
    M : float
        (Asymptotic) number of decades in display scale units. If `data` is
        None, `M` defaults to 4.5. If `data` is not None, specifying `M`
        overrides the default value that would be calculated from `data`.
    W : float
        Width of linear range in display scale units. If `data` is None,
        `W` defaults to 0.5. If `data` is not None, specifying `W`
        overrides the default value that would be calculated from `data`.
    data : FCSData or numpy array or list of FCSData or numpy array
        Flow cytometry data from which a set of T, M, and W parameters will
        be generated.
    channel : str or int
        Channel of `data` from which a set of T, M, and W parameters will
        be generated. `channel` should be specified if `data` is not None.

    """
    # String name of the scaling
    name = 'logicle'

    def __init__(self, axis, **kwargs):
        # Run parent's constructor
        mpl.scale.ScaleBase.__init__(self)
        # Initialize and store logicle transform object
        self._transform = _LogicleTransform(**kwargs)

    def get_transform(self):
        """
        Get a new object to perform the scaling transformation.

        """
        return _InterpolatedInverseTransform(transform=self._transform,
                                             smin=0,
                                             smax=self._transform._M)

    def set_default_locators_and_formatters(self, axis):
        """
        Set up the locators and formatters for the scale.

        Parameters
        ----------
        axis: mpl.axis
            Axis for which to set locators and formatters.

        """
        axis.set_major_locator(_LogicleLocator(self._transform))
        axis.set_minor_locator(_LogicleLocator(self._transform,
                                               subs=np.arange(2.0, 10.)))
        axis.set_major_formatter(mpl.ticker.LogFormatterSciNotation(
            labelOnlyBase=True))

    def limit_range_for_scale(self, vmin, vmax, minpos):
        """
        Return minimum and maximum bounds for the logicle axis.

        Parameters
        ----------
        vmin : float
            Minimum data value.
        vmax : float
            Maximum data value.
        minpos : float
            Minimum positive value in the data. Ignored by this function.

        Return
        ------
        float
            Minimum axis bound.
        float
            Maximum axis bound.

        """
        vmin_bound = self._transform.transform_non_affine(0)
        vmax_bound = self._transform.transform_non_affine(self._transform.M)
        vmin = max(vmin, vmin_bound)
        vmax = min(vmax, vmax_bound)
        return vmin, vmax

class _InterpolatedInverseTransform(mpl.transforms.Transform):
    """
    Class that inverts a given transform class using interpolation.

    Parameters
    ----------
    transform : mpl.transforms.Transform
        Transform class to invert. It should be a monotonic transformation.
    smin : float
        Minimum value to transform.
    smax : float
        Maximum value to transform.
    resolution : int, optional
        Number of points to use to evaulate `transform`. Default is 1000.

    Methods
    -------
    transform_non_affine(x)
        Apply inverse transformation to a Nx1 numpy array.

    Notes
    -----
    Upon construction, this class generates an array of `resolution` points
    between `smin` and `smax`. Next, it evaluates the specified
    transformation on this array, and both the original and transformed
    arrays are stored. When calling ``transform_non_affine(x)``, these two
    arrays are used along with ``np.interp()`` to inverse-transform ``x``.

    Note that `smin` and `smax` are also transformed and stored. When using
    ``transform_non_affine(x)``, any values in ``x`` outside the range
    specified by `smin` and `smax` transformed are masked.

    """
    # ``input_dims``, ``output_dims``, and ``is_separable`` are required by
    # mpl.
    input_dims = 1
    output_dims = 1
    is_separable = True

    def __init__(self, transform, smin, smax, resolution=1000):
        # Call parent's constructor
        mpl.transforms.Transform.__init__(self)
        # Store transform object
        self._transform = transform

        # Generate input array
        self._s_range = np.linspace(smin, smax, resolution)
        # Evaluate provided transformation and store result
        self._x_range = transform.transform_non_affine(self._s_range)
        # Transform bounds and store
        self._xmin = transform.transform_non_affine(smin)
        self._xmax = transform.transform_non_affine(smax)
        if self._xmin > self._xmax:
            self._xmax, self._xmin = self._xmin, self._xmax

    def transform_non_affine(self, x, mask_out_of_range=True):
        """
        Transform a Nx1 numpy array.

        Parameters
        ----------
        x : array
            Data to be transformed.
        mask_out_of_range : bool, optional
            Whether to mask input values out of range.

        Return
        ------
        array or masked array
            Transformed data.

        """
        # Mask out-of-range values
        if mask_out_of_range:
            x_masked = np.ma.masked_where((x < self._xmin) | (x > self._xmax),
                                          x)
        else:
            x_masked = x
        # Calculate s and return
        return np.interp(x_masked, self._x_range, self._s_range)

    def inverted(self):
        """
        Get an object representing an inverse transformation to this class.

        Since this class implements the inverse of a given transformation,
        this function just returns the original transformation.

        Return
        ------
        mpl.transforms.Transform
            Object implementing the reverse transformation.

        """
        return self._transform

# Register custom scales
mpl.scale.register_scale(_LogicleScale)
