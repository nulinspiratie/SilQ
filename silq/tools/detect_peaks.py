from typing import List, Tuple

import numpy as np
from scipy import signal
from scipy.signal import convolve2d
from math import pi
from matplotlib import pyplot as plt
from qcodes import load_data, MatPlot

"""
TODO plan:
*- Do any TODO that Serwan or myself adds.
*- Add plot capabilities and any minor modifications to functions
*  - 'Off'     = No plots
*  - 'Simple'  = Plot of DC data and transition data next to it
*  - 'Complex' = All of simple, plus the transition_gradient and theta plots for each point.
- Make a function to test sweeps and display it nicely
  Ideal function:
  - Input: X, Y, Z, data
  - Output: slope of transition(s) while varying gate (Z)
- test this on new test data
- If time allows, use capacitance matrix to find location of the donor.

* = tentatively done
"""

def max_index(M: np.ndarray) -> Tuple[int, int]:
    """Returns the index of the maximum element in M.

    Args:
        M: n-dimensional matrix. Ideally a 2-dimensional theta matrix,
            2-dimensional charge stability diagram, or 2-dimensional transition
            gradient matrix.

    Returns:
        Array with x and y index of maximum element.
        For a transition gradient matrix, I[0] is dx, and I[1] is x1.
    """
    return np.unravel_index(np.argmax(M), M.shape)


def calculate_theta_matrix(Z: np.ndarray, filter: bool = False) -> np.ndarray:
    """Computes the theta matrix for a 2-dimensional charge stability diagram.

    The theta matrix indicates the direction of the 2-dimensional gradient.

    # TODO Please elaborate code. At this moment all these arrays don't make
    #     sense. Either say why the values are chosen, or refer to a
    #     website/paper where they were obtained from.
    #     -> The wikipedia page for sobel operator explains most of this funtion https://en.wikipedia.org/wiki/Sobel_operator
    # TODO Could modifying these values improve the code?
    #     -> Yes, fine-tuning these values could definitely improve performance. This can be done later.
    # TODO From what I gather, we're applying a kernel, so perhaps it makes sense
    #     to have kernel_size as a keyword argument.
    #     -> This is a good point, i want to add that later.
    # TODO what does filter do excactly? What type of filtering is applied?
    #     -> SY and SX are differentiating kernels, all others are binomial window filters.
    Args:
        Z: 2-dimensional charge stability diagram matrix.
        filter: Enables filtering during the calculations.

    Returns:
        theta: 2-dimensional theta matrix.
    """

    ### Filter coefficients

    # Refer to https://en.wikipedia.org/wiki/Sobel_operator
    # That explains the prinicples used here.

    # Sobel Operator
    # SY and SX are differentiating kernels, while the ySfil and xSfil are averaging.
    SY = np.array([[1], [0], [-1]])
    ySfil = np.array([[1, 2, 1]])
    SY = convolve2d(SY, ySfil)

    SX = np.array([[1, 0, -1]])
    xSfil = np.array([[1], [2], [1]])
    SX = convolve2d(SX, xSfil)

    # Binomial filter kernel for the X and Y gradient matrices.
    xGfil = np.array([[1], [2], [1]])
    yGfil = np.array([[1, 2, 1]])
    Gfil = convolve2d(xGfil, yGfil)

    # Binomial filter kernel for the source matrix Z prior to computing the gradients.
    xZfil = np.array([[1], [2], [1]])
    yZfil = np.array([[1, 2, 1]])
    Zfil = convolve2d(xZfil, yZfil)

    if filter:
        # This will filter the source matrix Z prior to computing the gradients.
        Z = convolve2d(Z, Zfil, mode='valid')

    #Calculate X and Y gradient matrices
    GY = convolve2d(Z, SY, mode='valid')
    GX = convolve2d(Z, SX, mode='valid')

    if filter:
        #This will filter the gradient matrices once they have been calculated.
        GY = convolve2d(GY, Gfil, mode='valid')
        GX = convolve2d(GX, Gfil, mode='valid')

    #Calculate gradient direction.
    theta = np.arctan(GY / GX)
    return theta


def find_matrix_mode(M: np.ndarray, bins: int = 100) -> float:
    """Determines the mode of a matrix (most-often occurring element).

    # TODO check if this description is accurate
        -> Yes it is.
    Mode is found by first generating a histogram of matrix values, and then
    returning the center value of the bin with the highest count.
    Values are grouped because floating numbers are only approximately equal.

    Args:
        M: n-dimensional matrix. Ideally a 2-dimensional theta matrix.

    Returns:
        mode: most common element of M after grouping via a histogram.
    """

    hist, hist_edges = np.histogram(M, np.linspace(-pi, pi, bins))
    ind = max_index(hist)
    mode = (hist_edges[ind] + hist_edges[ind + np.array([1])]) / 2
    return mode[0]


def calculate_transition_gradient(theta: np.ndarray, filter: bool = True) -> np.ndarray:
    """Compute the transition gradient matrix from a given theta matrix.

    # TODO minor explanation of what a transition gradient is

    Args:
        theta: 2-dimensional theta matrix of a charge stability diagram.

    Returns:
        2-dimensional transition gradient matrix.
            x-axis is start position, y-axis is gradient.
    """
    # Low priority: duplicate this function to recalculate transition_gradent with reduced range.

    # Generate Lines
    ly, lx = theta.shape
    yl = np.arange(ly, dtype=int)

    # TODO where does this value come from?
    # -> This value was found to be roughly twice the maximum dx value a transition will have
    dx_max = int(np.ceil(ly / 3))

    transition_gradient = np.zeros((dx_max, lx))

    # TODO (Serwan) there's probably a loopless way to implement this
    # -> I don't suspect there is.
    for x1 in range(lx):
        for dx in range(min([x1 + 1, dx_max])):
            xl = x1 + np.round(-dx * yl / ly).astype(int)
            transition_gradient[dx, x1] = np.mean(theta[yl, xl])

    if filter:
        #This filters transition_gradient but can also lose some information.
        filt = np.ones((3,3))
        transition_gradient = convolve2d(transition_gradient, filt, mode='same')/9

    return transition_gradient


def delete_transition(theta: np.ndarray, location: int, gradient: float) -> np.ndarray:
    """Removes a transition from a theta matrix. In order to find transitions, they are identified one at a time. 
    The most prominent transition is identified first, then removed from the theta matrix so that the second most 
    prominent transition can be found.
    
    Transitions can be identitified by their theta matrix being significantly different to the most common theta value. 
    Thus, by replacing a transition with the most common value, it is essentially being removed.
    
    Args:
        theta: 2-dimensional theta matrix of a charge stability diagram.
        location: Base index of the charge transfer event in Z
        gradient: Gradient of the charge transfer event in Z

    Returns:
        theta: modified 2-dimensional theta matrix, with the specified transition removed.
    """

    ly,lx = theta.shape

    yl = np.arange(ly, dtype=int)

    # Start and stop are the base locations from which to delete a transition from.
    # TODO improve start, stop, why is +-3 chosen?
    # -> Because of how theta is filtered, the transition will roughly be visible within a +-3 range.
    #    This could definitely be fine tuned later to be variable length depending on how much filtering there is. 
    #    I'd say this low priority for the moment but should be done in future.
    start = location - 3
    stop = location + 3
    dx = gradient

    if start < 0:
        start = 0
    if stop > lx:  # this needs some fix
        stop = lx
    if start - dx < 0:
        dx = start

    # TODO (Serwan) there's probably a faster loop-less way to do this
    for x1 in range(start, stop):
        xl = x1 + np.round(-dx * yl / ly).astype(int)
        theta[yl, xl] = 0

    return theta


def calculate_theta_deviation(theta: np.ndarray, theta_mode: float) -> np.ndarray:
    #i will add documentation soon
    #you can change this method for potential improvements
    theta_deviation = 1 - np.abs(np.round(np.cos(theta_mode - theta) ** 2))
    return theta_deviation


# Serwan i re-did the plot_transitions to make it easier for me to generate plots.
# I imagine this function could easily be defined in a notebook and used still seperate from this package if you need it.
# def plot_transitions(transitions, ax=None, **plot_kwargs):
#     if ax is None:
#         fig, ax = plt.subplots()

#     plot_kwargs.setdefault('linestyle', '-')

#     for transition in transitions:
#         yvals = ax.get_ylim()
#         xvals = [transition['location'], transition['location']]
#         xvals[1] += (yvals[1] - yvals[0]) / transition['gradient']
#         ax.plot(xvals, yvals, **plot_kwargs)

def plot_transitions(x: np.ndarray, y: np.ndarray, Z: np.ndarray, transitions: List[dict]):
    #will add documentation later
    #Plotting code
    fig0,(ax0,ax1) = plt.subplots(1, 2, figsize=[12,4])
    fig0.suptitle('Transition Identification', fontsize=14, fontweight='semibold')

    ax0.pcolormesh(x, y, Z, cmap='hot')
    ax0.set_xlabel('DBL & DBR Voltage (V)')
    ax0.set_ylabel('TGAC Voltage (V)')
    ax0.set_title('Source scan')

    ax1.pcolormesh(x, y, Z, cmap='hot')
    ax1.set_xlabel('DBL & DBR Voltage (V)')
    ax1.set_title('Transitions Identified')

    yvals = ax1.get_ylim()
    for transition in transitions:
        x_base = transition['location']
        if (type(x_base) is int) : x_base = x[x_base]

        xvals = [x_base, x_base]
        xvals[1] += (yvals[1] - yvals[0]) / transition['gradient']
        ax1.plot(xvals, yvals, '-', linewidth=4)
            # TODO add plot of transition in DC scan
    plt.show()


def plot_transition_gradient(transition_gradient: np.ndarray, theta_deviation: np.ndarray):
    #will add documentation later
    fig, axes = plt.subplots(1, 2, figsize=[13,4])

    c = axes[0].pcolormesh(transition_gradient, cmap='inferno')
    axes[0].set_ylabel('∆x value')
    axes[0].set_xlabel('DBL & DBR voltage index')
    axes[0].set_title('Transition Gradient Matrix')
    fig.colorbar(c, ax=axes[0])

    axes[1].pcolormesh(theta_deviation, cmap='gray')
    axes[1].set_xlabel('DBL & DBR voltage index')
    axes[1].set_ylabel('TGAC voltage index')
    axes[1].set_title('Theta Matrix')

    plt.show()


def find_transitions(Z: np.ndarray,
                     x: np.ndarray,
                     y: np.ndarray,
                     #Serwan, i removed min_gradient because it is not really a minimum gradient and i have updated conditions.
                     #Perhaps this could be changed later but for now i think it should be kept set.
                     true_units: bool = False,
                     charge_transfer: bool = False,
                     plot: str = 'Off') -> List[dict]:
    """Locate transitions within a 2-dimensional charge stability diagram

    Args:
        Z: 2-dimensional charge stability diagram matrix.
        x: 1-dimensional voltage vector for the x-axis of Z
        y: 1-dimensional voltage vector for the y-axis of Z
        min_gradient: Minimum gradient to count as a transition
        true_units:
            if True:
                Where applicable, return all values in proper units. i.e. voltage and current.
            if False:
                Return values in calculation specific form. i.e. index and ratios.
        charge_transfer:
            Enables calculation of voltage and current shift information about transitions.
            This is required to calculate dV, dI, dI_x, dI_y
        plot:
             - 'Off'     = No plots
             - 'Simple'  = Plot of DC data and transition data next to it
             - 'Complex' = All of simple, plus the transition_gradient and theta plots for each transition.

    Returns: a list of dictionaries, one entry for each transition found:
    # TODO (Serwan) simplify this part
    if true_units == True:
        location  (float): Voltage at the base of the transition.
        gradient  (float): Gradient of the transition. in y_Voltage/x_Voltage
        intensity (float): Value between 0 and 1, indicating the strength of the transition
        dV        (float): The shift of coulomb peaks from a charge transfer event. dV = dVtop = ∆q/Ctop
        dI        (array): An array of current change from before to after a transition.
                           Returns -1 if error.
        dI_x      (array): An array of x-voltages corresponding to the points in dI.
        dI_y      (array): An array of y-voltages corresponding to the points in dI.
    if true_units == False):
        location    (int): Index at the base of the transition.
        gradient  (float): Gradient of the transition. in y-index/x-index
        intensity (float): Value between 0 and 1, indicating the strength of the transition
        dV          (int): The shift of coulomb peaks from a charge transfer event in terms of index in X.
                           dV*(y[1]-y[0]) = dVtop = ∆q/Ctop
        dI        (array): An array of current change from before to after a transition.
                           Returns -1 if error.
        dI_x      (array): An array of x-indices corresponding to the points in dI.
        dI_y      (array): An array of y-indices corresponding to the points in dI.
    """

    theta = calculate_theta_matrix(Z, filter=True)
    theta_mode = find_matrix_mode(theta)
    theta_deviation = calculate_theta_deviation(theta,theta_mode)

    transition_gradient = calculate_transition_gradient(theta_deviation, filter=True)

    if (plot == 'Complex'): plot_transition_gradient(transition_gradient,theta_deviation)

    transitions = []

    # This condition seems good now, could be improved later but i'd say low priority.
    while ((np.max(transition_gradient) > 3*np.mean(transition_gradient)) & (np.max(transition_gradient) >0.3)):
        
        #maximum element of transition_gradient will reveal where the transition is
        raw_gradient, raw_location = max_index(transition_gradient) 
        intensity = np.max(transition_gradient) 

        # When filtering with convolution, the size of theta and transition_gradient will differ from the initial Z matrix.
        # The following lines adjust the raw_location from transition_gradient to be a true location in Z
        difx = (x.shape[0] - theta.shape[1]) / 2 #difference in x-axis size
        dify = (y.shape[0] - theta.shape[0]) / 2 #difference in y-axis size
        #Adjusting the location: 
        # raw_location 
        # + difference in x
        # + difference in x from dify due to gradient shift.
        location = int(difx + raw_location + np.round(dify * raw_gradient / theta.shape[0]))
        
        #Recalculate theta with the identified transition removed
        theta_deviation = delete_transition(theta_deviation, raw_location, raw_gradient)
        #Recalculate transition_gradient with an updated theta
        transition_gradient = calculate_transition_gradient(theta_deviation, filter=True)
        
        #If the gradient registers as being close to perfectly vertical, skip over this transition, 
        #since transitions are never perfectly vertical. 
        #You can change this if you don't believe me but the algorithm will be more buggy
        if(raw_gradient <2): continue
        
        #gradient = dy/dx = y_length/dx = theta.shape[0]/raw_gradient
        gradient = -(theta.shape[0]/raw_gradient)
        
        # The following lines combine together to come up with the final error calculation
        #minimum_gradient = dy/max_dx = -(theta.shape[0]/(raw_gradient+1))
        #maximum_gradient = dy/min_dx = -(theta.shape[0]/(raw_gradient-1))
        #abs_error        = (maximum_gradient - minimum_gradient)/2
        #percent_error    = abs_error/observed_gradient*100                = abs_error/(-(theta.shape[0]/raw_gradient))*100
        gradient_error    = (np.abs(raw_gradient/(raw_gradient-1)-1) + np.abs(raw_gradient/(raw_gradient+1)-1))*50
        

        if true_units:  # Convert indices to units
            gradient = gradient * (y[1] - y[0]) / (x[1] - x[0])  # in V/V
            location = x[location]  # units in V

        if charge_transfer:  # TODO What is the downside to applying this by default?
            # This is not set to be enabled by default for three main reasons:
            # 1. It can clutter the output and make it less clear to understand/read
            # 2. It is called multiple times and takes up processing time (approx 10ms)
            # 3. I haven't found anything useful for it yet, so it's not needed all the time
            
            # dV = dVtop = delta_q/Ctop
            dV, dI, dI_x, dI_y = get_charge_transfer_information(
                Z, location, gradient, theta_mode)

            if true_units: # Convert indices to units
                dV = dV * (y[1] - y[0]) # units in V
                dI_y = y[dI_y] # units in V
                dI_x = x[dI_x] # units in V
            transition = {'location': location,
                          'gradient': gradient,
                          'gradient_error': gradient_error,
                          'intensity': intensity,
                          'dVtop': dV,
                          'dI_y': dI_y,
                          'dI_x': dI_x,
                          'dI': dI}
        else:
            transition = {'location': location,
                          'gradient': gradient,
                          'gradient_error': gradient_error,
                          'intensity': intensity}

        #Add transition entry onto the output list
        transitions.append(transition)

        if (plot == 'Complex'): plot_transition_gradient(transition_gradient,theta_deviation)


    if (plot == 'Simple')|(plot == 'Complex'): plot_transitions(x,y,Z,transitions)

    return transitions


def get_charge_transfer_information(Z: np.ndarray,
                                    location: int,
                                    gradient: float,
                                    theta_mode: float) -> Tuple[int, np.ndarray,
                                                   np.ndarray, np.ndarray]:
    """Calculate information about a particular charge transfer event.

    Firstly, calculates how much the coulomb peaks shift at a charge transfer event.
    It does this by taking two slices either side of the transition, then comparing the shift.
    
    Secondly, the algorithm locates points where the current difference could be conducive as a tuning point.
    It does this by again taking two slices either side of the transition and comparing the current difference.

    Args:
        Z: 2-dimensional charge stability diagram matrix.
        location: Base index of the charge transfer event in Z
        gradient: Gradient of the charge transfer event in Z
        theta_mode: Mode of theta (most common theta value)

    Returns:
        dV: The shift of coulomb peaks from a charge transfer event.
                      Given as a shift of index in Z. When properly scaled:
                          dV = dVtop = delta_q/Ctop
        dI: An array of current change from before to after a transition.
        dI_x: An array of x-indices corresponding to the points in dI.
        dI_y: An array of y-indices corresponding to the points in dI.
    """
    #9.95 ms ± 369 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

    ly = Z.shape[0]
    yl = np.arange(ly, dtype=int)
    xl = (location + np.round(yl / gradient)).astype(int)


    # Take two current lines to the left and right of the transition, these will be compared to see what changes.
    # 3 can be chosen arbitrarily, it doesn't matter too much, 
    # as long as each line is definitely on each side of the transition
    shift = 3
    pre  = xl - shift
    post = xl + shift
    if ((min(start) < 0)|(max(stop)>Z.shape[1])):
        #if the pre-post lines are out of bounds, then don't bother computing. This could be improved later.
        return -1, -1, -1, -1
    line_pre = Z[yl, pre]
    line_pos = Z[yl, post]
    
    # Average Magnitude Difference Function. 
    # This will shift and compare the lines before and after to see how much the transition shifted the coulomb peaks
    AMDF = np.zeros(ly)
    for i in range(ly):
        
        AMDF[i] = -np.mean(np.abs(
            line_pre[np.array(range(0, ly - i))]
            -line_pos[np.array(range(i, ly))]))  \
            * (ly + i) / ly                      #Adjustment for the decreasing comparison window as lines are shifted

    # qc.MatPlot(AMDF, figsize=(14,5))
    # peakshift exists to find out how much of the shift in coulomb peaks in the difference is due to the 
    # natural gradient of the coulomb peaks. This can be worked out using tan, the coulomb peak gradient (theta_mode), 
    # and the shift amount
    peakshift = np.round(
        np.abs(np.tan(theta_mode - np.pi / 2)) * (1 + 2 * shift)).astype(
        int)
    dV = max_index(AMDF)[0] + peakshift

    #*** This following section of code could be improved.
    #*** It is a rudimentary implimentation of finding potential tuning points. 
    
    #Now we will take closer lines to compare, in order to find the biggest difference in SET current.
    shift = 1
    line_pre = Z[yl, xl - shift]
    line_pos = Z[yl, xl + shift]

    #Compare the lines and find peaks in the difference 
    #the find_peaks parameters could really be improved.
    #also, using a %difference could be much better than absolute. Use (line_pre-line_pos)/line_pos when re-evaluating.
    peaks = (signal.find_peaks(line_pre - line_pos, distance=25, height=0.2))
    dI_y = peaks[0] #y-index of the peaks location
    dI_x = (location + np.round(dI_y / gradient)).astype(int) #x-index of the peaks
    dI = peaks[1]['peak_heights'] #the value of the peaks

    return dV, dI, dI_x, dI_y
