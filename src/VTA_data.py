# import relevant modules
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import product, combinations
from qutip import sigmax, sigmay, sigmaz, qeye, tensor, qzero, commutator

# import custom modules
from spin_data import L2_norm, spin_chain_data, generate_SWAP_operators

# Add the LaTeX binary location to the PATH
# os.environ['PATH'] += os.pathsep + '/Library/TeX/texbin'

# Enable LaTeX text rendering in Matplotlib
# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['font.family'] = 'serif'
# matplotlib.rcParams['font.serif'] = ['Computer Modern']

# Useful for running matplotlib on high-dpi displays
# %config InlineBackend.figure_format='retina'

def compute_R_H_Es_array(α_start, α_end, α_steps, Es, N, Jx, Jy, Jz, tolerance): 
    
    '''
    computes VTA for N sites and converts to P basis 
    '''
    
    if (Jx == Jy == Jz) == False: 
        ValueError("Please enter valid entries for Jx, Jy and Jz.")
        
    # construct α_array 
    α_array = np.linspace(α_start, α_end, α_steps)
        
    # collect lists of projection operators
    π_list, π_sq_A_list, π_sq_B_list = generate_SWAP_operators(N, Jx, Jy, Jz)

    # generate all possible combinations of binary tuples 
    # for N-sites (2^N possible tuples)
    binary_tuples = list(product([0, 1], repeat=int(N)))
    
    # create empty list in which to store VTA operators 
    VTA_list = []
    
    # construct list of VTA operators
    for α in α_array: 

        # initialize VTA for N sites
        # we will iteratively add to this 0 matrix to compute sum of MPOs
        VTA_N = qzero([2]*N)

        # calculate VTA by summing set of MPOs for N sites
        for cvec in binary_tuples: 

            # initialize matrix product operators for the A and B sublattices
            MPO_A = qeye([2]*N)
            MPO_B = qeye([2]*N)

            # computes matrix product operator for given binary tuples
            for j, (πkl_sq_A, πkl_sq_B) in enumerate(zip(π_sq_A_list, π_sq_B_list)):
                c_A_indx = cvec[j]
                c_B_indx = cvec[j + int(N/2)]
                MPO_A *= MPO_A * πkl_sq_A[c_A_indx]
                MPO_B *= MPO_B * πkl_sq_B[c_B_indx]

            # construct VTA operator
            VTA_N += (np.exp((-((α**2)/2)*(Es - N + 4*sum(cvec))**2)) \
                      * MPO_B*MPO_A).tidyup(tolerance)
            
        # append VTA_N to VTA_list
        VTA_list.append(VTA_N)
        
        
    # convert VTA_list into an array and return VTA_array
    VTA_array = np.array(VTA_list, dtype = object)
    return VTA_array


def compute_P_H_Es_array(H_N, α_start, α_end, α_steps, Es): 
    
    """
    Compute the exact (ideal) VTA operator in the computational basis.

    The returned operator is given by:
        exp[-(α(H_N - Es))² / 2]

    Parameters
    ----------
    H_N : Qobj
        The system Hamiltonian (assumed Hermitian) in the computational basis.
    Es : float
        The target energy around which the Gaussian filter is centered.
    α : float
        Width parameter for the Gaussian filter; controls localization in energy space.

    Returns
    -------
    Qobj
        The exact VTA operator in the computational basis.
    """
    
    # define α_array 
    α_array = np.linspace(α_start, α_end, α_steps)
    
    # define shifted Hamiltonian in computational basis
    H_N_Es = H_N - Es
    
    # compute exact (ideal) VTA in the computational basis 
    P_H_Es = lambda α: (-((α*H_N_Es)**2)/2).expm()
    
    # calculate array of P_H_Es values
    P_H_Es_array = np.array([P_H_Es(α) for α in α_array], dtype = object)
    
    # return filter
    return P_H_Es_array


def plot_error_bounds(N_qubits, Jx, Jy, Jz, periodic_bc, tolerance, α_start, α_end, α_steps): 
    
    """
    Compute the operator error and theoretical error bound between exact and compiled VTA operators.

    This function compares the exact VTA operator (constructed via a Gaussian of the Hamiltonian)
    and the compiled VTA operator (constructed as a sum of matrix product operators)
    for a 1D spin chain with N_qubits. It returns both the L2 norm of the difference between 
    the exact and compiled operators, and a theoretical error bound derived from the
    sum of commutators between local Hamiltonian terms.

    The theoretical error bound is computed using:
        (α² / 2) * Σ_{γ1 < γ2} || [H_{γ1}, H_{γ2}] ||₂

    Parameters
    ----------
    α_start : float
        Starting value of α, which controls the width of the Gaussian filter.
    α_end : float
        Ending value of α.
    α_steps : int
        Number of α values to sample between α_start and α_end.
    Es : float
        Target energy value for constructing the exact VTA operator.
    N_qubits : int
        Number of qubits (sites) in the spin chain.
    Jx, Jy, Jz : float
        Coupling constants for the spin interactions in the x, y, and z directions.
    periodic_bc : bool
        Whether to use periodic boundary conditions.
    tolerance : float
        Numerical threshold for operator simplification (used in `.tidyup()`).

    Returns
    -------
    R_minus_P_norm_array : np.ndarray
        Array of spectral norms || R_P(α) - P_P(α) ||₂ for each value of α.
    error_bound_array : np.ndarray
        Theoretical upper bounds on the norm differences based on commutator norms.
    """
    
    # compute α_array
    α_array = np.linspace(α_start, α_end, α_steps)

    # collect properties of spin chain
    H_N, H_N_list, _, _, E0, _ = spin_chain_data(N_qubits, Jx, Jy, Jz, periodic_bc, tolerance)
    
    # collect VTA_list for α_array (VTAs we can actually compile)
    R_H_list = compute_R_H_Es_array(α_start, α_end, α_steps, E0, N_qubits, Jx, Jy, Jz, tolerance)
    
    # collect array of exact VTA operators 
    P_H_list = compute_P_H_Es_array(H_N, α_start, α_end, α_steps, E0)

    # compute L2 norm of R_P(α) - P_P(α) for an array of α
    ϵ_array = np.array(L2_norm(R_H_list - P_H_list))

    # define list [0, 1, ..., N-1] and r, the length of each combination
    site_list = list(range(N_qubits))
    r = 2

    γ1_γ2_tuples = list(combinations(site_list, r))
    # print('Local Hamiltonian Indices: \n', γ1_γ2_tuples)

    # initialize list in which to stor || [H_γ1, H_γ2] || for all γ1 < γ2
    H_γ1_γ2_comm_norm_list = []

    for γ_tuple in γ1_γ2_tuples: 

        # compute H_γ1 and H_γ2 where γ1 < γ2
        H_γ1 = H_N_list[γ_tuple[0]]
        H_γ2 = H_N_list[γ_tuple[1]]

        # compute ||[H_γ1, H_γ2]||
        H_γ1_γ2_comm_norm = L2_norm(commutator(H_γ1, H_γ2))

        # append to list
        H_γ1_γ2_comm_norm_list.append(H_γ1_γ2_comm_norm)

    # null norms that should equal 0, but aren't due to machine error
    H_γ1_γ2_comm_norm_list = [0 if x < tolerance else x for x in H_γ1_γ2_comm_norm_list]

    # find sum of norm of [H_γ1, H_γ2] for all γ1 < γ2
    H_γ1_γ2_comm_norm_sum = sum(H_γ1_γ2_comm_norm_list)

    # compute error bound
    B_array = np.array([(α**2)/2 * H_γ1_γ2_comm_norm_sum for α in α_array])
    
    # Create a figure and axes object
    fig, ax = plt.subplots(dpi = 600)
    
    # plot || R(α) - P(α)||_2
    ax.plot(α_array, ϵ_array, \
            label= fr'$\epsilon_{N_qubits}(\alpha) $', linestyle = 'solid', color='blue')

    # Plot analytically-derived error bounds
    ax.plot(α_array, B_array, \
            label= fr' $ \mathcal{{B}}_{N_qubits}(\alpha) $', linestyle='--', color= 'red')
    
    # add axes, labels, gridlines, tickmarks, and legend to the plot
    ax.set_xlabel(r'$\alpha$', fontsize = 20)
    ax.set_ylabel(r'Error Bounds', fontsize = 20)
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.tick_params(axis='both', length=5, width=1, labelsize = 15)
    ax.legend(fontsize = 15)
    plt.show()
    
    return α_array, ϵ_array, B_array