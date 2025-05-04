# import relevant modules
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import product, combinations
import qutip
from qutip import sigmax, sigmay, sigmaz, qeye, tensor, qzero, commutator


def L2_norm(array_or_Qobj): 
    
    """
    Compute the L2 norm (largest singular value) for either a single matrix or an array of matrices.

    This function accepts either a QuTiP Qobj or a NumPy array of Qobj matrices and returns
    the maximum singular value (the L2 norm) of each matrix.

    Parameters
    ----------
    array_or_Qobj : qutip.Qobj or np.ndarray
        Either a single QuTiP Qobj representing a matrix,
        or a NumPy array of Qobj matrices.

    Returns
    -------
    float or list of float
        The L2 norm of the matrix, or a list of L2 norms if an array is provided.
    """
    
    # compute L2 norm for an array of operators 
    if isinstance(array_or_Qobj, np.ndarray): 
        
        # rename input matrix_array
        matrix_array = array_or_Qobj
    
        # initialize list in which to store singular values
        max_sv_array = []

        # compute the singular values of each matrix in matrix_array
        for M in matrix_array: 

            # compute singular values of M
            Σ = np.linalg.svd(M.full(), full_matrices=False, compute_uv = False)

            # find maximum sinuglar value, which corresponds to L2 norm of M
            max_sv_array.append(max(Σ))

        return max_sv_array
    
    # compute L2 norm for a single operator
    if isinstance(array_or_Qobj, qutip.Qobj): 
        
        # rename input matrix as M
        M = array_or_Qobj
        
        # compute maximum singular value of matrix 
        max_sv = max(np.linalg.svd(M.full(), full_matrices=False, compute_uv = False))
        
        return max_sv
    
    
    
def spin_chain_data(N, Jx, Jy, Jz, periodic_bc, tolerance):
    
    """
    Construct the nearest-neighbor Heisenberg Hamiltonian for a 1D spin chain 
    of N qubits and compute its spectral properties.

    Parameters
    ----------
    N : int
        Number of qubits (must be even).
    Jx : float
        Coupling strength for σ_x ⊗ σ_x interactions.
    Jy : float
        Coupling strength for σ_y ⊗ σ_y interactions.
    Jz : float
        Coupling strength for σ_z ⊗ σ_z interactions.
    periodic_bc : bool, optional
        Whether to use periodic boundary conditions (default is False).
    tolerance : float, optional
        Numerical threshold for identifying degenerate ground states 
        (default is 1e-10).

    Returns
    -------
    H_N : Qobj
        Full Hamiltonian of the spin chain.
    H_list : list of Qobj
        List of local two-site interaction Hamiltonians.
    eigenstates : list of Qobj
        All eigenstates of H_N.
    eigenvalues : ndarray
        Corresponding eigenvalues of H_N.
    E_0 : float
        Ground state energy.
    ρ_ground_state : Qobj
        Density matrix of the ground state (or mixed state if degenerate).

    Raises
    ------
    ValueError
        If N is not even.

    Notes
    -----
    The Hamiltonian has the form:
        H = ∑_j (Jx σ_x^j σ_x^{j+1} + Jy σ_y^j σ_y^{j+1} + Jz σ_z^j σ_z^{j+1}),
    with optional periodic boundary terms added if `periodic_bc` is True.

    The ground state density matrix is computed as the projector onto the 
    ground state subspace, normalized to trace one. This accounts for 
    degeneracies.
    """

    
    if N % 2 != 0: 
        raise ValueError("Please enter an even number of sites")
    
    # define Pauli matrices and constants
    σ_x = sigmax()
    σ_y = sigmay()
    σ_z = sigmaz()
    π = np.pi

    # Interaction coefficients, which we assume are uniform throughout the lattice
    Jx_list = Jx*np.ones(N)
    Jy_list = Jy*np.ones(N)
    Jz_list = Jz*np.ones(N)

    # Setup operators for individual qubits; 
    # here sx_list[j] = X_j, sy_list[j] = Y_j, and sz_list[j] = Z_j
    # since the Pauli matrix occupies the jth location in the tensor product of N terms
    # of which N-1 terms are the identity
    sx_list, sy_list, sz_list = [], [], []

    for i in range(N):
        op_list = [qeye(2)]*N
        op_list[i] = σ_x
        sx_list.append(tensor(op_list))
        op_list[i] = σ_y
        sy_list.append(tensor(op_list))
        op_list[i] = σ_z
        sz_list.append(tensor(op_list))

    # define variable for total Hamiltonian H_N and the list of all local 
    # Hamiltonians H_list
    H_N = 0 
    H_list = []
    
    # collect 
    for j in range(N - 1):

        # find H_ij, the Hamiltonian between the ith and jth sites 
        H_ij = Jx_list[j] * sx_list[j] * sx_list[j + 1] + \
               Jy_list[j] * sy_list[j] * sy_list[j + 1] + \
               Jz_list[j] * sz_list[j] * sz_list[j + 1]
        
        # add H_ij to H_N and append H_ij to H_list
        H_N += H_ij
        H_list.append(H_ij)

    # execute if periodic boundary conditions are specified
    if periodic_bc: 
        
        # find H_N1, the Hamiltonian between the Nth and first site
        H_N1 = Jx_list[N-1] * sx_list[N - 1] * sx_list[0] + \
               Jy_list[N-1] * sy_list[N - 1] * sy_list[0] + \
               Jz_list[N-1] * sz_list[N - 1] * sz_list[0]

        # add H_N1 to H_N and append H_N1 to H_list
        H_N += H_N1
        H_list.append(H_N1)

    # find eigenavlues and eigenstates of Hamiltonian 
    eigenvalues, eigenstates = H_N.eigenstates()

    # find indices of smallest eigenvalues, which correspond to the energy(ies) 
    # of the ground state (space in the case of degeneracy); 
    E_0 = min(eigenvalues)
    indices = [index for index, value in enumerate(eigenvalues) \
               if np.allclose(value, E_0, tolerance)]

    # find eigenstates corresponding to ground state
    eigenstates_list = eigenstates[indices]

    # create sum of density matrices of ground states in case ground state is degenerate
    ρ_ground_state = 0 
    for j in range(len(eigenstates_list)):
        ρ_ground_state += (eigenstates_list[j])*(eigenstates_list[j]).dag()

    # return normalized ground state
    return H_N, H_list, eigenstates, eigenvalues, E_0, ρ_ground_state


def generate_SWAP_operators(N, Jx, Jy, Jz):
    
    """
    Construct and validate two-site projection operators for a 1D spin chain with even N.

    This function builds the projection operators π⁺ and π⁻ for each adjacent spin pair
    in a periodic 1D lattice of N spin-1/2 particles, with uniform Heisenberg-type
    interaction coefficients Jx, Jy, and Jz. The lattice is split into two alternating
    sublattices (A and B), and the squared projection operators are grouped accordingly.

    The projection operators are defined as:
        π⁺ = (3 + H_kl) / 4,
        π⁻ = (1 - H_kl) / 4,
    where H_kl is the two-site Hamiltonian between neighboring sites k and l.

    The function performs the following steps:
    - Builds Pauli operator lists for each site.
    - Constructs pairwise Hamiltonians H_kl and corresponding projectors π⁺ and π⁻.
    - Validates orthogonality and completeness relations:
          π⁺π⁻ = 0 and π⁺ + π⁻ = I
    - Groups squared projectors by sublattice (even-indexed pairs as A, odd-indexed as B).
    - Checks that grouping is consistent and that squared projectors match π⁺² and π⁻².

    Parameters
    ----------
    N : int
        The total number of spin sites. Must be even.

    Returns
    -------
    π_list : list of tuple (Qobj, Qobj)
        A list of tuples (π⁺, π⁻) for each neighboring site pair.
    π_sq_A_list : list of tuple (Qobj, Qobj)
        Squared projectors for sublattice A (even-indexed links).
    π_sq_B_list : list of tuple (Qobj, Qobj)
        Squared projectors for sublattice B (odd-indexed links).

    Raises
    ------
    ValueError
        If N is not even, or if the projection operators fail the validation checks.
        If Jx, Jy, Jz do not mutually equal 1. 
    """
    
    if N % 2 != 0: 
        raise ValueError("Please enter an even number of sites.")
        
    if not (Jx == Jy == Jz == 1): 
        raise ValueError("Please enter valid coupling constants")
        
    # define zero and identity matrices corresponding to dimensions of VTA
    zeros_N = qzero([2]*N)
    I_N = qeye([2]*N)
    
    # define Pauli matrices and constants
    σ_x = sigmax()
    σ_y = sigmay()
    σ_z = sigmaz()

    # Setup operators for individual qubits; 
    # here σ_x_list[j] = X_j, σ_y_list[j] = Y_j, and σ_z_list[j] = Z_j
    # since the Pauli matrix occupies the jth location in the tensor product of N terms
    # for which (N-1) terms are the identity
    σ_x_list, σ_y_list, σ_z_list = [], [], []

    for i in range(N):
        op_list = [qeye(2)]*N
        op_list[i] = σ_x
        σ_x_list.append(tensor(op_list))
        op_list[i] = σ_y
        σ_y_list.append(tensor(op_list))
        op_list[i] = σ_z
        σ_z_list.append(tensor(op_list))

    # define empty lists for + and - projection operators
    π_list = []
    
    # collect list of all tuples corresponding to π_p and π_m 
    # SWAP operators
    for k in range(N):

        # find H_ij, the Hamiltonian between the ith and jth sites 
        H_kl = σ_x_list[k] * σ_x_list[(k + 1) % N] + \
               σ_y_list[k] * σ_y_list[(k + 1) % N] + \
               σ_z_list[k] * σ_z_list[(k + 1) % N]
        
        # add π_p to π_m to π_p_list and π_m_list, respectively
        π_p = (3 + H_kl)/4
        π_m = (1 - H_kl)/4
        π_list.append((π_p, π_m))
        
    # collect the squares of each projection operator
    π_sq_list = [[πkl[0]**2, πkl[1]**2] for πkl in π_list]
    
    # collect odd and even projection operators
    π_sq_A_list = π_sq_list[::2]
    π_sq_B_list = π_sq_list[1::2]
    
    # check to ensure projectors obey established summation and orthogonality relations
    π_kl_bool_list = []
    for π_kl in π_list: 
        π_kl_bool_list.append(π_kl[0] * π_kl[1] == zeros_N and \
                              π_kl[0] + π_kl[1] == I_N)
    
    # verify that we have correctly computed the squared projection operators
    πkl_sq_bool_list = []
    for πkl, πkl_sq in zip(π_list, π_sq_list):
        πkl_sq_bool_list.append((πkl[0]**2 == πkl_sq[0] and πkl[1]**2 == πkl_sq[1]))
        
    # verify that we have collected even and odd projection operators 
    # in the correct order
    π_sq_A_bool = []
    π_sq_B_bool = []

    # compare lists containing even and odd projection operators to list 
    # contains list of all squared projection operators
    for j, π_sq_A in enumerate(π_sq_A_list): 
        π_sq_A_bool.append(π_sq_list[int(2*j)] == π_sq_A)
    for j, π_sq_B in enumerate(π_sq_B_list): 
        π_sq_B_bool.append(π_sq_list[int(2*j + 1)] == π_sq_B)
    π_sq_AB_bool = π_sq_A_bool + π_sq_B_bool
    π_sq_AB_bool

    if all(π_kl_bool_list + πkl_sq_bool_list + π_sq_AB_bool):
#         display(Latex(r'$ \pi^{+}_{kl} \pi^{-}_{kl} = 0 \text{ and } $'
#                       r'$\pi^{+}_{kl} + \pi^{-}_{kl} = \mathbb{1}$' 
#                      rf'$ \ \forall \ \{{ (k,l) \ | k \in \{{1, \dots, {N} \}}, l = k+1 \}}$'))
#         display(Latex(r'$\text{Correctly computed squared projection operators}$'))
#         display(Latex(r'$\text{Projection operators for A and B sublattices are correctly ordered}$'))
        return π_list, π_sq_A_list, π_sq_B_list
    
    else: 
        display(Latex(r'$ \pi^{+}_{kl} \pi^{-}_{kl} \neq 0 \text{ or } $'
                      r'$\pi^{+}_{kl} + \pi^{-}_{kl} \neq \mathbb{1}$' 
                     rf'$ \ \forall \ \{{ (k,l) \ | k \in \{{1, \dots, {N} \}}, l = k+1 \}}$'))
        raise ValueError(f'SWAP operators do not obey the desired summation and/or' + \
                          ' orthogonality conditions')