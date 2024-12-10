import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from tqdm import tqdm
import logging as logger
from sklearn.cluster import KMeans
import matplotlib.patches as mpatches
from scipy.sparse import kron, identity

spins = ["up", "down"]
logger.basicConfig(level=logger.INFO)

# no energy dissipation system, i will be using hermicity checks to partialy validate the model 
def check_hermiticity(matrix: csr_matrix, name: str, tol: float = 1e-10):
    difference = matrix - matrix.getH()
    fro_norm = np.sqrt(np.sum(np.abs(difference.data)**2))
    if fro_norm < tol:
        print(f"{name} is Hermitian (Frobenius norm of difference: {fro_norm:.2e} < {tol})")
    else:
        print(f"{name} is NOT Hermitian (Frobenius norm of difference: {fro_norm:.2e} >= {tol})")

#creation operator for full hs
def create_c_dag(site_index, orbital_index, spin, N_sites, N_orbitals):
    dim = 2 ** (2 * N_sites * N_orbitals)
    data = []
    row = []
    col = []

    for state_index in range(dim):
        binary_state = format(state_index, f'0{2 * N_sites * N_orbitals}b')[::-1]
        position = site_index * N_orbitals * 2 + orbital_index * 2 + (0 if spin == 'up' else 1)

        occupation = int(binary_state[position])
        if occupation == 0:
            #bitlike + - declare if no occupation for full hs 
            new_state_binary = list(binary_state)
            new_state_binary[position] = '1'
            new_state_index = int(''.join(new_state_binary)[::-1], 2)  #for (likely logical) some reason going backwards works, going forward does not.

            # fermionic anticommutation reltion for 2nd quant fermion
            sign = (-1) ** sum(int(bit) for bit in binary_state[:position])
            data.append(sign)
            row.append(new_state_index)
            col.append(state_index)

    return csr_matrix((data, (row, col)), shape=(dim, dim), dtype=complex)

#anihilation operator expanded for full hs
def create_c(site_index, orbital_index, spin, N_sites, N_orbitals):
    dim = 2 ** (2 * N_sites * N_orbitals)
    data = []
    row = []
    col = []

    for state_index in range(dim):
        binary_state = format(state_index, f'0{2 * N_sites * N_orbitals}b')[::-1]
        position = site_index * N_orbitals * 2 + orbital_index * 2 + (0 if spin == 'up' else 1)

        occupation = int(binary_state[position])
        if occupation == 1:
            # adding possibility of no ocupation to + - spin bit-like function
            new_state_binary = list(binary_state)
            new_state_binary[position] = '0'
            new_state_index = int(''.join(new_state_binary)[::-1], 2)  #


            sign = (-1) ** sum(int(bit) for bit in binary_state[:position])
            data.append(sign)
            row.append(new_state_index)
            col.append(state_index)

    return csr_matrix((data, (row, col)), shape=(dim, dim), dtype=complex)


N_sites = 1  # atoms
N_orbitals = 5  # d orbital, i account for 3 t2g and 2 eg 
dim = 2 ** (2 * N_sites * N_orbitals)
N_phonon_max_x = 2 #necessary phonon modes for JT distortion, for now i treat them as uniform wave packets
N_phonon_max_y = 1
N_phonon_max = N_phonon_max_x + N_phonon_max_y 

dim_phonon = (N_phonon_max_x + 1) * (N_phonon_max_y + 1)
dim_total = dim *dim_phonon
# projection operator to project onto high low spin orbitals
def create_projection_operator(orbitals, spins, N_sites, N_orbitals):
    """Creates a projection operator for specified orbitals and spins."""
    P = csr_matrix((dim_total, dim_total), dtype=complex)
    for orbital in orbitals:
        for spin in spins:
            c_dag = create_c_dag(0, orbital, spin, N_sites, N_orbitals)
            c = create_c(0, orbital, spin, N_sites, N_orbitals)
            n = c_dag @ c  
            n= kron(n, I_phonon)
            P += n
    return P

# second quantisation operators for this magnificent hilbert space
def create_second_quantized_operator(L_matrix, N_sites, N_orbitals, spins=['up', 'down']):
    operator = csr_matrix((dim, dim), dtype=complex)

    for m in range(N_orbitals):
        for m_prime in range(N_orbitals):
            L_mmprime = L_matrix[m, m_prime]
            if L_mmprime != 0:
                for spin in spins:
                    c_dag = create_c_dag(0, m, spin, N_sites, N_orbitals)
                    c = create_c(0, m_prime, spin, N_sites, N_orbitals)
                    term = L_mmprime * (c_dag @ c)
                    operator += term
    #enforcing hermicity condition
    operator = (operator + operator.getH()) / 2
    return operator

dim = 2 ** (2 * N_sites * N_orbitals)  # Hilbert space dimension

# initialization of mtrices
H_kin = csr_matrix((dim, dim), dtype=complex) #currently not accounted for, phenomenological alternative for enforcing orbital distortion without expanding hilbert space via phononic degrees of freedom by enforcing jumping from technically"quenched" O4h orbitals
H_int = csr_matrix((dim, dim), dtype=complex) # this is intra orbital and same-orbital hubbard creation-anihialation phenomenological expression. 
H_crystal_field = csr_matrix((dim, dim), dtype=complex) # necessary for t2g eg, phenomenologically distorted via exponential function
H_SOC = csr_matrix((dim, dim), dtype=complex) # soc for magnetism. I am not using hartree-fock, as one can say based on bit-wise functions
H_e_ph = csr_matrix((dim_total, dim_total), dtype=complex) # electron-phonon
H_ph = csr_matrix((dim_total, dim_total), dtype=complex) # phononic

# interactions
U = 4.0      # Intra-orbital Coulomb repulsion
Jh = 1.0      # Hund's coupling
U_prime = U - 2 * Jh  # Interorbital Coulomb repulsion
lambda_soc = 0.00      # Spin-Orbit Coupling strength
Delta_CF = 1.4       # Crystal field splitting strength
t2g_energy = -6
# phononh
omega_phonon_x = 0.9
omega_phonon_y = 0.1# Phonon frequency
g_phonon = 0.5     # Electron-phonon coupling strength

# orbitals numeration
t2g_orbitals = [0,1,2]  #  d_xy, d_yz, d_zx - not in superexchange without JT
eg_orbitals = [3,4]      #  d_x2-y2, d_z2 - quenched L without JT. Both important

# I am not using hartree-fock but i am also not using full Pauli.
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

# SOC necessary untill i will figure out approximtion (necessity of observing how phonon's influence the lattice exclude kramers doublets)
L_x = (1 / np.sqrt(2)) * np.array([
    [0, 0, 0, 0, 0],
    [0, 0, -1j, 0, 1],
    [0, -1j, 0, 1, 0],
    [0, 0, 1, 0, -1j],
    [0, 1, 0, -1j, 0]
], dtype=complex)

L_y = (1 / np.sqrt(2)) * np.array([
    [0, 0, 0, 0, 0],
    [0, 0, -1, 0, -1j],
    [0, 1, 0, -1j, 0],
    [0, 0, 1j, 0, -1],
    [0, 1j, 0, -1, 0]
], dtype=complex)

L_z = np.array([
    [0, 0, 0, 0, 0], #
    [0, 1, 0, 0, 0], #
    [0, 0, -1, 0, 0], 
    [0, 0, 0, 2, 0], #
    [0, 0, 0, 0, -2]
], dtype=complex)

# expanding discrete hilbert space with kron. operators:
def extend_operator_to_total_space(operator_electronic, I_phonon):
    return kron(operator_electronic, I_phonon)
#####################################################################phononic degree of freedom:
def create_bosonic_annihilation_operator(N_max):
    dim = N_max + 1
    data = np.sqrt(np.arange(1, dim))
    row = np.arange(0, dim - 1)
    col = np.arange(1, dim)
    return csr_matrix((data, (row, col)), shape=(dim, dim), dtype=complex)

def create_bosonic_creation_operator(N_max):
    dim = N_max + 1
    data = np.sqrt(np.arange(1, dim))
    row = np.arange(1, dim)
    col = np.arange(0, dim - 1)
    return csr_matrix((data, (row, col)), shape=(dim, dim), dtype=complex)


#phonon operators
a_q = create_bosonic_annihilation_operator(N_phonon_max)
a_q_dag = create_bosonic_creation_operator(N_phonon_max)
a_plus_a_dag = a_q + a_q_dag

 # identity matrices:
I_electronic = identity(dim, format='csr')

#phonons quasi particles treated as fermions
a_q_x = create_bosonic_annihilation_operator(N_phonon_max_x)
a_q_x_dag = create_bosonic_creation_operator(N_phonon_max_x)
a_plus_a_dag_x = a_q_x + a_q_x_dag

a_q_y = create_bosonic_annihilation_operator(N_phonon_max_y)
a_q_y_dag = create_bosonic_creation_operator(N_phonon_max_y)
a_plus_a_dag_y = a_q_y + a_q_y_dag

I_phonon_y = identity(N_phonon_max_y + 1, format='csr')
a_plus_a_dag_x_total = kron(a_plus_a_dag_x, I_phonon_y)
I_phonon_x = identity(N_phonon_max_x + 1, format='csr')
a_plus_a_dag_y_total = kron(I_phonon_x, a_plus_a_dag_y)
I_phonon = kron(I_phonon_x, I_phonon_y) 

for spin in ['up', 'down']:

    c_dag_eg1 = create_c_dag(0, 3, spin, N_sites, N_orbitals)
    c_eg1 = create_c(0, 3, spin, N_sites, N_orbitals)
    c_dag_eg2 = create_c_dag(0, 4, spin, N_sites, N_orbitals)
    c_eg2 = create_c(0, 4, spin, N_sites, N_orbitals)
    c_dag_eg1_c_eg2 = c_dag_eg1 @ c_eg2
    c_dag_eg2_c_eg1 = c_dag_eg2 @ c_eg1 
    # off-diagonal coupling terms x
    term1 = kron(c_dag_eg1_c_eg2, a_plus_a_dag_x_total)
    term2 = kron(c_dag_eg2_c_eg1, a_plus_a_dag_x_total)
    # off-diagonal coupling terms y
    term3 = kron(c_dag_eg1_c_eg2, a_plus_a_dag_y_total)
    term4 = kron(c_dag_eg2_c_eg1, a_plus_a_dag_y_total)
    
    H_e_ph += g_phonon * (term1 + term2 + term3 + term4)
        

I_phonon_y = identity(N_phonon_max_y + 1, format='csr')
a_q_x_dag_a_q_x_total = kron(a_q_x_dag @ a_q_x, I_phonon_y)
      
I_phonon_x = identity(N_phonon_max_x + 1, format='csr')
a_q_y_dag_a_q_y_total = kron(I_phonon_x, a_q_y_dag @ a_q_y)
H_ph =  kron(I_electronic, omega_phonon_x *a_q_x_dag_a_q_x_total + omega_phonon_y *a_q_y_dag_a_q_y_total)

# CF
for i in t2g_orbitals:
    n_i_t2g = (
        create_c_dag(0, i, 'up', N_sites, N_orbitals) @ create_c(0, i, 'up', N_sites, N_orbitals) +
        create_c_dag(0, i, 'down', N_sites, N_orbitals) @ create_c(0, i, 'down', N_sites, N_orbitals)
    )
    H_crystal_field += t2g_energy * n_i_t2g
    check_hermiticity(t2g_energy * n_i_t2g, f"H_crystal_field component for t2g orbital {i}")

    for j in eg_orbitals:
        n_j_eg = (
            create_c_dag(0, j, 'up', N_sites, N_orbitals) @ create_c(0, j, 'up', N_sites, N_orbitals) +
            create_c_dag(0, j, 'down', N_sites, N_orbitals) @ create_c(0, j, 'down', N_sites, N_orbitals)
        )
        H_crystal_field += (t2g_energy + Delta_CF) * n_j_eg
        print(f"Crystal field splitting added for eg orbital {j}.")
        check_hermiticity((t2g_energy + Delta_CF) * n_j_eg, f"H_crystal_field component for eg orbital {j}")

# hunds coupling
all_orbitals = t2g_orbitals + eg_orbitals

#interaction term
for i in all_orbitals:
    n_i_up = create_c_dag(0, i, 'up', N_sites, N_orbitals) @ create_c(0, i, 'up', N_sites, N_orbitals)
    n_i_down = create_c_dag(0, i, 'down', N_sites, N_orbitals) @ create_c(0, i, 'down', N_sites, N_orbitals)
    n_i_updown = n_i_up @ n_i_down
    H_int += U * n_i_updown
    check_hermiticity(U * n_i_updown, f"H_int U term for orbital {i}")

for i in all_orbitals:
    for j in all_orbitals:
        if i < j:
            for spin_i in ['up', 'down']:
                n_i = create_c_dag(0, i, spin_i, N_sites, N_orbitals) @ create_c(0, i, spin_i, N_sites, N_orbitals)
                for spin_j in ['up', 'down']:
                    n_j = create_c_dag(0, j, spin_j, N_sites, N_orbitals) @ create_c(0, j, spin_j, N_sites, N_orbitals)
                    n_i_n_j = n_i @ n_j
                    H_int += U_prime * n_i_n_j
                    check_hermiticity(U_prime * n_i_n_j, f"H_int U' term between orbitals {i} and {j}, spins {spin_i}, {spin_j}")

H_int_total = kron(H_int, I_phonon)

check_hermiticity(H_int_total, "H_int_total")
 
for i in all_orbitals:
    for j in all_orbitals:
        if i < j:  
            #spin operators...
            S_i_plus = create_c_dag(0, i, 'up', N_sites, N_orbitals) @ create_c(0, i, 'down', N_sites, N_orbitals)
            S_i_minus = create_c_dag(0, i, 'down', N_sites, N_orbitals) @ create_c(0, i, 'up', N_sites, N_orbitals)
            S_i_z = 0.5 * (
                create_c_dag(0, i, 'up', N_sites, N_orbitals) @ create_c(0, i, 'up', N_sites, N_orbitals) -
                create_c_dag(0, i, 'down', N_sites, N_orbitals) @ create_c(0, i, 'down', N_sites, N_orbitals)
            )

            S_j_plus = create_c_dag(0, j, 'up', N_sites, N_orbitals) @ create_c(0, j, 'down', N_sites, N_orbitals)
            S_j_minus = create_c_dag(0, j, 'down', N_sites, N_orbitals) @ create_c(0, j, 'up', N_sites, N_orbitals)
            S_j_z = 0.5 * (
                create_c_dag(0, j, 'up', N_sites, N_orbitals) @ create_c(0, j, 'up', N_sites, N_orbitals) -
                create_c_dag(0, j, 'down', N_sites, N_orbitals) @ create_c(0, j, 'down', N_sites, N_orbitals)
            )

            # ... for hunds 
            H_int_component = -Jh * (
                S_i_plus @ S_j_minus + S_i_minus @ S_j_plus + S_i_z @ S_j_z
            )
            H_int += H_int_component
            check_hermiticity(H_int_component, f"H_int Hund's term between orbitals {i} and {j}")

# SOC for Lz, as im thinking if maybe i can use projections or ladders here, not convinced to any yet
H_SOC = create_second_quantized_operator(L_z, N_sites, N_orbitals, spins=['up', 'down'])
H_SOC *= lambda_soc
check_hermiticity(H_SOC, "H_SOC after SOC scaling")


#discrete expansion of matrices with no phonon mode
H_SOC_total = kron(H_SOC, I_phonon)
H_kin_total = kron(H_kin, I_phonon)
H_crystal_field_total = kron(H_crystal_field, I_phonon)

# total hamiltonian
H_total = H_int_total + H_SOC_total  + H_crystal_field_total + H_e_ph + H_ph #+ H_kin_total

def commutator(A, B):
    return A.dot(B) - B.dot(A)

print(f"crystal field-electron potentialstart checking... ")
cf_int_check = commutator(H_crystal_field, H_int)
print(f"crystal field-electron potential check: {cf_int_check}")

# t2g eg projections
P_t2g = create_projection_operator(t2g_orbitals, ['up', 'down'], N_sites, N_orbitals)
P_eg = create_projection_operator(eg_orbitals, ['up', 'down'], N_sites, N_orbitals)
print(f"Projection P_t2g constructed. Shape: {P_t2g.shape}")
check_hermiticity(P_t2g, "P_t2g")
print(f"Projection P_eg constructed. Shape: {P_eg.shape}")
check_hermiticity(P_eg, "P_eg")

def create_total_number_operator(N_sites, N_orbitals):
    P_total = csr_matrix((dim_total, dim_total), dtype=complex)
    for orbital in range(N_orbitals):
        for spin in ['up', 'down']:
            c_dag = create_c_dag(0, orbital, spin, N_sites, N_orbitals)
            c = create_c(0, orbital, spin, N_sites, N_orbitals)
            n = c_dag @ c  # Number operator for this orbital and spin
            n = kron(n, I_phonon)  # Extend to total space
            print(f"Adding number operator for orbital {orbital}, spin {spin}. Shape: {n.shape}")
            P_total += n
    print(f"Total number operator constructed. Shape: {P_total.shape}")
    return P_total


# unfortunately, not always integral (?)
N_e_operator = create_total_number_operator(N_sites, N_orbitals)
print(f"Neoperator size: {N_e_operator.shape}")
check_hermiticity(N_e_operator, "N_e_operator")

commutator_Ne = commutator(H_total, N_e_operator)
norm_commutator = np.linalg.norm(commutator_Ne.data)
print(f"Norm of commutator [H_total, N_e_operator]: {norm_commutator}")

# i am attempting some adiabatic approach, so only ground state. Still computationally terrible, as to see ground state of 5 e i need hamiltonian diagonalisation to reach 5 e. 
num_eigenvalues = min(7000, dim_total - 2)
print(f"Diagonalizing Hamiltonian to find {num_eigenvalues} lowest eigenvalues...")
eigenvalues, eigenvectors = eigsh(H_total, k=num_eigenvalues, which='SA')
print("Diagonalization complete.")

# adiabatic gs vec
ground_state_vector = eigenvectors[:, 0]
print(f"Ground state eigenvalue: {eigenvalues[0]:.4f}")
print(f"Ground state vector size: {ground_state_vector.size}")
############################################################################################

H_Lx = create_second_quantized_operator(L_x, N_sites, N_orbitals)
H_Ly = create_second_quantized_operator(L_y, N_sites, N_orbitals)
H_Lz = create_second_quantized_operator(L_z, N_sites, N_orbitals)
H_Lx_total = kron(H_Lx, I_phonon)
H_Ly_total = kron(H_Ly, I_phonon)
H_Lz_total = kron(H_Lz, I_phonon)
check_hermiticity(H_Lx, "H_Lx")
check_hermiticity(H_Ly, "H_Ly")
check_hermiticity(H_Lz, "H_Lz")
#shape checks
print(f"I_phonon shape: {I_phonon.shape}") 
print(f"I_electronic shape: {I_electronic.shape}")
#angular momentum- lattice relations
def compute_expectation_values(eigenvectors, operator):
    expectation = np.real(np.diag(eigenvectors.conj().T @ (operator @ eigenvectors)))
    return expectation

expect_Lx = compute_expectation_values(eigenvectors, H_Lx_total)
expect_Ly = compute_expectation_values(eigenvectors, H_Ly_total)
expect_Lz = compute_expectation_values(eigenvectors, H_Lz_total)

total_electrons = compute_expectation_values(eigenvectors, N_e_operator)

#seeking for nont integral separately, but integral in total (hybridsation, JT maybe even). 
occ_t2g = compute_expectation_values(eigenvectors, P_t2g)
occ_eg = compute_expectation_values(eigenvectors, P_eg)

try:
    logger.info("Computing occupancies using vectorized operations...")
    occ_t2g = np.real(np.diag(eigenvectors.conj().T @ (P_t2g @ eigenvectors)))
    occ_eg = np.real(np.diag(eigenvectors.conj().T @ (P_eg @ eigenvectors)))
    total = occ_t2g + occ_eg
    print(f"Sum of t2g and eg occupancies: {total}")
    mixing_measure = 2 * occ_t2g * occ_eg
    logger.info("Occupancies computed successfully.")
except MemoryError:
    logger.warning("MemoryError encountered. Switching to loop-based occupancy computation.")
    occ_t2g = []
    occ_eg = []
    mixing_measure = []

    for idx, vec in enumerate(tqdm(eigenvectors.T, desc="Computing Occupancies")):
        vec_col = vec.reshape(-1, 1)
        
        occ_t2g_value = (vec_col.conj().T @ (P_t2g @ vec_col)).real[0, 0]
        occ_eg_value = (vec_col.conj().T @ (P_eg @ vec_col)).real[0, 0]
        
        occ_t2g.append(occ_t2g_value)
        occ_eg.append(occ_eg_value)
        
        mixing = 2 * occ_t2g_value * occ_eg_value
        mixing_measure.append(mixing)
        
        logger.info(f"Eigenstate {idx}: t2g = {occ_t2g_value:.4f}, eg = {occ_eg_value:.4f}, Mixing = {mixing:.4f}")

    occ_t2g = np.array(occ_t2g)
    occ_eg = np.array(occ_eg)
    total = occ_t2g + occ_eg 
    
if 'occ_t2g' not in locals() or 'occ_eg' not in locals():
    raise NameError("idk")
plt.figure(figsize=(10, 6))
plt.scatter(total_electrons, eigenvalues, alpha=0.6, c='blue', edgecolors='k')
plt.xlabel('Total Electron Number', fontsize=14)
plt.ylabel('Energy', fontsize=14)
plt.title('Energy vs. Total Electron Number', fontsize=16)
plt.grid(True)
plt.show()

# manual verification
selected_indices = [0, 2, 65, 75, 85]
for idx in selected_indices:
    if idx < len(eigenvalues):
        vec = eigenvectors[:, idx].reshape(-1, 1)
        occ_t2g_manual = (vec.conj().T @ (P_t2g @ vec)).real[0, 0]
        occ_eg_manual = (vec.conj().T @ (P_eg @ vec)).real[0, 0]
        total_manual = occ_t2g_manual + occ_eg_manual
        logger.info(f"Manual Check - Eigenstate {idx}: t2g = {occ_t2g_manual:.4f}, eg = {occ_eg_manual:.4f}, Total = {total_manual:.4f}")

plt.figure(figsize=(10, 8))
scatter = plt.scatter(occ_t2g, occ_eg, c=eigenvalues, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Energy')
plt.xlabel('$t_{2g}$ Occupancy', fontsize=14)
plt.ylabel('$e_g$ Occupancy', fontsize=14)
plt.title('$t_{2g}$ vs. $e_g$ Occupancy Colored by Energy', fontsize=16)
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(total_electrons, bins=range(int(min(total_electrons)), int(max(total_electrons)) + 2), color='skyblue', edgecolor='black')
plt.xlabel('Total Electron Number', fontsize=14)
plt.ylabel('Number of Eigenstates', fontsize=14)
plt.title('Distribution of Total Electron Numbers Across Eigenstates', fontsize=16)
plt.grid(True)
plt.show()

mean_electrons = np.mean(total_electrons)
std_electrons = np.std(total_electrons)
print(f"Mean Total Electron Number: {mean_electrons:.2f}")
print(f"Standard Deviation of Electron Number: {std_electrons:.2f}")

# not a great approach below
mean_t2g = np.mean(occ_t2g)
std_t2g = np.std(occ_t2g)
mean_eg = np.mean(occ_eg)
std_eg = np.std(occ_eg)
print(f"Mean $t_{{2g}}$ Occupancy: {mean_t2g:.2f} ± {std_t2g:.2f}")
print(f"Mean $e_g$ Occupancy: {mean_eg:.2f} ± {std_eg:.2f}")

check_hermiticity(H_total, "H_total")
check_hermiticity(P_t2g, "P_t2g")
check_hermiticity(P_eg, "P_eg")

filtered_eigenvalues = []
filtered_occ_t2g = []
filtered_occ_eg = []
filtered_total_electrons = []

desired_occupancy = 3.0
occupancy_tol = 0.2 

for eigenvalue, occ_t2g, occ_eg, N in zip(eigenvalues, occ_t2g, occ_eg, total_electrons):
    total_occ = occ_t2g + occ_eg
    if desired_occupancy - occupancy_tol <= total_occ <= desired_occupancy + occupancy_tol:
        filtered_eigenvalues.append(eigenvalue)
        filtered_occ_t2g.append(occ_t2g)
        filtered_occ_eg.append(occ_eg)
        filtered_total_electrons.append(N)

filtered_eigenvalues = np.array(filtered_eigenvalues)
filtered_occ_t2g = np.array(filtered_occ_t2g)
filtered_occ_eg = np.array(filtered_occ_eg)
filtered_total_electrons = np.array(filtered_total_electrons)


plt.figure(figsize=(10, 8))
scatter = plt.scatter(filtered_occ_t2g, filtered_occ_eg, c=filtered_eigenvalues, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Energy')
plt.xlabel('$t_{2g}$ Occupancy (filtered)', fontsize=14)
plt.ylabel('$e_g$ Occupancy(filtered)', fontsize=14)
plt.title('filtered $t_{2g}$ vs. $e_g$ Occupancy Colored by Energy', fontsize=16)
plt.grid(True)
plt.show()

expect_Lz_t2g = compute_expectation_values(eigenvectors, P_t2g @ H_Lz_total)
expect_Lz_eg = compute_expectation_values(eigenvectors, P_eg @ H_Lz_total)

expect_Ly_t2g = compute_expectation_values(eigenvectors, P_t2g @ H_Ly_total)
expect_Ly_eg = compute_expectation_values(eigenvectors, P_eg @ H_Ly_total)

expect_Lx_t2g = compute_expectation_values(eigenvectors, P_t2g @ H_Lx_total)
expect_Lx_eg = compute_expectation_values(eigenvectors, P_eg @ H_Lx_total)

sum_Lz_contributions = expect_Lz_t2g + expect_Lz_eg
difference = np.abs(expect_Lz - sum_Lz_contributions)
max_difference = np.max(difference)
print(f"Maximum difference between total <L_z> and sum of contributions: {max_difference:.2e}")

if max_difference < 1e-1:
    print("Verification passed: <L_z> = <L_z>_{t2g} + <L_z>_{eg} within tolerance.")
else:
    print("Verification failed: Discrepancies found in <L_z> contributions.")

plt.figure(figsize=(10, 6))
plt.scatter(total_electrons, eigenvalues, alpha=0.6, c='blue', edgecolors='k')
plt.xlabel('Total Electron Number', fontsize=14)
plt.ylabel('Energy', fontsize=14)
plt.title('Energy vs. Total Electron Number', fontsize=16)
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 8))
scatter = plt.scatter(expect_Lz_t2g, expect_Lz_eg, c=eigenvalues, cmap='coolwarm', alpha=0.7, edgecolors='k')
plt.colorbar(scatter, label='Energy')
plt.xlabel('Lz t2g', fontsize=14)
plt.ylabel('Lz eg', fontsize=14)
plt.title('Contribution', fontsize=16)
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 8))
scatter = plt.scatter(expect_Ly_t2g, expect_Ly_eg, c=eigenvalues, cmap='coolwarm', alpha=0.7, edgecolors='k')
plt.colorbar(scatter, label='Energy')
plt.xlabel('Ly t2g', fontsize=14)
plt.ylabel('Ly eg', fontsize=14)
plt.title('Contribution', fontsize=16)
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 8))
scatter = plt.scatter(expect_Lx_t2g, expect_Lx_eg, c=eigenvalues, cmap='coolwarm', alpha=0.7, edgecolors='k')
plt.colorbar(scatter, label='Energy')
plt.xlabel('Lx t2g', fontsize=14)
plt.ylabel('Lx eg', fontsize=14)
plt.title('Contribution', fontsize=16)
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 5))
plt.hist(expect_Lz_t2g, bins=30, color='orange', edgecolor='black', alpha=0.7)
plt.xlabel('Lz t2g', fontsize=14)
plt.ylabel('Number of Eigenstates', fontsize=14)
plt.title('Lz t2g', fontsize=16)
plt.grid(True)
plt.show()
plt.figure(figsize=(12, 5))
plt.hist(expect_Lz_eg, bins=30, color='green', edgecolor='black', alpha=0.7)
plt.xlabel('Lz eg', fontsize=14)
plt.ylabel('Number of Eigenstates', fontsize=14)
plt.title('Distribution of  L_z eg', fontsize=16)
plt.grid(True)
plt.show()

mean_electrons = np.mean(total_electrons)
std_electrons = np.std(total_electrons)
print(f"Mean Total Electron Number: {mean_electrons:.2f}")
print(f"Standard Deviation of Electron Number: {std_electrons:.2f}")

mean_Lz = np.mean(expect_Lz)
std_Lz = np.std(expect_Lz)
print(f"Mean <L_z>: {mean_Lz:.2f} ± {std_Lz:.2f}")

mean_Lz_t2g = np.mean(expect_Lz_t2g)
std_Lz_t2g = np.std(expect_Lz_t2g)
print(f"Mean <L_z>_{'{t2g}'}: {mean_Lz_t2g:.2f} ± {std_Lz_t2g:.2f}")

mean_Lz_eg = np.mean(expect_Lz_eg)
std_Lz_eg = np.std(expect_Lz_eg)
print(f"Mean <L_z>_{'{eg}'}: {mean_Lz_eg:.2f} ± {std_Lz_eg:.2f}")

c_dag = create_c_dag(0, 0, 'up', N_sites, N_orbitals)
c = create_c(0, 0, 'up', N_sites, N_orbitals)
identity = c @ c_dag + c_dag @ c
print(identity.toarray())  #identity matrix check 

