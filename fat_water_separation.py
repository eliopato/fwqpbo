import thinqpbo as tq
import numpy as np
from skimage.filters import threshold_otsu

gyro = 42.576


def QPBO(D, Vx, Vy, Vz):
    graph = tq.QPBOFloat()
    nz, ny, nx = D.shape[1:]
    num_nodes = nz * ny *nx
    graph.add_node(num_nodes)

    # Add unary terms:
    for i in range(num_nodes):
        (z, y, x) = np.unravel_index(i, (nz, ny, nx))
        graph.add_unary_term(i, D[0, z, y, x], D[1, z, y, x])
    
    # Add binary terms in x-direction:
    for z in range(nz):
        for y in range(ny):
            for x in range(nx-1):
                i = np.ravel_multi_index((z, y, x), (nz, ny, nx)) # node index
                j = np.ravel_multi_index((z, y, x+1), (nz, ny, nx)) # x neighbor node index
                graph.add_pairwise_term(i, j, Vx[0, z, y, x], Vx[1, z, y, x], Vx[2, z, y, x], Vx[3, z, y, x])
    
    # Add binary terms in y-direction
    for z in range(nz):
        for y in range(ny-1):
            for x in range(nx) :
                i = np.ravel_multi_index((z, y, x), (nz, ny, nx)) # node index
                j = np.ravel_multi_index((z, y+1, x), (nz, ny, nx)) # y neighbor node index
                graph.add_pairwise_term(i, j, Vy[0, z, y, x], Vy[1, z, y, x], Vy[2, z, y, x], Vy[3, z, y, x])

    # Add binary terms in z-direction
    for z in range(nz-1):
        for y in range(ny):
            for x in range(nx):
                i = np.ravel_multi_index((z, y, x), (nz, ny, nx)) # node index
                j = np.ravel_multi_index((z+1, y, x), (nz, ny, nx)) # z neighbor node index
                graph.add_pairwise_term(i, j, Vz[0, z, y, x], Vz[1, z, y, x], Vz[2, z, y, x], Vz[3, z, y, x])

    graph.solve()

    label = np.zeros(num_nodes)
    for i in range(num_nodes):
        label[i] = graph.get_label(i)

    return label.reshape((nz, ny, nx))


# Calculate LS error J as function of R2*
def get_r2_residuals(Y, dB0, C, n_b0, n_r2, D=None):
    J = np.zeros(shape=(n_r2, Y.shape[1], Y.shape[2], Y.shape[3]))
    for b in range(n_b0):
        for r in range(n_r2):
            if not D:  # complex-valued estimates
                y = Y[:, dB0 == b]
            else:  # real-valued estimates
                y = get_real_demodulated(Y[:, dB0 == b], D[r][b])[0]
            J[r, dB0 == b] = np.linalg.norm(np.tensordot(C[r][b], y, axes=(1,0)), axis=0)**2
    return J


def icm(prev, L, max_icm_update, n_icm_iter, J, V, wx, wy, wz):
    current = np.array(prev)
    for k in range(n_icm_iter):  # icm iterate
        print(str(k+1), ', ', end='')
        prev[:] = current[:]
        min_cost = np.full(current.shape, np.inf)

        updates = [0]*(2*max_icm_update+1)  # Update order
        # Even are positive
        updates[2:len(updates):2] = list(range(1, max_icm_update+1))
        # Odd are negative
        updates[1:len(updates):2] = list(range(-1, -max_icm_update-1, -1))
        for update in updates:
            cost = J[(prev.flatten()+update) % L, range(J.shape[1])].reshape(prev.shape)  # Unary cost
            # Binary costs:
            cost[:,:,1:]  += wx * V[abs((prev[:,:,1:]  + update) % L - prev[:,:,:-1])]
            cost[:,:,:-1] += wx * V[abs((prev[:,:,:-1] + update) % L - prev[:,:,1:])]
            cost[:,1:,:]  += wy * V[abs((prev[:,1:,:]  + update) % L - prev[:,:-1,:])]
            cost[:,:-1,:] += wy * V[abs((prev[:,:-1,:] + update) % L - prev[:,1:,:])]
            cost[1:,:,:]  += wz * V[abs((prev[1:,:,:]  + update) % L - prev[:-1,:,:])]
            cost[:-1,:,:] += wz * V[abs((prev[:-1,:,:] + update) % L - prev[1:,:,:])]

            current[cost < min_cost] = (prev[cost < min_cost]+update) % L
            min_cost[cost < min_cost] = cost[cost < min_cost]
    return current


# Find all local minima of discretely evaluated function f(t) with period T
def find_minima(f): 
    return np.where((f < np.roll(f, 1))*(f < np.roll(f, -1)))[0]


# In each voxel, find two smallest local residual minima in a period of omega
def find_two_smallest_minima(J):
    A = np.zeros(J.shape[1:], dtype=int)
    B = np.zeros(J.shape[1:], dtype=int)
    for z in range(J.shape[1]):
        for y in range(J.shape[2]):
            for x in range(J.shape[3]):
                minima = sorted(find_minima(J[:,z,y,x]), key=lambda b: J[b,z,y,x])[:2]
                if len(minima) == 2:
                    A[z,y,x], B[z,y,x] = minima
                elif len(minima) == 1:
                    A[z,y,x] = B[z,y,x] = minima[0]
                else:
                    A[z,y,x] = B[z,y,x] = 0  # Assign dummy minimum
    return A, B


# 2D measure of isotropy defined as
# the square area over the square perimeter (area normalized to 1)
def isotropy_2d(dx, dy): 
    return np.sqrt(dx*dy)/(2*(dx+dy))


# 3D measure of isotropy defined as
# the cube volume over the cube area (volume normalized to 1)
def isotropy_3d(dx, dy, dz): 
    return (dx*dy*dz)**(2/3)/(2*(dx*dy+dx*dz+dy*dz))


def get_higher_level(level: dict):
    high = {'L': level['L']+1}
    # Isotropy promoting downsampling
    max_isotropy = 0
    for sx in [1, 2]:
        for sy in [1, 2]:
            # Loop over all 2^3=8 downscaling combinations
            for sz in [1, 2]:  
                # at least one dimension must change and the size of all
                # dimensions at lower level must permit any downscaling
                if (sx*sy*sz > 1 and level['nx'] >= sx and
                   level['ny'] >= sy and level['nz'] >= sz):
                    if (level['nx'] == 1):
                        iso = isotropy_2d(level['dy']*sy, level['dz']*sz)
                    elif (level['ny'] == 1):
                        iso = isotropy_2d(level['dx']*sx, level['dz']*sz)
                    elif (level['nz'] == 1):
                        iso = isotropy_2d(level['dx']*sx, level['dy']*sy)
                    else:
                        iso = isotropy_3d(
                          level['dx']*sx, level['dy']*sy, level['dz']*sz)
                    if iso > max_isotropy:
                        max_isotropy = iso
                        high['sx'] = sx
                        high['sy'] = sy
                        high['sz'] = sz
    high['dx'] = level['dx']*high['sx']
    high['dy'] = level['dy']*high['sy']
    high['dz'] = level['dz']*high['sz']

    high['nx'] = int(np.ceil(level['nx']/high['sx']))
    high['ny'] = int(np.ceil(level['ny']/high['sy']))
    high['nz'] = int(np.ceil(level['nz']/high['sz']))
    return high


def get_high_level_residual_image(J, high, level: dict):
    Jhigh = np.zeros((J.shape[0], level['nz']+level['nz'] % high['sz'],
                                  level['ny']+level['ny'] % high['sy'],
                                  level['nx']+level['nx'] % high['sx']))
    Jhigh[:, :level['nz'], :level['ny'], :level['nx']] = J
    return Jhigh.reshape((J.shape[0], high['nz'], high['sz'], high['ny'], high['sy'], high['nx'], high['sx'])).mean(axis=(2,4,6))


def get_b0_from_high_level(dB0high, level: dict, high):
    return np.repeat(np.repeat(np.repeat(dB0high, high['sx'], axis=2), high['sy'], axis=1), high['sz'], axis=0)[:level['nz'], :level['ny'], :level['nx']]


def calculate_field_map(n_b0, level: dict, graph_cut_level, multiscale, max_icm_update,
                        n_icm_iter, J, V, mu, offres_penalty=0, offres_center=0):
    A, B = find_two_smallest_minima(J)
    dB0 = np.array(A)

    # Multiscale recursion
    if dB0.size == 1:  # Trivial case at coarsest level with only one voxel
        print('Level (1, 1, 1): Trivial case')
        return dB0

    if multiscale:
        high = get_higher_level(level)
        Jhigh = get_high_level_residual_image(J, high, level)
        # Recursion:
        dB0high = calculate_field_map(n_b0, high, graph_cut_level, multiscale,
                                    max_icm_update, n_icm_iter, Jhigh, V, mu,
                                    offres_penalty, offres_center).reshape(
                                    high['nz'], high['ny'], high['nx'])
        dB0 = get_b0_from_high_level(dB0high, level, high)
        print('Level ({},{},{}): '.format(
            level['nx'], level['ny'], level['nz']))

    # Prepare MRF
    print('Preparing MRF...', end='')
    # Prepare discontinuity costs
    
    # 2nd derivative of residual function
    # NOTE: No division by square(steplength) since square(steplength) not included in V    
    J.shape = (J.shape[0], np.prod(J.shape[1:]))
    vxls = range(J.shape[1])
    ddJ = (J[(A.flatten()+1) % n_b0, vxls]+J[(A.flatten()-1) % n_b0, vxls]-2*J[A.flatten(), vxls]).reshape(A.shape)

    wx = np.minimum(ddJ[:,:,:-1], ddJ[:,:,1:]) * mu / level['dx']
    wy = np.minimum(ddJ[:,:-1,:], ddJ[:,1:,:]) * mu / level['dy']
    wz = np.minimum(ddJ[:-1,:,:], ddJ[1:,:,:]) * mu / level['dz']

    # Prepare data fidelity costs
    OP = (1-np.cos(2*np.pi*(np.arange(n_b0)-offres_center)/n_b0)) / 2 * offres_penalty

    D = np.array([J[A.flatten(), vxls].reshape(A.shape) + OP[A],
                  J[B.flatten(), vxls].reshape(A.shape) + OP[B]])
    
    print('DONE')

    # QPBO
    graphcut = level['L'] >= graph_cut_level
    if graphcut:
        Vx = np.array(wx*[
                      V[abs(A[:,:,:-1]-A[:,:,1:])],
                      V[abs(A[:,:,:-1]-B[:,:,1:])],
                      V[abs(B[:,:,:-1]-A[:,:,1:])],
                      V[abs(B[:,:,:-1]-B[:,:,1:])]])
        Vy = np.array(wy*[
                      V[abs(A[:,:-1,:]-A[:,1:,:])],
                      V[abs(A[:,:-1,:]-B[:,1:,:])],
                      V[abs(B[:,:-1,:]-A[:,1:,:])],
                      V[abs(B[:,:-1,:]-B[:,1:,:])]])
        Vz = np.array(wz*[
                      V[abs(A[:-1,:,:]-A[1:,:,:])],
                      V[abs(A[:-1,:,:]-B[1:,:,:])],
                      V[abs(B[:-1,:,:]-A[1:,:,:])],
                      V[abs(B[:-1,:,:]-B[1:,:,:])]])

        print('Solving MRF using QPBO...', end='')
        label = QPBO(D, Vx, Vy, Vz)
        print('DONE')

        dB0[label == 0] = A[label == 0]
        dB0[label == 1] = B[label == 1]

    # icm
    if n_icm_iter > 0:
        print('Solving MRF using icm...', end='')
        dB0 = icm(dB0, n_b0, max_icm_update, n_icm_iter, J, V, wx, wy, wz)
        print('DONE')
    return dB0


# Calculate initial phase phi according to
# Bydder et al. MRI 29 (2011): 216-221.
def get_phi(Y, D):
    phi = np.zeros((Y.shape[1]))
    for i in range(Y.shape[1]):
        y = Y[:, i]
        phi[i] = .5*np.angle(np.dot(np.dot(y.transpose(), D), y))
    return phi


# Calculate phi, remove it from Y and return separate real and imag parts
def get_real_demodulated(Y, D):
    phi = get_phi(Y, D)
    y = Y/np.exp(1j*phi)
    return np.concatenate((np.real(y), np.imag(y))), phi


# Calculate LS error J as function of B0
def get_b0_residuals(Y, C, n_b0, i_r2_cand, D=None):
    J = np.zeros(shape=(n_b0, Y.shape[1], Y.shape[2], Y.shape[3], len(i_r2_cand)))
    for r in range(len(i_r2_cand)):
        for b in range(n_b0):
            if not D:  # complex-valued estimates
                y = Y
            else:  # real-valued estimates
                y, phi = get_real_demodulated(Y, D[r][b])
            J[b, :, :, :, r] = np.linalg.norm(np.tensordot(C[i_r2_cand[r]][b], y, axes=(1,0)), axis=0)**2
    J = np.min(J, axis=4) # minimum over R2* candidates
    return J


# Construct modulation vectors for each B0 value
def modulation_vectors(n_b0, N):
    B, Bh = [], []
    for b in range(n_b0):
        omega = 2.*np.pi*b/n_b0
        B.append(np.eye(N)+0j*np.eye(N))
        for n in range(N):
            B[b][n, n] = np.exp(complex(0., n*omega))
        Bh.append(B[b].conj())
    return B, Bh


# Construct matrix RA
def model_matrix(data_param, model_param, R2):
    RA = np.zeros(shape=(data_param['nb_echoes'], model_param['M']), dtype=complex)
    for n in range(data_param['nb_echoes']):
        t = data_param['t1'] + n * data_param['dt']
        for m in range(model_param['M']): # Loop over components/species
            for p in range(model_param['P']):  # Loop over all resonances
                # Chemical shift between water and peak m (in ppm)
                omega = 2. * np.pi * gyro * data_param['B0'] * (model_param['CS'][p] - model_param['CS'][0])
                RA[n, m] += model_param['alpha'][m][p]*np.exp(complex(-(t-data_param['t1'])*R2, t*omega))
    return RA


# Get matrix Dtmp defined so that D = Bconj*Dtmp*Bh
# Following Bydder et al. MRI 29 (2011): 216-221.
def get_dtmp(A):
    Ah = A.conj().T
    inv = np.linalg.inv(np.real(np.dot(Ah, A)))
    Dtmp = np.dot(A.conj(), np.dot(inv, Ah))
    return Dtmp


# Separate and concatenate real and imag parts of complex matrix M
def realify(M):
    R = np.real(M)
    I = np.imag(M)
    return np.concatenate((np.concatenate((R, I)), np.concatenate((-I, R))), 1)


# Get mean square signal magnitude within foreground
def get_mean_energy(Y):
    energy = np.linalg.norm(Y, axis=0)**2
    thres = threshold_otsu(energy)
    return np.mean(energy[energy >= thres])


# Perform the actual reconstruction
def reconstruct(data_param, algo_param, model_param, B0map=None, R2map=None):
    determineB0 = algo_param['graph_cut_level'] is not None or algo_param['n_icm_iter'] > 0
    determineR2 = (algo_param['n_r2'] > 1) and (R2map is None)

    Y = data_param['img']

    # Prepare matrices
    # Off-resonance modulation vectors (one for each off-resonance value)
    B, Bh = modulation_vectors(algo_param['n_b0'], data_param['nb_echoes'])
    RA, RAp, C, Qp = [], [], [], []
    D = None
    if algo_param['realEstimates']:
        D = []  # Matrix for calculating phi (needed for real-valued estimates)
    for r in range(algo_param['n_r2']):
        R2 = r*algo_param['R2step']
        RA.append(model_matrix(data_param, model_param, R2))
        if algo_param['realEstimates']:
            D.append([])
            Dtmp = get_dtmp(RA[r])
            for b in range(algo_param['n_b0']):
                D[r].append(np.dot(B[b].conj(), np.dot(Dtmp, Bh[b])))
            RA[r] = np.concatenate((np.real(RA[r]), np.imag(RA[r])))
        RAp.append(np.linalg.pinv(RA[r]))

    if algo_param['realEstimates']:
        for b in range(algo_param['n_b0']):
            B[b] = realify(B[b])
            Bh[b] = realify(Bh[b])
    for r in range(algo_param['n_r2']):
        C.append([])
        Qp.append([])
        # Null space projection matrix
        proj = np.eye(data_param['nb_echoes']*(1+algo_param['realEstimates']))-np.dot(RA[r], RAp[r])
        for b in range(algo_param['n_b0']):
            C[r].append(np.dot(np.dot(B[b], proj), Bh[b]))
            Qp[r].append(np.dot(RAp[r], Bh[b]))

    # For B0 index -> off-resonance in ppm
    B0step = 1.0/algo_param['n_b0']/np.abs(data_param['dt'])/gyro/data_param['B0']
    if determineB0:
        V = []  # Precalculate discontinuity costs
        for b in range(algo_param['n_b0']):
            V.append(min(b**2, (b-algo_param['n_b0'])**2))
        V = np.array(V)

        level = {'L': 0, 'nx': data_param['nx'], 'ny': data_param['ny'], 'nz': data_param['nz'],
                 'sx': 1, 'sy': 1, 'sz': 1,
                 'dx': data_param['dx'], 'dy': data_param['dy'], 'dz': data_param['dz']}
        J = get_b0_residuals(Y, C, algo_param['n_b0'], algo_param['i_r2_cand'], D)
        offres_penalty = algo_param['offres_penalty']
        if algo_param['offres_penalty'] > 0:
            offres_penalty *= get_mean_energy(Y)

        dB0 = calculate_field_map(algo_param['n_b0'], level, algo_param['graph_cut_level'],
                                algo_param['multiscale'], algo_param['max_icm_update'],
                                algo_param['n_icm_iter'], J, V, algo_param['mu'],
                                offres_penalty, int(data_param['offres_center']/B0step))
    elif B0map is None:
        dB0 = np.zeros(Y.shape[1:], dtype=int)
    else:
        dB0 = np.array(B0map/B0step, dtype=int)

    if determineR2:
        J = get_r2_residuals(Y, dB0, C, algo_param['n_b0'], algo_param['n_r2'], D)
        R2 = np.argmin(J, axis=0) # brute force minimization
    elif R2map is None:
        R2 = np.zeros(Y.shape[1:], dtype=int)
    else:
        R2 = np.array(R2map/algo_param['R2step'], dtype=int)

    # Find least squares solution given dB0 and R2
    rho = np.zeros(shape=(model_param['M'], data_param['nz'], data_param['ny'], data_param['nx']), dtype=complex)
    for r in range(algo_param['n_r2']):
        for b in range(algo_param['n_b0']):
            vxls = (dB0 == b)*(R2 == r)
            if not D:  # complex estimates
                y = Y[:, vxls]
            else:  # real-valued estimates
                y, phi = get_real_demodulated(Y[:, vxls], D[r][b])
            rho[:, vxls] = np.dot(Qp[r][b], y)
            if D:
                #  Assert phi is the phase angle of water
                phi[rho[0, vxls] < 0] += np.pi
                rho[:, vxls] *= np.exp(1j*phi)

    if B0map is None:
        B0map = np.zeros(Y.shape[1:])
    if R2map is None:
        R2map = np.empty(Y.shape[1:])

    if determineR2:
        R2map[:] = R2*algo_param['R2step']

    if determineB0:
        B0map[:] = dB0*B0step

    return rho, B0map, R2map
