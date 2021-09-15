def pdr_edd(job_set: tf.Tensor) -> tf.Tensor:
    """ EDD Heuristic.
    :return: perm (batch x time x job)
    """
    job_idx = np.argsort(job_set[..., 2], axis=-1)
    perm = tf.one_hot(tf.cast(job_idx, tf.int32), tf.shape(job_set)[-2])
    job_idx = np.tile(job_idx[..., None], (1, 1, tf.shape(job_set)[-1]))
    job_vec = np.take_along_axis(job_set, job_idx, axis=-2)
    return tf.cast(job_vec, tf.float32), tf.cast(perm, tf.float32)
    
    
def pdr_spt(job_set: tf.Tensor) -> tf.Tensor:
    """ SPT Heuristic.
    """
    job_idx = np.argsort(job_set[..., 0], axis=-1)
    perm = tf.one_hot(tf.cast(job_idx, tf.int32), tf.shape(job_set)[-2])
    job_idx = np.tile(job_idx[..., None], (1, 1, tf.shape(job_set)[-1]))
    job_vec = np.take_along_axis(job_set, job_idx, axis=-2)
    return tf.cast(job_vec, tf.float32), tf.cast(perm, tf.float32)
     

def pdr_family(job_set: tf.Tensor) -> tf.Tensor:
    """ Family clustering.
    """
    # interprets setup one-hot vector as binary code and converts it to integer
    # to achieve this the dot product between the one-hot vector and a 2^n vector is computed
    # this gives us integers of (batch x jobs)
    setup_bins = job_set[..., 3:].dot(2**np.arange(job_set.shape[-1] - 3)[::-1])
    # now we sort according to the integer values and use the permutation matrix to permute
    # the real input vector accordingly
    job_idx = np.argsort(setup_bins, axis=-1)
    perm = tf.one_hot(tf.cast(job_idx, tf.int32), tf.shape(job_set)[-2])
    job_idx = np.tile(job_idx[..., None], (1, 1, tf.shape(job_set)[-1]))
    job_vec = np.take_along_axis(job_set, job_idx, axis=-2)
    # compute the transition points (points in the matrix, where the setup changes)
    # this is useful for subsequent intra-cluster sorting
    # we compute this using the changes in setup binaries, thus first need to permute
    setup_bins_perm = tf.matmul(tf.cast(perm, tf.int32), 
                                tf.cast(setup_bins, tf.int32)[..., None])
    setup_bins_perm = tf.squeeze(setup_bins_perm, axis=-1)
    transitions = []
    # loop over batch elements
    for j in setup_bins_perm:
        # shift sequence one to the right and compare with original to find where changes occure
        t = np.where(np.roll(j,1)!=j)[0]
        # hack to ensure consistency in the output (first element is always new, thus zero at front)
        if len(t) == 0:
            t = np.array(0)
        else:
            t[0] = 0
        transitions.append(t)
    # ensure shape consistency (batch x #transitions)
    transitions = np.array(transitions)
    transitions = transitions if transitions.ndim == 2 else transitions[:, None]
    return job_vec, perm, transitions
        

def pdr_family_edd(job_set: tf.Tensor) -> tf.Tensor:
    """ Family clustering and EDD Heuristic intra-cluster sorting.
    Note: the perm cannot be computed by F' = PF since underdetermined
    due to the non-square, ergo non-invertable matrices F,F'
    """
    # family inter-cluster sort
    job_vec, prior_perm, transitions = pdr_family(job_set)
    # loop over batch elements and transitions
    n_transitions = tf.shape(transitions)[-1]
    perm = []
    for b in range(tf.shape(job_set)[0]):
        perm_list = []
        for t in range(n_transitions):
            # the last transition
            if t == n_transitions - 1:
                # EDD sort of subset (upto end of sequence)
                idxs = np.argsort(job_vec[b][transitions[b][t]:, 2], axis=-1)
                # length of subset
                subset_length = tf.math.abs(transitions[b][t] - tf.shape(job_set)[1])
                # permutation matrix -> and permute subset
                local_perm = tf.one_hot(tf.cast(idxs, tf.int32), tf.cast(subset_length, tf.int32))
                job_vec[b][transitions[b][t]:] = tf.matmul(local_perm, job_vec[b][transitions[b][t]:])
            else:
                # EDD sort of subset (upto transition point)
                idxs = np.argsort(job_vec[b][transitions[b][t]:transitions[b][t+1], 2], axis=-1)
                # length of subset
                subset_length = tf.math.abs(transitions[b][t] - transitions[b][t+1])
                # permutation matrix -> and permute subset
                local_perm = tf.one_hot(tf.cast(idxs, tf.int32), tf.cast(subset_length, tf.int32))
                job_vec[b][transitions[b][t]:transitions[b][t+1]] = tf.matmul(
                    local_perm, job_vec[b][transitions[b][t]:transitions[b][t+1]])
            # add permutation to list (num items in to recover the true matrix index)
            perm_list.extend(idxs + np.array(len(perm_list)))
        perm.append(perm_list)
    perm = tf.one_hot(perm, tf.shape(job_set)[1])
    perm = tf.matmul(perm, prior_perm)
    return job_vec, tf.cast(perm, tf.float32)


def pdr_family_spt(job_set: tf.Tensor) -> tf.Tensor:
    """ Family clustering and SPT Heuristic intra-cluster sorting.
    Note: the perm cannot be computed by F' = PF since underdetermined
    due to the non-square, ergo non-invertable matrices F,F'
    """
    # family inter-cluster sort
    job_vec, prior_perm, transitions = pdr_family(job_set)
    # loop over batch elements and transitions
    n_transitions = tf.shape(transitions)[-1]
    perm = []
    for b in range(tf.shape(job_set)[0]):
        perm_list = []
        for t in range(n_transitions):
            # the last transition
            if t == n_transitions - 1:
                # SPT sort of subset (upto end of sequence)
                idxs = np.argsort(job_vec[b][transitions[b][t]:, 0], axis=-1)
                # length of subset
                subset_length = tf.math.abs(transitions[b][t] - tf.shape(job_set)[1])
                # permutation matrix -> and permute subset
                local_perm = tf.one_hot(tf.cast(idxs, tf.int32), tf.cast(subset_length, tf.int32))
                job_vec[b][transitions[b][t]:] = tf.matmul(local_perm, job_vec[b][transitions[b][t]:])
            else:
                # SPT sort of subset (upto transition point)
                idxs = np.argsort(job_vec[b][transitions[b][t]:transitions[b][t+1], 0], axis=-1)
                # length of subset
                subset_length = tf.math.abs(transitions[b][t] - transitions[b][t+1])
                # permutation matrix -> and permute subset
                local_perm = tf.one_hot(tf.cast(idxs, tf.int32), tf.cast(subset_length, tf.int32))
                job_vec[b][transitions[b][t]:transitions[b][t+1]] = tf.matmul(
                    local_perm, job_vec[b][transitions[b][t]:transitions[b][t+1]])
            # add permutation to list (plus t to recover the true matrix index)
            perm_list.extend(idxs + np.array(len(perm_list)))
        perm.append(perm_list)
    perm = tf.one_hot(perm, tf.shape(job_set)[1])
    perm = tf.matmul(perm, prior_perm)
    return job_vec, tf.cast(perm, tf.float32)
