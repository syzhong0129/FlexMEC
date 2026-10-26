from .runtime import *

def dis_pairs(coord_1, coord_2):
    coord_1_x = coord_1[-3]
    coord_1_y = coord_1[-2]
    coord_1_z = coord_1[-1]
    coord_2_x = coord_2[-3]
    coord_2_y = coord_2[-2]
    coord_2_z = coord_2[-1]
    distance = np.sqrt((float(coord_1_x) - float(coord_2_x)) ** 2 + (float(coord_1_y) - float(coord_2_y)) ** 2 + (
                float(coord_1_z) - float(coord_2_z)) ** 2)
    return distance


def dis_pairs_safe(coord_1, coord_2):
    """Return Euclidean distance, or infinity if either coordinate contains NaN."""
    c1 = np.asarray(coord_1, dtype=np.float64)[-3:]
    c2 = np.asarray(coord_2, dtype=np.float64)[-3:]
    if np.isnan(c1).any() or np.isnan(c2).any():
        return float('inf')
    return float(np.sqrt(((c1 - c2) ** 2).sum()))


def index_mink(data, k):
    Lst = data[:]
    index_k = []
    for i in range(k):
        index_i = Lst.index(min(Lst))
        index_k.append(index_i)
        Lst[index_i] = float('inf')
    return (index_k)


def CreateGearnetGraph(data, ag_node_attr, ab_node_attr):
    """Build antigen and antibody graphs with coordinate validity masks.

    Relations are sequential (0), precomputed contact (1) and spatial (2).
    Spatial edges exclude missing coordinates; contact edges are kept as given.
    Each result is [Graph, None, valid_mask], with all input nodes retained.
    """
    from torchdrug import data as drugdata

    def _build_one(coord, edge_pkl):
        coord = np.asarray(coord, dtype=np.float64)
        N = coord.shape[0]
        valid = ~np.isnan(coord).any(axis=1)

        # Relation 1: retain precomputed contact edges.
        edges_T = np.asarray(edge_pkl).T  # (E, 2)
        if edges_T.size == 0:
            edge_radius = []
        else:
            ones_col = np.ones((edges_T.shape[0], 1), dtype=int)
            edge_radius = np.hstack((edges_T, ones_col)).tolist()

        # Relation 0: connect i to i + 1, including nodes with missing coordinates.
        edge_seq = [[p, p + 1, 0] for p in range(N - 1)]

        # Relation 2: connect up to 10 nearest neighbors with valid coordinates.
        edge_knn = []
        valid_idx = np.flatnonzero(valid)
        if valid_idx.size > 1:
            valid_coord = coord[valid_idx][:, -3:].astype(np.float32, copy=False)
            distance_matrix = np.linalg.norm(
                valid_coord[:, None, :] - valid_coord[None, :, :],
                axis=-1,
            )
            np.fill_diagonal(distance_matrix, np.inf)

            k = min(10, valid_idx.size - 1)
            nearest_pos = np.argpartition(distance_matrix, kth=k - 1, axis=1)[:, :k]
            row_ids = np.arange(valid_idx.size)[:, None]
            nearest_pos = nearest_pos[
                row_ids,
                np.argsort(distance_matrix[row_ids, nearest_pos], axis=1),
            ]

            edge_knn = [
                [int(valid_idx[src]), int(valid_idx[dst]), 2]
                for src, dst_group in enumerate(nearest_pos)
                for dst in dst_group.tolist()
            ]

        all_edges = edge_seq + edge_radius + edge_knn
        graph = drugdata.Graph(all_edges, num_node=N, num_relation=3).to(device)
        valid_t = torch.as_tensor(valid, dtype=torch.bool).to(device)
        return [graph, None, valid_t]

    ag_edge_ind = _build_one(data["coord_AG"], data["edge_AG"])
    ab_edge_ind = _build_one(data["coord_AB"], data["edge_AB"])
    return ag_edge_ind, ab_edge_ind


def CreateKnearestEdge(data):
    num_nodes_AG = data["vertex_AB"].shape[0]
    edge_AG_10nearest_p = []
    edge_AG_10nearest_q = []
    for p in range(num_nodes_AG):
        dis_pq = []
        for q in range(num_nodes_AG):
            dis_pq.append(dis_pairs(data["coord_AG"][p], data["coord_AG"][q]))
        near10_q = index_mink(dis_pq, 11)
        del near10_q[near10_q.index(p)]
        near10_p = [p] * 10
        edge_AG_10nearest_p = edge_AG_10nearest_p + near10_p
        edge_AG_10nearest_q = edge_AG_10nearest_q + near10_q
    edge_AG_10nearest = [edge_AG_10nearest_p, edge_AG_10nearest_q]
    num_nodes_AB = data["vertex_AB"].shape[0]
    edge_AB_10nearest_p = []
    edge_AB_10nearest_q = []
    for p in range(num_nodes_AB):
        dis_pq = []
        for q in range(num_nodes_AB):
            dis_pq.append(dis_pairs(data["coord_AB"][p], data["coord_AB"][q]))
        near10_q = index_mink(dis_pq, 11)
        del near10_q[near10_q.index(p)]
        near10_p = [p] * 10
        edge_AB_10nearest_p = edge_AB_10nearest_p + near10_p
        edge_AB_10nearest_q = edge_AB_10nearest_q + near10_q
    edge_AB_10nearest = [edge_AB_10nearest_p, edge_AB_10nearest_q]
    return edge_AG_10nearest, edge_AB_10nearest
