import numpy as np
import torch
from typing import Tuple, List, Union


np.random.seed(1)


class Triangulation:
    def __init__(self, points: np.ndarray, simplices: np.ndarray):
        self.points = points
        self.simplices = simplices

    @classmethod
    def init_domain(cls, x: np.ndarray):
        """Generate a basic grid that covers all input points `x`."""
        _, dim = x.shape
        min_x = np.min(x, axis=0)
        max_x = np.max(x, axis=0)

        min_x, max_x = 1.1 * min_x - 0.1 * max_x, 2.1 * max_x - 1.1 * min_x
        vertices = np.array([[max_x[j] if i == j else min_x[j] for j in range(dim)] for i in range(dim + 1)])
        simplices = np.arange(dim + 1).reshape(1, dim + 1)
        return Triangulation(vertices, simplices)
    
    def compute_b2t_mask(self, simplices: np.ndarray, x: np.ndarray):
        t, v = simplices.shape

        # Convert point coordinates (x, y) to barycentric coordinates (u, v, w).
        # A point is considered inside the simplex if and only if all the barycentric coordinates are non-negative. 
        # [1] The tensor `x` containing point coordinates. It has shape [b, dim], where b is the number of points and dim is the number of dimensions.
        # [2] The tensor `shape_values` containing barycentric coordinates of points. It has shape [b, t, dim+1], where t is the number of simplices.
        coefficient_matrix= np.concatenate((self.points[simplices, :], np.ones((t, v, 1))), axis=2)  # [t, dim + 1, dim + 1]
        inv = np.linalg.inv(coefficient_matrix)  # [t, dim + 1, dim + 1]
        shape_values = np.einsum("tdv,bd->btv", inv[:, :-1, :], x) + inv[np.newaxis, :, -1, :] # [b, t, dim+1]

        # The tensor `b2t_mask` indicates the relationship between input points and simplices.  It has shape [b, t].
        # If b2t_mask[i, j] is True, it signifies that the i-th point is located inside the j-th simplex.
        b2t_mask = np.all(shape_values >= 0, axis=-1)  # [b, t]
        return b2t_mask
    
    _global_b2t_mask = None

    def sort_edges(self, new_edges: List[set], x_counts: Union[None, np.array], uniform=False) -> List[int]:
        """Consider an edge $E$ and let $S_E = {T_i | E \in T_i, i=1, 2, \cdots, t }$ be the set of simplices that contain edge $E$. 
        Furthermore, define $C(p, T) = # { p \in T | p \in P }$ and $\Xsi_{E, T} = 1 if E is the longest edge of T else 0$. 
        The evaluation of edge priority for splitting is as follows:
            \sum_{T \in S_E} (\Xsi_{E, T} * C(p, T)) / \sum_{T \in S_E} C(p, T)
        """
        if uniform:
            _edges = [list(edge) for edge in new_edges]
            e_vectors = np.array([self.points[i, :] - self.points[j, :] for i, j in _edges])
            e_len = np.linalg.norm(e_vectors, axis=-1)
            return list(e_len)

        _, dim = self.points.shape

        # The tensor `e2t_mask` indicates the relationship between new edges and global simplices. It has shape [e, t].
        # e2t_mask[i, j] = 1 signifies that the i-th edge is the edge of the j-th simplex.
        e2t_mask = np.array([[edge.issubset(set(simplex)) for simplex in self.simplices] for edge in new_edges])

        # [1] Get the relationship between new edges and local simplices.
        # [2] Get the relationship between input data and local simplices.
        dense_ids = [i for i, item in enumerate(np.sum(e2t_mask, axis=0)) if item != 0]
        local_simplices = self.simplices[dense_ids, :]
        local_e2t_mask = e2t_mask[:, dense_ids]  # [e, t]
        nonzero_mask = np.sum(self._global_b2t_mask, axis=1) != 0
        local_b2t_mask = self._global_b2t_mask[:, dense_ids][nonzero_mask, :]  # [b, t]
        if x_counts is not None:
            local_b2t_mask = local_b2t_mask * x_counts[nonzero_mask, np.newaxis]

        if dim == 1:  # For the one-dimensional case, any edge of any simplex is the longest. Therefore, we sort edges by counting the density.
            return list(np.einsum("et,bt->e", local_e2t_mask, local_b2t_mask))

        # The tensor `t2e_ids` list the (dim + 1) * dim / 2 edges of each simplices. It has shape [t, (dim + 1) * dim / 2, 2].
        # t2e_ids[i, j, :] indicates the j-th edge of the i-th simplex.
        t2e_ids = np.array([[[simplex[i], simplex[j]] for i in range(dim) for j in range(i + 1, dim + 1)] for simplex in local_simplices])

        # The tensor `t2e_len` list the length of (dim + 1) * dim / 2 edges in each simplex. It has shape [t, (dim + 1) * dim / 2].
        # t2e_len[i, j] indicates the length of j-th edge in the i-th simplex.
        t2e_vectors = np.array([[self.points[i, :] - self.points[j, :] for i, j in edge_ids] for edge_ids in t2e_ids])
        t2e_len = np.linalg.norm(t2e_vectors, axis=-1)

        # The tensor `t2e_max` indicates the longest edge in each simplex. It has shape [t, 2].
        # t2e_max[i, :] signifies that it is the longest edge of the i-th simplex.
        t2e_amax = np.argmax(t2e_len, axis=-1)
        t2e_max = t2e_ids[np.arange(len(t2e_ids)), t2e_amax, :]

        # The tensor `max_mask` indicates the relationship between global edges and simplices. It has shape [e, t].
        # max_mask[i, j] signifies that the i-th edge is the longest edge of the j-th simplex.
        max_mask = np.array([[1 if set([i, j]) == edge else 0 for i, j in t2e_max] for edge in new_edges])

        edge_priority = np.einsum("et,bt->e", max_mask, local_b2t_mask) / np.maximum(np.einsum("et,bt->e", local_e2t_mask, local_b2t_mask), 1)
        return list(edge_priority)
    
    _global_edges = None
    _global_edge_priority = None

    def add_simplices(self, new_edges: List[set], selected_edge: set, x: np.ndarray, x_counts=None, uniform=False):
        """Split a chosen edge and creating new simplices to refine the mesh."""
        if new_edges is None or selected_edge is None:  # If we start grid refinement from scratch.
            _, dim = self.points.shape
            self._global_b2t_mask = self.compute_b2t_mask(self.simplices, x)
            global_edges = {frozenset({simplex[i], simplex[j]}) for simplex in self.simplices for i in range(dim) for j in range(i + 1, dim + 1)}
            self._global_edges = [set(edge) for edge in global_edges]
            self._global_edge_priority = self.sort_edges(self._global_edges, x_counts, uniform)
            selected_edge = self._global_edges[np.argmax(self._global_edge_priority)]
        else:
            # Update `global_edges` and its evaluation before selecting the edge for splitting.
            try:
                local_edges, local_edge_priority = map(list, zip(*[(i, j) for i, j in zip(self._global_edges, self._global_edge_priority) if i not in new_edges + [selected_edge]]))
                new_edge_priority = self.sort_edges(new_edges, x_counts, uniform)
                self._global_edges = local_edges + new_edges
                self._global_edge_priority = local_edge_priority + new_edge_priority
            except ValueError:  # If all simplices are updated in the last mesh refinement.
                self._global_edges = new_edges
                self._global_edge_priority = self.sort_edges(new_edges, x_counts, uniform)

            # Find the edge which has the best evaluation.
            selected_edge = self._global_edges[np.argmax(self._global_edge_priority)]
        _selected_edge = list(selected_edge)

        # Update `self.points`
        p, dim = self.points.shape
        new_point = np.mean(self.points[_selected_edge, :], axis=0, keepdims=True)
        self.points = np.concatenate([self.points, new_point], axis=0)

        # Update `self.simplices`
        drop_ids = [i for i, simplex in enumerate(self.simplices) if _selected_edge[0] in simplex and _selected_edge[1] in simplex]
        protected_simplices = [list(simplex) for i, simplex in enumerate(self.simplices) if i not in drop_ids]
        upper_simplices = [[p if point_id == _selected_edge[1] else point_id for point_id in simplex] for i, simplex in enumerate(self.simplices) if i in drop_ids]
        lower_simplices = [[p if point_id == _selected_edge[0] else point_id for point_id in simplex] for i, simplex in enumerate(self.simplices) if i in drop_ids]
        self.simplices = np.array(protected_simplices + upper_simplices + lower_simplices)

        # Update `self._global_b2t_mask`
        t, _ = self.simplices.shape
        protected_b2t_mask = self._global_b2t_mask[:, [i for i in range(protected_simplices.__len__() + drop_ids.__len__()) if i not in drop_ids]]
        unsafe_ids = [i for i, count in enumerate(np.sum(self._global_b2t_mask[:, drop_ids], axis=1)) if count != 0] 
        upper_b2t_mask = np.zeros(shape=(x.__len__(), upper_simplices.__len__()))
        upper_b2t_mask[unsafe_ids, :] = self.compute_b2t_mask(np.array(upper_simplices), x[unsafe_ids, :])
        lower_b2t_mask = np.zeros(shape=(x.__len__(), lower_simplices.__len__()))
        lower_b2t_mask[unsafe_ids, :] = self.compute_b2t_mask(np.array(lower_simplices), x[unsafe_ids, :])
        self._global_b2t_mask = np.concatenate([protected_b2t_mask, upper_b2t_mask, lower_b2t_mask], axis=1)  # [b, t]

        # Update `new_edges`, it is a collection of edges in the updated simplices.
        new_edges = {frozenset({simplex[i], simplex[j]}) for simplex in upper_simplices + lower_simplices for i in range(dim) for j in range(i + 1, dim + 1)}
        new_edges = [set(edge) for edge in new_edges]
        return new_edges, selected_edge

    def neighbor_indices(self, point_idx):
        """Search the grid nodes that are near the input node, then return their indices."""
        collection = []
        for simplex in self.simplices:
            if point_idx in simplex:
                for idx in simplex:
                    if idx != point_idx:
                        collection.append(idx)
        return list(set(collection))

    @classmethod
    def unit_test(cls, dof=512):
        import numpy as np
        import matplotlib.pyplot as plt

        base_size = 64  # Number of values to generate
        radius = []
        for k in range(1, 4):
            radius.append(k * 2 * np.pi + np.random.normal(0., 1., k ** 2 * base_size))
        radius = np.concatenate(radius, axis=0)
        angles = 2 * np.pi * np.random.rand(radius.__len__())
        points = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)

        # Plot the points
        plt.scatter(points[:, 0], points[:, 1])

        init_points = np.array([[-8 * np.pi, -8 * np.pi], [8 * np.pi, -8 * np.pi], [8 * np.pi, 8 * np.pi], [-8 * np.pi, 8 * np.pi]])
        init_simplices = np.array([[0, 1, 3], [1, 2, 3]])
        tri = cls(init_points, init_simplices)

        new_edges, selected_edge = None, None
        for _ in range(dof):
            new_edges, selected_edge = tri.add_simplices(new_edges, selected_edge, points, uniform=False)
            print("Adding a node to the middle of edge {} \t Total: {:4d} nodes {:4d} simplices".format(
                selected_edge, tri.points.shape[0], tri.simplices.shape[0]), end="\r")

        plt.triplot(tri.points[:, 0], tri.points[:, 1], triangles=tri.simplices, c='red', linewidth=0.5)

        plt.axis('equal')
        plt.show()


class FiniteElement(torch.nn.Module):
    def __init__(self, tri):
        super(FiniteElement, self).__init__()

        p, dim = tri.points.shape
        t, _ = tri.simplices.shape
        coefficient_matrix = np.concatenate((tri.points[tri.simplices, :], np.ones((t, dim + 1, 1))), axis=2)  # [t, dim + 1, dim + 1]
        self.inv = torch.from_numpy(np.linalg.inv(coefficient_matrix)).type(torch.float32)  # [t, dim + 1, dim + 1]

        # The tensor `p2t_mask` indicates the relationship between vertices and simplices.  It has shape [p, t, dim+1].
        # If p2t_mask[i, j, k] is True, it signifies that the i-th node matches the k-th vertex of the j-th simplex.
        ids = [[ip, it, list(ips).index(ip)] for ip in range(p) for it, ips in enumerate(tri.simplices) if ip in ips]
        indices = torch.from_numpy(np.array(ids).T)
        values = torch.ones((indices.size()[1], ), dtype=torch.float32)
        size = [p, t, dim + 1]
        self.p2t_mask = torch.sparse_coo_tensor(indices, values, size).to_dense()  # [p, t, dim+1]

    def forward(self, x):
        # Convert point coordinates (x, y) to barycentric coordinates (u, v, w). 
        # A point is considered inside the simplex if and only if all the barycentric coordinates are non-negative. 
        # [1] The tensor `x` containing point coordinates. It has shape [b, dim], where b is the number of points and dim is the number of dimensions.
        # [2] The tensor `barycentric_values` containing barycentric coordinates of points. It has shape [b, t, dim+1], where t is the number of simplices.
        barycentric_values = torch.einsum("tdv,bd->btv", self.inv[:, :-1, :], x) + self.inv[None, :, -1, :] # [b, t, dim+1]

        # The tensor `b2t_mask` indicates the relationship between input points and simplices.  It has shape [b, t].
        # If b2t_mask[i, j] is True, it signifies that the i-th point is located inside the j-th simplex.
        b2t_mask = torch.all(barycentric_values >= 0, dim=-1)  # [b, t]

        # Gather the matching tensors p2t_mask and b2t_mask
        mask = b2t_mask[:, None, :, None] * self.p2t_mask[None, :, :, :]  # [b, p, t, dim+1]

        # Calculate the function values on all shape functions of the current basis function.
        # shape_values[i,j,k,l] signifies that the value of j-th basis function in i-th point
        # where j-th basis function should match l-th vertex of the k-th simplex.
        shape_values = mask * barycentric_values[:, None, :, :] # [b, p, t, dim+1]

        # Concatenate the shape functions corresponding to the basis function 
        # and update the values of the basis function within the support boundary of all shape functions.
        return torch.sum(shape_values, dim=(2, 3)) / (torch.nn.functional.relu(torch.sum(mask, dim=(2, 3)) - 1) + 1)

    @classmethod
    def unit_test_1(cls):
        from scipy.spatial import Delaunay
        from matplotlib import pyplot as plt

        # Generate random points
        points = np.random.rand(8, 2)

        # Compute Delaunay triangulation
        tri = Delaunay(points)

        # Compute basis functions
        basis = cls(tri)
        x = np.linspace(-0.02, 0.44, 500)
        y = np.linspace(0.05, 0.91, 500)
        X, Y = np.meshgrid(x, y)
        xv, yv = X.flatten(), Y.flatten()
        xy = np.stack((xv, yv), axis=1)
        zvs = basis(torch.from_numpy(xy).type(torch.float32)).numpy()

        fig1, ax = plt.subplots(figsize=(10, 10))
        for vtx_ids in tri.simplices:
            if 3 in vtx_ids:
                ax.fill(points[vtx_ids, 0], points[vtx_ids, 1], facecolor='black', alpha=0.25, hatch='///')
            else:
                ax.fill(points[vtx_ids, 0], points[vtx_ids, 1], facecolor='black', alpha=0.25)
        ax.triplot(points[:, 0], points[:, 1], triangles=tri.simplices, c='red', linewidth=0.5)
        ax.scatter(points[:, 0], points[:, 1], c='blue', label='Points')
        indptr, indices = tri.vertex_neighbor_vertices
        neighbor_indices = indices[indptr[3]:indptr[4]]
        neighbors = tri.points[neighbor_indices, :]
        ax.scatter(points[3, 0], points[3, 1], c='y', s=25)
        ax.scatter(neighbors[:, 0], neighbors[:, 1], c='m', s=25)
        ax.text(points[0, 0] + 0.003, points[0, 1], r'$e_0$', fontsize=24)
        ax.text(points[1, 0] - 0.021, points[1, 1] + 0.015, r'$e_1$', fontsize=24)
        ax.text(points[2, 0] + 0.01, points[2, 1], r'$e_2$', fontsize=24)
        ax.text(points[3, 0], points[3, 1] + 0.025, r'$e_3$', fontsize=24)
        ax.text(points[4, 0] + 0.01, points[4, 1], r'$e_4$', fontsize=24)
        ax.text(points[5, 0], points[5, 1] - 0.03, r'$e_5$', fontsize=24)
        ax.text(points[6, 0] + 0.01, points[6, 1], r'$e_6$', fontsize=24)
        ax.text(points[7, 0] - 0.025, points[7, 1], r'$e_7$', fontsize=24)
        for i, vtx_ids in enumerate(tri.simplices):
            ax.text(np.mean(points[vtx_ids, 0]), np.mean(points[vtx_ids, 1]), r'$T_{}$'.format(i), fontsize=24, horizontalalignment='center', verticalalignment='center')

        P, _ = tri.points.shape  # Number of subplots
        cols, rows = 4, 2

        # Create the figure and axes objects
        fig2, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(16, 6))

        # Iterate over the axes and plot your images
        for i in range(cols * rows):
            row = i // cols
            col = i % cols
            if i >= P:
                axes[row, col].axis('off')
                continue

            #  Plot the contour of a basis function
            Z = np.reshape(zvs[:, i], newshape=X.shape)
            contour = axes[row, col].contourf(X, Y, Z, levels=64, cmap='viridis', antialiased=True, alpha=0.5)
            cbar = plt.colorbar(contour)

            # Plot the triangles
            axes[row, col].triplot(points[:, 0], points[:, 1], tri.simplices, c='red', linewidth=0.5, label='Triangles')

            # Plot the points
            axes[row, col].scatter(points[:, 0], points[:, 1], c='blue', label='Points')

            # Plot the vertices of a basis function
            indptr, indices = tri.vertex_neighbor_vertices
            neighbor_indices = indices[indptr[i]:indptr[i + 1]]
            neighbors = tri.points[neighbor_indices, :]
            axes[row, col].scatter(points[i, 0], points[i, 1], c='y', s=25)
            axes[row, col].scatter(neighbors[:, 0], neighbors[:, 1], c='m', s=25)

            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            axes[row, col].set_xticklabels([])
            axes[row, col].set_yticklabels([])

        fig1.savefig('Figure_1.png', bbox_inches='tight')
        fig2.savefig('Figure_2.png', bbox_inches='tight')
        plt.show()

    @classmethod
    def unit_test_2(cls):
        from matplotlib import pyplot as plt

        # Generate random points
        points = np.random.rand(8, 2)
        xmin, ymin = np.min(points, axis=0) - 1
        xmax, ymax = np.max(points, axis=0) + 1

        # Compute Delaunay triangulation
        tri = Triangulation.init_domain(points)
        new_edges, selected_edge = None, None
        for _ in range(5):
            new_edges, selected_edge = tri.add_simplices(new_edges, selected_edge, points, uniform=False)
            print("Splitting edge {:<16} \t Update grid to {:4d} simplices".format(str(selected_edge), tri.simplices.shape[0]), end="\r")

        # Compute basis functions
        basis = cls(tri)
        x = np.linspace(xmin, xmax, 500)
        y = np.linspace(ymin, ymax, 500)
        X, Y = np.meshgrid(x, y)
        xv, yv = X.flatten(), Y.flatten()
        xy = np.stack((xv, yv), axis=1)
        zvs = basis(torch.from_numpy(xy).type(torch.float32)).numpy()

        import math
        from matplotlib import pyplot as plt

        P, _ = tri.points.shape  # Number of subplots
        cols, rows = 4, 2

        # Create the figure and axes objects
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(16, 6))

        # Iterate over the axes and plot your images
        for i in range(cols * rows):
            row = i // cols
            col = i % cols
            if i >= P:
                axes[row, col].axis('off')
                continue

            #  Plot the contour of a basis function
            Z = np.reshape(zvs[:, i], newshape=X.shape)
            contour = axes[row, col].contourf(X, Y, Z, levels=64, cmap='viridis', antialiased=True, alpha=0.5)
            cbar = plt.colorbar(contour)

            # Plot the triangles
            axes[row, col].triplot(tri.points[:, 0], tri.points[:, 1], tri.simplices, c='red', linewidth=0.5, label='Triangles')

            # Plot the points
            axes[row, col].scatter(points[:, 0], points[:, 1], c='blue', label='Points')

            neighbor_indices = tri.neighbor_indices(i)
            neighbors = tri.points[neighbor_indices, :]
            axes[row, col].scatter(tri.points[i, 0], tri.points[i, 1], c='y', s=25)
            axes[row, col].scatter(neighbors[:, 0], neighbors[:, 1], c='m', s=25)

            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            axes[row, col].set_xticklabels([])
            axes[row, col].set_yticklabels([])

        fig.savefig('Figure.png', bbox_inches='tight')
        plt.show()


class LagrangeEmbedding(torch.nn.Module):
    """Compute first-order elements, then create basis functions."""
    def auto_mesh(self, data: np.ndarray, dof: int, data_counts=None):
        _, dim = data.shape
        assert dof >= dim + 1  # In the case of first-order element, the degree of freedom (dof) must be the number of nodes

        print("Initializing the input domain")
        tri = Triangulation.init_domain(data)

        print("Refining the triangulation mesh")
        new_edges, selected_edge = None, None
        if data_counts is None:
            data, data_counts = np.unique(data, axis=0, return_counts=True)
        for _ in range(dof - dim - 1):
            new_edges, selected_edge = tri.add_simplices(new_edges, selected_edge, data, data_counts, uniform=False)
            print("[non-GPU process] Splitting edge {:<16} \t Update grid to {:4d} simplices, {:4d} dofs".format(str(selected_edge), tri.simplices.shape[0], tri.points.shape[0]), end="\r")

        print("\nCreating basis functions")
        # The tensor `coefficient_matrices` indicates the grid structure. It has shape [t, dim+1, dim+1].
        # coefficient_matrices[i, j, :-1] is the coordinates of the j-th vertex in the i-th simplex.
        assert dof == tri.points.shape[0]  # In the case of first-order element, the degree of freedom (dof) must be the number of nodes
        t, _ = tri.simplices.shape
        coefficient_matrices = np.concatenate((tri.points[tri.simplices, :], np.ones((t, dim + 1, 1))), axis=2)  # [t, dim + 1, dim + 1]
        
        # The sparse tensor `p2t_mask` indicates the relationship between vertices and simplices. It has shape [p, t, dim+1].
        # If p2t_mask[i, j, k] is True, it signifies that the i-th node matches the k-th vertex of the j-th simplex.
        indices = np.array([[ip, it, list(ips).index(ip)] for ip in range(dof) for it, ips in enumerate(tri.simplices) if ip in ips]).T
        values = np.ones(shape=(indices.shape[1], ), dtype=np.float32)
        size = [dof, t, dim + 1]
        p2t_mask = (indices, values, size)

        # The tensor `edges` indicates all edges. It has shape [e, 2].
        # If edges[i] signifies the edges[i, 0]-th node and the edges[i, 1]-th node are connected.
        edges = {frozenset({simplex[i], simplex[j]}) for simplex in tri.simplices for i in range(dim) for j in range(i + 1, dim + 1)}
        edges = np.array([list(edge) for edge in edges])

        return coefficient_matrices, p2t_mask, edges

    def __init__(self, data: torch.Tensor, dof:int, data_counts=None):
        super(LagrangeEmbedding, self).__init__()

        coefficient_matrices, (indices, values, size), edges = self.auto_mesh(data.numpy().astype(np.float32), dof, data_counts)
        self.inv = torch.from_numpy(np.linalg.inv(coefficient_matrices)).type(torch.float32)  # [t, dim + 1, dim + 1]
        self.p2t_mask = torch.sparse_coo_tensor(torch.from_numpy(indices), torch.from_numpy(values), size).to_dense()  # [p, t, dim+1]
        self.edges = torch.from_numpy(edges).type(torch.int64)  # [e, 2]

    def forward(self, x):  # [N, dim] -> [N, p]
        if self.inv.device != x.device:
            self.inv = self.inv.to(x.device)
        if self.p2t_mask.device != x.device:
            self.p2t_mask = self.p2t_mask.to(x.device)
        if self.edges.device != x.device:
            self.edges = self.edges.to(x.device)

        # Convert point coordinates (x, y) to barycentric coordinates (u, v, w). 
        # A point is considered inside the simplex if and only if all the barycentric coordinates are non-negative. 
        # [1] The tensor `x` containing point coordinates. It has shape [b, dim], where b is the number of points and dim is the number of dimensions.
        # [2] The tensor `barycentric_values` containing barycentric coordinates of points. It has shape [N, t, dim+1], where t is the number of simplices.
        barycentric_values = torch.einsum("tdv,bd->btv", self.inv[:, :-1, :], x) + self.inv[None, :, -1, :] # [N, t, dim+1]

        # The tensor `n2t_mask` indicates the relationship between input points and simplices.  It has shape [N, t].
        # If n2t_mask[i, j] is True, it signifies that the i-th point is located inside the j-th simplex.
        n2t_mask = torch.all(barycentric_values >= 0, dim=-1).to(torch.float32)  # [N, t]

        # Gather the matching tensors p2t_mask and n2t_mask
        mask = torch.einsum("bt,ptv->bp", n2t_mask, self.p2t_mask) # [N, p]
        
        # Calculate the function values on all shape functions of the current basis function.
        # shape_values[i,j] signifies that the value of j-th basis function in i-th point.
        # print(n2t_mask.shape, self.p2t_mask.shape, barycentric_values.shape)
        # shape_values = torch.einsum("bt,ptv,btv->bp", n2t_mask, self.p2t_mask, barycentric_values) # [N, p]
        shape_values = torch.einsum("ptv,btv->bpt", self.p2t_mask, barycentric_values) # [N, p]
        shape_values = torch.einsum("bt,bpt->bp", n2t_mask, shape_values) # [N, p]
        
        # Concatenate the shape functions corresponding to the basis function 
        # and update the values of the basis function within the support boundary of all shape functions.
        return shape_values / torch.maximum(mask, torch.ones_like(mask))


class Net(torch.nn.Module):
    def change_var(self, raw_data, **kwargs):
        return raw_data, None
    
    def post_proc(self, x, *args):
        return x
    
    def __init__(self, raw_data: np.ndarray, dof: int, output_size=1):
        super(Net, self).__init__()

        data, _ = self.change_var(raw_data)

        self.backbone = LagrangeEmbedding(data, dof)

        self.fc = torch.nn.Linear(dof, output_size, bias=False)

    def forward(self, x):
        x, _ = self.change_var(x)  # [N, ?] -> [N, ?]
        basis_values = self.backbone(x)  # [N, ?] -> [N, dof]
        x = self.post_proc(basis_values)  # [N, dof] -> [N, dof]
        return self.fc(x)
    
    @classmethod
    def unit_test_1(cls, dof=128, num_epochs=5, num_samples=3000, device="cuda"):
        _min_x, _max_x = 0.02, 0.5
        nodes = 1 / ((1 / _min_x - 1 / _max_x) * np.random.rand(2*num_samples, 1) + 1 / _max_x)
        values = np.random.normal(loc=np.sin(1 / nodes), scale=0.5 * (nodes ** 2), size=(2*num_samples, 1))
        (x_train, y_train) = np.asarray(nodes[:num_samples], dtype=np.float32), np.asarray(values[:num_samples], dtype=np.float32)
        (x_test, y_test) = np.asarray(nodes[num_samples:], dtype=np.float32), np.asarray(values[num_samples:], dtype=np.float32)

        print("Loading data")
        x_train_tensor = torch.tensor(x_train)
        y_train_tensor = torch.tensor(y_train)
        x_test_tensor = torch.tensor(x_test)
        y_test_tensor = torch.tensor(y_test)
        batch_size = 200
        data = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

        print("Creating model")
        model = Net(x_train_tensor, dof, output_size=1).to(device)

        # Count the number of trainable parameters
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of Trainable Parameters: {num_trainable_params}")

        criterion = torch.nn.SmoothL1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        print("Start training")
        # Training loop
        for epoch in range(num_epochs):
            # Set the model to training mode
            model.train()

            # Mini-batch training
            for x_batch, y_batch in data_loader:
                # Extract mini-batches
                inputs = x_batch.to(device)
                targets = y_batch.to(device)

                # Zero the gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

            # Set the model to evaluation mode
            model.eval()
            
            # Evaluation on the test set
            with torch.no_grad():
                mse = 0
                for i in range(0, x_test_tensor.size(0), batch_size):
                    # Extract mini-batches
                    inputs = x_test_tensor[i:i+batch_size].to(device)
                    targets = y_test_tensor[i:i+batch_size].to(device)
                    outputs = model(inputs)
                    mse += torch.sum(torch.square(outputs - targets))
                mse /= x_test_tensor.size(0) // batch_size
                clr = optimizer.param_groups[0]['lr']

            # Print training progress
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Test MSE: {mse:.4f}, Current LR: {clr:.4f}")

        from matplotlib import pyplot as plt
        plt.scatter(x_train.flatten(), y_train.flatten())
        x_flatten = np.linspace(0.02, 1.0, num=1000)
        y_flatten = model(torch.tensor(x_flatten[:, None], dtype=torch.float32).to(device)).detach().cpu().numpy().flatten()
        plt.plot(x_flatten, y_flatten, c='r')
        plt.show()
    
    @classmethod
    def unit_test_2(cls, dof=512, num_epochs=5, num_samples=10000, device="cuda"):
        input_examples = 2 * np.pi * np.random.rand(2*num_samples, 2)
        output_examples = np.sin(input_examples[:, [0]]) + np.cos(input_examples[:, [1]])

        (x_train, y_train) = np.asarray(input_examples[:num_samples], dtype=np.float32), np.asarray(output_examples[:num_samples], dtype=np.float32)
        (x_test, y_test) = np.asarray(input_examples[num_samples:], dtype=np.float32), np.asarray(output_examples[num_samples:], dtype=np.float32)

        print("Loading data")
        x_train_tensor = torch.tensor(x_train)
        y_train_tensor = torch.tensor(y_train)
        x_test_tensor = torch.tensor(x_test)
        y_test_tensor = torch.tensor(y_test)

        print("Creating model")
        model = Net(x_train_tensor, dof, output_size=1).to(device)

        # Count the number of trainable parameters
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of Trainable Parameters: {num_trainable_params}")

        criterion = torch.nn.SmoothL1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        batch_size = 100

        print("Start training")
        # Training loop
        for epoch in range(num_epochs):
            model.train()  # Set the model to training mode

            # Mini-batch training
            for i in range(0, x_train_tensor.size(0), batch_size):
                # Extract mini-batches
                inputs = x_train_tensor[i:i+batch_size].to(device)
                targets = y_train_tensor[i:i+batch_size].to(device)

                # Zero the gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()
            
            # Set the model to evaluation mode
            model.eval()
            
            # Evaluation on the test set
            with torch.no_grad():
                mse = 0
                for i in range(0, x_test_tensor.size(0), batch_size):
                    # Extract mini-batches
                    inputs = x_test_tensor[i:i+batch_size].to(device)
                    targets = y_test_tensor[i:i+batch_size].to(device)
                    outputs = model(inputs)
                    mse += torch.sum(torch.square(outputs - targets))
                mse /= x_test_tensor.size(0) // batch_size
                clr = optimizer.param_groups[0]['lr']

            # Print training progress
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Test MSE: {mse:.4f}, Current LR: {clr:.4f}")

        from matplotlib import pyplot as plt
        x = np.linspace(0, 2 * np.pi, 100)
        y = np.linspace(0, 2 * np.pi, 100)
        X, Y = np.meshgrid(x, y)
        x_tensor = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32)
        y_tensor = []
        for i in range(0, x_test_tensor.size(0), batch_size):
            inputs = x_tensor[i:i+batch_size].to(device)
            outputs = model(inputs)
            y_tensor.append(outputs)
        y_tensor = torch.concat(y_tensor, dim=0)
        Z = np.reshape(y_tensor.detach().cpu().numpy(), X.shape)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis')
        plt.show()


if __name__ == "__main__":
    # Triangulation.unit_test()
    # FiniteElement.unit_test_1()
    # FiniteElement.unit_test_2()
    Net.unit_test_1(dof=128, device="cuda")
    # Net.unit_test_2(dof=512, device="cuda")
