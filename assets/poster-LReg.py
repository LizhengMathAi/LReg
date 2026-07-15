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
    def unit_test_1d(cls, dof=128, num_samples=1000, save_path="adaptive_mesh_1d.gif", fps=12, step=2):
        """
        Animate one-dimensional adaptive mesh refinement.

        The input samples are concentrated near x=0, so the adaptive algorithm
        creates smaller intervals in regions with higher data density.

        Parameters
        ----------
        dof : int
            Number of edge-splitting operations.

        num_samples : int
            Number of one-dimensional input samples.

        save_path : str
            Output GIF filename.

        fps : int
            Frames per second.

        step : int
            Store one animation frame after every `step` refinements.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
        from matplotlib.lines import Line2D
        from matplotlib.patches import Rectangle

        # ---------------------------------------------------------------
        # Generate one-dimensional samples
        # ---------------------------------------------------------------
        min_x = 0.02
        max_x = 0.50

        # Same nonuniform sampling pattern used in Net.unit_test_1().
        samples = 1.0 / (
            (1.0 / min_x - 1.0 / max_x)
            * np.random.rand(num_samples, 1)
            + 1.0 / max_x
        )

        # Add a small deterministic vertical displacement only for plotting.
        # The adaptive refinement still uses the original one-dimensional data.
        sample_y = (
            0.055
            + 0.018 * np.sin(np.arange(num_samples) * 2.399)
        )

        # ---------------------------------------------------------------
        # Initial one-dimensional mesh
        # ---------------------------------------------------------------
        domain_padding = 0.03

        left_bound = min_x - domain_padding
        right_bound = max_x + domain_padding

        init_points = np.array(
            [
                [left_bound],
                [right_bound],
            ],
            dtype=float,
        )

        # A 1D simplex is an interval with two endpoint indices.
        init_simplices = np.array(
            [
                [0, 1],
            ],
            dtype=int,
        )

        tri = cls(
            init_points.copy(),
            init_simplices.copy(),
        )

        # ---------------------------------------------------------------
        # Run adaptive refinement and save mesh states
        # ---------------------------------------------------------------
        states = [
            (
                tri.points.copy(),
                tri.simplices.copy(),
                None,
            )
        ]

        new_edges = None
        selected_edge = None

        for split_index in range(dof):
            new_edges, selected_edge = tri.add_simplices(
                new_edges,
                selected_edge,
                samples,
                uniform=False,
            )

            print(
                f"\rSplit {split_index + 1:4d}/{dof}   "
                f"Nodes={len(tri.points):4d}   "
                f"Intervals={len(tri.simplices):4d}",
                end="",
            )

            if (
                (split_index + 1) % step == 0
                or split_index == dof - 1
            ):
                states.append(
                    (
                        tri.points.copy(),
                        tri.simplices.copy(),
                        None if selected_edge is None
                        else set(selected_edge),
                    )
                )

        print()

        # ---------------------------------------------------------------
        # Figure setup
        # ---------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(9.0, 3.2))

        y_min = -0.20
        y_max = 0.22
        mesh_y = 0.0

        def draw_background():
            """Draw the static background and sample distribution."""
            ax.clear()

            ax.set_xlim(left_bound, right_bound)
            ax.set_ylim(y_min, y_max)

            # Muted blue background.
            ax.add_patch(
                Rectangle(
                    (left_bound, y_min),
                    right_bound - left_bound,
                    y_max - y_min,
                    facecolor="#b9c9f2",
                    edgecolor="none",
                    alpha=0.55,
                    zorder=0,
                )
            )

            # White vertical grid.
            for x_value in np.linspace(
                left_bound,
                right_bound,
                13,
            ):
                ax.axvline(
                    x_value,
                    color="white",
                    linewidth=0.65,
                    alpha=0.9,
                    zorder=1,
                )

            # White horizontal grid.
            for y_value in np.linspace(
                y_min,
                y_max,
                7,
            ):
                ax.axhline(
                    y_value,
                    color="white",
                    linewidth=0.65,
                    alpha=0.9,
                    zorder=1,
                )

            # Input samples.
            ax.scatter(
                samples[:, 0],
                sample_y,
                s=11,
                color="#4f8662",
                edgecolors="none",
                alpha=0.72,
                zorder=2,
            )

            # Horizontal coordinate axis.
            ax.annotate(
                "",
                xy=(right_bound, mesh_y),
                xytext=(left_bound, mesh_y),
                arrowprops={
                    "arrowstyle": "->",
                    "color": "black",
                    "linewidth": 1.2,
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
                zorder=5,
            )

            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_visible(False)

        def draw_mesh(mesh_points, simplices, last_selected_edge):
            """Draw one-dimensional intervals and mesh nodes."""
            node_coordinates = mesh_points[:, 0]

            selected_coordinates = None

            if last_selected_edge is not None:
                selected_ids = list(last_selected_edge)

                if len(selected_ids) == 2:
                    selected_coordinates = sorted(
                        node_coordinates[selected_ids]
                    )

            # Draw each interval separately.
            for simplex in simplices:
                x_left, x_right = sorted(
                    node_coordinates[simplex]
                )

                is_selected_interval = (
                    selected_coordinates is not None
                    and np.allclose(
                        [x_left, x_right],
                        selected_coordinates,
                    )
                )

                ax.plot(
                    [x_left, x_right],
                    [mesh_y, mesh_y],
                    color=(
                        "#d8875f"
                        if is_selected_interval
                        else "#b35a5a"
                    ),
                    linewidth=(
                        3.2
                        if is_selected_interval
                        else 2.0
                    ),
                    solid_capstyle="round",
                    zorder=3,
                )

            # Mesh nodes.
            ax.scatter(
                node_coordinates,
                np.full_like(node_coordinates, mesh_y),
                s=26,
                color="#b35a5a",
                edgecolors="white",
                linewidths=0.6,
                zorder=4,
            )

            # Small vertical ticks make the mesh partition clearer.
            tick_height = 0.025

            for coordinate in node_coordinates:
                ax.plot(
                    [coordinate, coordinate],
                    [mesh_y - tick_height, mesh_y + tick_height],
                    color="#8f4545",
                    linewidth=0.8,
                    zorder=4,
                )

        # ---------------------------------------------------------------
        # Animation callback
        # ---------------------------------------------------------------
        def update(frame_index):
            mesh_points, mesh_simplices, selected = states[frame_index]

            draw_background()
            draw_mesh(
                mesh_points,
                mesh_simplices,
                selected,
            )

            split_count = min(
                frame_index * step,
                dof,
            )

            if frame_index == len(states) - 1:
                split_count = dof

            legend_title = (
                "Adaptive 1D mesh\n"
                f"DOF = {len(mesh_points)}\n"
                f"Intervals = {len(mesh_simplices)}\n"
                f"Splits = {split_count}/{dof}"
            )

            legend_handles = [
                Line2D(
                    [],
                    [],
                    linestyle="none",
                    marker="o",
                    markersize=5,
                    markerfacecolor="#4f8662",
                    markeredgecolor="none",
                    label="Input samples",
                ),
                Line2D(
                    [],
                    [],
                    color="#b35a5a",
                    linewidth=2.0,
                    marker="o",
                    markersize=5,
                    markerfacecolor="#b35a5a",
                    markeredgecolor="white",
                    label="Adaptive mesh",
                ),
            ]

            legend = ax.legend(
                handles=legend_handles,
                loc="upper right",
                frameon=True,
                fontsize=9,
                title=legend_title,
                title_fontsize=9,
            )

            legend.get_frame().set_facecolor("white")
            legend.get_frame().set_alpha(0.86)
            legend.get_frame().set_edgecolor("none")

            return []

        animation = FuncAnimation(
            fig,
            update,
            frames=len(states),
            interval=1000 / max(fps, 1),
            blit=False,
            repeat=True,
        )

        animation.save(
            save_path,
            writer=PillowWriter(fps=fps),
            dpi=120,
        )

        print(f"GIF saved to {save_path}")

        # -------------------------------------------------------
        # Save the final frame
        # -------------------------------------------------------
        update(len(states) - 1)

        png_path = save_path.replace(".gif", ".png")

        fig.savefig(
            png_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02,
        )

        # -------------------------------------------------------
        # Overlay a play icon onto the same PNG
        # -------------------------------------------------------
        from PIL import Image, ImageDraw

        img = Image.open(png_path).convert("RGBA")

        overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        w, h = img.size
        cx, cy = w // 2, h // 2

        radius = int(min(w, h) * 0.075)

        # Semi-transparent black circle
        draw.ellipse(
            (
                cx - radius,
                cy - radius,
                cx + radius,
                cy + radius,
            ),
            fill=(0, 0, 0, 110),
        )

        # White play triangle
        triangle = [
            (cx - radius * 0.28, cy - radius * 0.45),
            (cx - radius * 0.28, cy + radius * 0.45),
            (cx + radius * 0.45, cy),
        ]

        draw.polygon(
            triangle,
            fill=(255, 255, 255, 255),
        )

        # Overwrite the original PNG
        Image.alpha_composite(img, overlay).convert("RGB").save(png_path)

        print(f"Final frame with play icon saved to {png_path}")

        plt.show()
    
    @classmethod
    def unit_test_2d(cls, dof=512, save_path="adaptive_mesh_2d.gif", fps=12, step=4):
        """
        Animate adaptive mesh refinement and save it as a GIF.

        Parameters
        ----------
        dof : int
            Number of edge splits.
        save_path : str
            Output GIF filename.
        fps : int
            Frames per second.
        step : int
            Save one frame every 'step' refinements.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Polygon
        from matplotlib.animation import FuncAnimation, PillowWriter

        # ---------------------------------------------------------------
        # Generate spiral data
        # ---------------------------------------------------------------
        base_size = 64
        radius = []

        for k in range(1, 4):
            radius.append(
                k * 2 * np.pi +
                np.random.normal(0.0, 1.0, k**2 * base_size)
            )

        radius = np.concatenate(radius)
        angles = 2 * np.pi * np.random.rand(len(radius))

        points = np.stack(
            [radius * np.cos(angles),
            radius * np.sin(angles)],
            axis=1,
        )

        bound = 8 * np.pi

        init_points = np.array([
            [-bound, -bound],
            [ bound, -bound],
            [ bound,  bound],
            [-bound,  bound],
        ])

        init_simplices = np.array([
            [0, 1, 3],
            [1, 2, 3],
        ])

        # ---------------------------------------------------------------
        # Run refinement and store mesh states
        # ---------------------------------------------------------------
        tri = cls(init_points.copy(), init_simplices.copy())

        states = [(tri.points.copy(), tri.simplices.copy())]

        new_edges = None
        selected_edge = None

        for i in range(dof):

            new_edges, selected_edge = tri.add_simplices(
                new_edges,
                selected_edge,
                points,
                uniform=False,
            )

            print(
                f"\rSplit {i+1:4d}/{dof}   "
                f"Nodes={len(tri.points):4d}   "
                f"Triangles={len(tri.simplices):4d}",
                end="",
            )

            if ((i + 1) % step == 0) or (i == dof - 1):
                states.append(
                    (
                        tri.points.copy(),
                        tri.simplices.copy(),
                    )
                )

        print()

        # ---------------------------------------------------------------
        # Figure
        # ---------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(5.8, 5.8))

        def draw_mesh(points_mesh, simplices):

            ax.clear()

            ax.set_aspect("equal")
            ax.set_xlim(-bound, bound)
            ax.set_ylim(-bound, bound)

            # background
            ax.add_patch(
                Polygon(
                    [
                        (-bound, -bound),
                        ( bound, -bound),
                        ( bound,  bound),
                        (-bound,  bound),
                    ],
                    closed=True,
                    facecolor="#b9c9f2",
                    edgecolor="none",
                    alpha=0.55,
                    zorder=0,
                )
            )

            # grid
            grid_step = 0.16 * bound

            for x in np.arange(-bound, bound + grid_step, grid_step):
                ax.axvline(x, color="white", lw=0.55, zorder=1)
                ax.axhline(x, color="white", lw=0.55, zorder=1)

            # triangulation
            ax.triplot(
                points_mesh[:, 0],
                points_mesh[:, 1],
                simplices,
                color="#b35a5a",
                linewidth=0.55,
                alpha=0.9,
                zorder=2,
            )

            # spiral points
            ax.scatter(
                points[:, 0],
                points[:, 1],
                s=10,
                color="#4f8662",
                edgecolors="none",
                alpha=0.9,
                zorder=3,
            )

            # axes
            ax.annotate(
                "",
                (-bound, 0),
                (bound, 0),
                arrowprops=dict(arrowstyle="<-", lw=1.2),
            )

            ax.annotate(
                "",
                (0, -bound),
                (0, bound),
                arrowprops=dict(arrowstyle="<-", lw=1.2),
            )

            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_visible(False)

        # ---------------------------------------------------------------
        # Animation callback
        # ---------------------------------------------------------------
        def update(frame):

            mesh_points, mesh_triangles = states[frame]

            draw_mesh(mesh_points, mesh_triangles)

            split_count = min(frame * step, dof)

            if frame == len(states) - 1:
                split_count = dof

            legend_title = (
                "Adaptive 2D mesh\n"
                f"DOF = {len(mesh_points)}\n"
                f"Triangles = {len(mesh_triangles)}\n"
                f"Splits = {split_count}/{dof}"
            )

            legend_handles = [
                Line2D(
                    [],
                    [],
                    linestyle="none",
                    marker="o",
                    markersize=5,
                    markerfacecolor="#4f8662",
                    markeredgecolor="none",
                    label="Input samples",
                ),
                Line2D(
                    [],
                    [],
                    color="#b35a5a",
                    linewidth=1.5,
                    label="Adaptive mesh",
                ),
            ]

            legend = ax.legend(
                handles=legend_handles,
                loc="upper right",
                frameon=True,
                fontsize=9,
                title=legend_title,
                title_fontsize=9,
            )

            legend.get_frame().set_facecolor("white")
            legend.get_frame().set_alpha(0.86)
            legend.get_frame().set_edgecolor("none")

            return []

        ani = FuncAnimation(
            fig,
            update,
            frames=len(states),
            interval=1000 / fps,
            blit=False,
        )

        ani.save(
            save_path,
            writer=PillowWriter(fps=fps),
        )

        print(f"GIF saved to {save_path}")

        # -------------------------------------------------------
        # Save the final frame
        # -------------------------------------------------------
        update(len(states) - 1)

        png_path = save_path.replace(".gif", ".png")

        fig.savefig(
            png_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02,
        )

        # -------------------------------------------------------
        # Overlay a play icon onto the same PNG
        # -------------------------------------------------------
        from PIL import Image, ImageDraw

        img = Image.open(png_path).convert("RGBA")

        overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        w, h = img.size
        cx, cy = w // 2, h // 2

        radius = int(min(w, h) * 0.075)

        # Semi-transparent black circle
        draw.ellipse(
            (
                cx - radius,
                cy - radius,
                cx + radius,
                cy + radius,
            ),
            fill=(0, 0, 0, 110),
        )

        # White play triangle
        triangle = [
            (cx - radius * 0.28, cy - radius * 0.45),
            (cx - radius * 0.28, cy + radius * 0.45),
            (cx + radius * 0.45, cy),
        ]

        draw.polygon(
            triangle,
            fill=(255, 255, 255, 255),
        )

        # Overwrite the original PNG
        Image.alpha_composite(img, overlay).convert("RGB").save(png_path)

        print(f"Final frame with play icon saved to {png_path}")

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
    def unit_test_1d(
        cls,
        dof=128,
        num_epochs=8,
        num_samples=3000,
        device="cuda",
        save_path="LReg.gif",
        fps=2,
        basis_stride=1,
    ):
        """
        Train the one-dimensional Lagrange regressor and save the
        learning process as a GIF.

        Every one-dimensional Lagrange basis function is drawn directly
        from its center node and neighboring mesh nodes. This avoids
        missing very narrow basis functions near densely sampled regions.

        Parameters
        ----------
        dof : int
            Number of Lagrange basis functions.

        num_epochs : int
            Number of training epochs.

        num_samples : int
            Number of training samples and test samples.

        device : str
            Torch device, such as "cuda" or "cpu".

        save_path : str
            Output GIF path.

        fps : int
            Frames per second.

        basis_stride : int
            Draw every `basis_stride`-th basis function. Use 1 to draw
            every basis function.
        """
        import numpy as np
        import torch
        from matplotlib import pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
        from matplotlib.patches import Rectangle

        # ---------------------------------------------------------------
        # Validate arguments
        # ---------------------------------------------------------------
        if dof < 2:
            raise ValueError("dof must be at least 2.")

        if num_epochs < 1:
            raise ValueError("num_epochs must be positive.")

        if num_samples < 1:
            raise ValueError("num_samples must be positive.")

        if fps < 1:
            raise ValueError("fps must be positive.")

        if basis_stride < 1:
            raise ValueError("basis_stride must be positive.")

        # ---------------------------------------------------------------
        # Device selection
        # ---------------------------------------------------------------
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA is unavailable. Falling back to CPU.")
            device = "cpu"

        # ---------------------------------------------------------------
        # Generate training and test data
        # ---------------------------------------------------------------
        min_x = 0.02
        max_x = 0.50

        nodes = 1.0 / (
            (1.0 / min_x - 1.0 / max_x)
            * np.random.rand(2 * num_samples, 1)
            + 1.0 / max_x
        )

        values = np.random.normal(
            loc=np.sin(1.0 / nodes),
            scale=0.5 * nodes**2,
            size=(2 * num_samples, 1),
        )

        x_train = np.asarray(
            nodes[:num_samples],
            dtype=np.float32,
        )

        y_train = np.asarray(
            values[:num_samples],
            dtype=np.float32,
        )

        x_test = np.asarray(
            nodes[num_samples:],
            dtype=np.float32,
        )

        y_test = np.asarray(
            values[num_samples:],
            dtype=np.float32,
        )

        # ---------------------------------------------------------------
        # Prepare tensors and data loader
        # ---------------------------------------------------------------
        print("Loading data")

        x_train_tensor = torch.tensor(
            x_train,
            dtype=torch.float32,
        )

        y_train_tensor = torch.tensor(
            y_train,
            dtype=torch.float32,
        )

        x_test_tensor = torch.tensor(
            x_test,
            dtype=torch.float32,
        )

        y_test_tensor = torch.tensor(
            y_test,
            dtype=torch.float32,
        )

        batch_size = 200

        dataset = torch.utils.data.TensorDataset(
            x_train_tensor,
            y_train_tensor,
        )

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        # ---------------------------------------------------------------
        # Create the Lagrange regressor
        # ---------------------------------------------------------------
        print("Creating model")

        model = cls(
            x_train_tensor,
            dof,
            output_size=1,
        ).to(device)

        num_trainable_params = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

        print(
            "Number of Trainable Parameters: "
            f"{num_trainable_params}"
        )

        criterion = torch.nn.SmoothL1Loss()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.1,
        )

        # ---------------------------------------------------------------
        # Dense curve used for prediction and reference function
        # ---------------------------------------------------------------
        x_curve = np.linspace(
            min_x,
            max_x,
            1200,
            dtype=np.float32,
        )

        x_curve_tensor = torch.tensor(
            x_curve[:, None],
            dtype=torch.float32,
            device=device,
        )

        y_reference = np.sin(1.0 / x_curve)

        # ---------------------------------------------------------------
        # Recover the coordinate of every one-dimensional basis node
        # ---------------------------------------------------------------
        model.eval()

        with torch.no_grad():
            inverse_matrices = (
                model.backbone.inv
                .detach()
                .cpu()
                .numpy()
            )

            p2t_mask = (
                model.backbone.p2t_mask
                .detach()
                .cpu()
                .numpy()
            )

        # model.backbone.inv contains the inverse of each interval's
        # coefficient matrix. Invert it again to recover:
        #
        #     [[x_left,  1],
        #      [x_right, 1]]
        #
        coefficient_matrices = np.linalg.inv(
            inverse_matrices
        )

        interval_coordinates = (
            coefficient_matrices[:, :, 0]
        )

        number_of_basis_functions = p2t_mask.shape[0]

        basis_node_coordinates = np.empty(
            number_of_basis_functions,
            dtype=np.float64,
        )

        for basis_index in range(
            number_of_basis_functions
        ):
            # Each nonzero element identifies an interval and the local
            # vertex corresponding to this global basis function.
            matches = np.argwhere(
                p2t_mask[basis_index] > 0.5
            )

            if len(matches) == 0:
                raise RuntimeError(
                    "No mesh coordinate was found for basis "
                    f"function {basis_index}."
                )

            interval_index = int(matches[0, 0])
            local_vertex_index = int(matches[0, 1])

            basis_node_coordinates[basis_index] = (
                interval_coordinates[
                    interval_index,
                    local_vertex_index,
                ]
            )

        # Sort the global basis functions from left to right.
        basis_order = np.argsort(
            basis_node_coordinates
        )

        sorted_basis_nodes = (
            basis_node_coordinates[basis_order]
        )

        # Check for duplicate node coordinates.
        coordinate_differences = np.diff(
            sorted_basis_nodes
        )

        if np.any(coordinate_differences <= 0):
            raise RuntimeError(
                "Recovered basis-node coordinates are not strictly "
                "increasing. Check the one-dimensional mesh."
            )

        print(
            f"Recovered {len(sorted_basis_nodes)} "
            "one-dimensional basis functions."
        )

        print(
            "Basis-node range: "
            f"[{sorted_basis_nodes[0]:.6f}, "
            f"{sorted_basis_nodes[-1]:.6f}]"
        )

        # ---------------------------------------------------------------
        # Store animation frames
        # ---------------------------------------------------------------
        prediction_frames = []
        loss_history = []
        mse_history = []

        def record_frame(loss_value, mse_value):
            """Record the model's current prediction."""
            model.eval()

            with torch.no_grad():
                prediction = (
                    model(x_curve_tensor)
                    .detach()
                    .cpu()
                    .numpy()
                    .flatten()
                )

            prediction_frames.append(prediction)
            loss_history.append(loss_value)
            mse_history.append(mse_value)

        # Frame zero: untrained model.
        record_frame(
            loss_value=np.nan,
            mse_value=np.nan,
        )

        # ---------------------------------------------------------------
        # Training loop
        # ---------------------------------------------------------------
        print("Start training")

        for epoch in range(num_epochs):
            model.train()

            epoch_loss = 0.0
            number_of_batches = 0

            for x_batch, y_batch in data_loader:
                inputs = x_batch.to(device)
                targets = y_batch.to(device)

                optimizer.zero_grad(
                    set_to_none=True
                )

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                number_of_batches += 1

            average_loss = (
                epoch_loss
                / max(number_of_batches, 1)
            )

            # -----------------------------------------------------------
            # Evaluate test MSE
            # -----------------------------------------------------------
            model.eval()

            squared_error = 0.0
            number_of_test_values = 0

            with torch.no_grad():
                for start in range(
                    0,
                    x_test_tensor.size(0),
                    batch_size,
                ):
                    inputs = x_test_tensor[
                        start:start + batch_size
                    ].to(device)

                    targets = y_test_tensor[
                        start:start + batch_size
                    ].to(device)

                    outputs = model(inputs)

                    squared_error += torch.sum(
                        torch.square(
                            outputs - targets
                        )
                    ).item()

                    number_of_test_values += (
                        targets.numel()
                    )

            mse = (
                squared_error
                / max(number_of_test_values, 1)
            )

            current_lr = (
                optimizer.param_groups[0]["lr"]
            )

            record_frame(
                loss_value=average_loss,
                mse_value=mse,
            )

            print(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Loss: {average_loss:.4f}, "
                f"Test MSE: {mse:.4f}, "
                f"Current LR: {current_lr:.4f}"
            )

        # ---------------------------------------------------------------
        # Determine stable plot limits for every frame
        # ---------------------------------------------------------------
        all_predictions = np.concatenate(
            prediction_frames
        )

        all_y = np.concatenate(
            [
                y_train.flatten(),
                y_reference,
                all_predictions,
            ]
        )

        y_range = np.max(all_y) - np.min(all_y)

        if y_range == 0:
            y_range = 1.0

        y_padding = 0.08 * y_range

        x_plot_min = 0.0
        x_plot_max = max_x

        y_plot_min = (
            np.min(all_y) - y_padding
        )

        y_plot_max = (
            np.max(all_y) + y_padding
        )

        # ---------------------------------------------------------------
        # Basis-function display configuration
        # ---------------------------------------------------------------
        basis_baseline = 0.0

        # All tents have the same peak height.
        basis_height = 0.42 * max(
            y_plot_max,
            1.0,
        )

        basis_color = "#7089c4"

        # ---------------------------------------------------------------
        # Create animation figure
        # ---------------------------------------------------------------
        fig, ax = plt.subplots(
            figsize=(8.2, 5.2)
        )

        fig.tight_layout(pad=0.2)

        def draw_background():
            """Draw the canvas and all static plot elements."""
            ax.clear()

            ax.set_xlim(
                x_plot_min,
                x_plot_max,
            )

            ax.set_ylim(
                y_plot_min,
                y_plot_max,
            )

            # Muted blue canvas.
            ax.add_patch(
                Rectangle(
                    (x_plot_min, y_plot_min),
                    x_plot_max - x_plot_min,
                    y_plot_max - y_plot_min,
                    facecolor="#b9c9f2",
                    edgecolor="none",
                    alpha=0.55,
                    zorder=0,
                )
            )

            # White grid.
            for x_value in np.linspace(
                x_plot_min,
                x_plot_max,
                11,
            ):
                ax.axvline(
                    x_value,
                    color="white",
                    linewidth=0.65,
                    alpha=0.9,
                    zorder=1,
                )

            for y_value in np.linspace(
                y_plot_min,
                y_plot_max,
                9,
            ):
                ax.axhline(
                    y_value,
                    color="white",
                    linewidth=0.65,
                    alpha=0.9,
                    zorder=1,
                )

            # -----------------------------------------------------------
            # Draw every exact tent-shaped basis function
            # -----------------------------------------------------------
            for sorted_index in range(
                0,
                len(sorted_basis_nodes),
                basis_stride,
            ):
                center = sorted_basis_nodes[
                    sorted_index
                ]

                # The support of an interior basis function extends
                # from the preceding mesh node to the following node.
                if sorted_index == 0:
                    left = center
                else:
                    left = sorted_basis_nodes[
                        sorted_index - 1
                    ]

                if (
                    sorted_index
                    == len(sorted_basis_nodes) - 1
                ):
                    right = center
                else:
                    right = sorted_basis_nodes[
                        sorted_index + 1
                    ]

                # Skip only bases whose entire support is outside the
                # visible horizontal range.
                if (
                    right < x_plot_min
                    or left > x_plot_max
                ):
                    continue

                # Left boundary basis: one-sided descending function.
                if np.isclose(left, center):
                    tent_x = np.array(
                        [center, right],
                        dtype=float,
                    )

                    tent_y = np.array(
                        [
                            basis_height,
                            basis_baseline,
                        ],
                        dtype=float,
                    )

                # Right boundary basis: one-sided ascending function.
                elif np.isclose(right, center):
                    tent_x = np.array(
                        [left, center],
                        dtype=float,
                    )

                    tent_y = np.array(
                        [
                            basis_baseline,
                            basis_height,
                        ],
                        dtype=float,
                    )

                # Interior basis: complete triangular tent.
                else:
                    tent_x = np.array(
                        [
                            left,
                            center,
                            right,
                        ],
                        dtype=float,
                    )

                    tent_y = np.array(
                        [
                            basis_baseline,
                            basis_height,
                            basis_baseline,
                        ],
                        dtype=float,
                    )

                ax.fill_between(
                    tent_x,
                    basis_baseline,
                    tent_y,
                    color=basis_color,
                    alpha=0.055,
                    linewidth=0,
                    zorder=1.4,
                )

                ax.plot(
                    tent_x,
                    tent_y,
                    color=basis_color,
                    linewidth=0.82,
                    alpha=0.80,
                    solid_capstyle="round",
                    solid_joinstyle="round",
                    zorder=1.5,
                )

            # Training samples.
            ax.scatter(
                x_train.flatten(),
                y_train.flatten(),
                s=11,
                color="#4f8662",
                edgecolors="none",
                alpha=0.62,
                label="Training samples",
                zorder=2,
            )

            # Reference function.
            ax.plot(
                x_curve,
                y_reference,
                color="#6c6c6c",
                linewidth=1.15,
                linestyle="--",
                alpha=0.8,
                label=r"$\sin(1/x)$",
                zorder=3,
            )

            # Arrow-style coordinate axes.
            arrow_style = {
                "arrowstyle": "->",
                "color": "black",
                "linewidth": 1.2,
                "shrinkA": 0,
                "shrinkB": 0,
            }

            ax.annotate(
                "",
                xy=(x_plot_max, 0.0),
                xytext=(x_plot_min, 0.0),
                arrowprops=arrow_style,
                zorder=5,
            )

            ax.annotate(
                "",
                xy=(0.0, y_plot_max),
                xytext=(0.0, y_plot_min),
                arrowprops=arrow_style,
                zorder=5,
            )

            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_visible(False)

        def update(frame_index):
            """Draw one animation frame."""
            draw_background()

            learned_line, = ax.plot(
                x_curve,
                prediction_frames[frame_index],
                color="#b35a5a",
                linewidth=2.0,
                label="Learned function",
                zorder=4,
            )

            if frame_index == 0:
                legend_title = (
                    "Lagrange Regressor\n"
                    f"DOF = {dof}\n"
                    f"Parameters = {num_trainable_params}\n"
                    f"Epoch = 0/{num_epochs}\n"
                    "Loss = N/A\n"
                    "Test MSE = N/A"
                )
            else:
                legend_title = (
                    "Lagrange Regressor\n"
                    f"DOF = {dof}\n"
                    f"Parameters = {num_trainable_params}\n"
                    f"Epoch = {frame_index}/{num_epochs}\n"
                    f"Loss = "
                    f"{loss_history[frame_index]:.4f}\n"
                    f"Test MSE = "
                    f"{mse_history[frame_index]:.4f}"
                )

            legend = ax.legend(
                loc="lower right",
                frameon=True,
                fontsize=9,
                title=legend_title,
                title_fontsize=10,
            )

            legend.get_frame().set_facecolor(
                "white"
            )

            legend.get_frame().set_alpha(
                0.85
            )

            legend.get_frame().set_edgecolor(
                "none"
            )

            return [learned_line]

        animation = FuncAnimation(
            fig,
            update,
            frames=len(prediction_frames),
            interval=1000 / fps,
            blit=False,
            repeat=True,
        )

        animation.save(
            save_path,
            writer=PillowWriter(fps=fps),
            dpi=120,
        )

        print(f"GIF saved to {save_path}")

        # -------------------------------------------------------
        # Save the final frame
        # -------------------------------------------------------
        update(len(prediction_frames) - 1)

        png_path = save_path.replace(".gif", ".png")

        fig.savefig(
            png_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02,
        )

        # -------------------------------------------------------
        # Overlay a play icon onto the same PNG
        # -------------------------------------------------------
        from PIL import Image, ImageDraw

        img = Image.open(png_path).convert("RGBA")

        overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        w, h = img.size
        cx, cy = w // 2, h // 2

        radius = int(min(w, h) * 0.075)

        # Semi-transparent black circle
        draw.ellipse(
            (
                cx - radius,
                cy - radius,
                cx + radius,
                cy + radius,
            ),
            fill=(0, 0, 0, 110),
        )

        # White play triangle
        triangle = [
            (cx - radius * 0.28, cy - radius * 0.45),
            (cx - radius * 0.28, cy + radius * 0.45),
            (cx + radius * 0.45, cy),
        ]

        draw.polygon(
            triangle,
            fill=(255, 255, 255, 255),
        )

        # Overwrite the original PNG
        Image.alpha_composite(img, overlay).convert("RGB").save(png_path)

        print(f"Final frame with play icon saved to {png_path}")

        plt.show()

    @classmethod
    def unit_test_2d(cls, dof=512, num_epochs=5, num_samples=10000, device="cuda"):
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


class MLP(torch.nn.Module):
    """Traditional multilayer perceptron for one-dimensional regression."""

    def __init__(
        self,
        hidden_width=128,
        hidden_layers=3,
        activation="tanh",
        output_size=1,
    ):
        super().__init__()

        if hidden_width < 1:
            raise ValueError("hidden_width must be positive.")

        if hidden_layers < 1:
            raise ValueError("hidden_layers must be positive.")

        activation_table = {
            "tanh": torch.nn.Tanh,
            "relu": torch.nn.ReLU,
            "gelu": torch.nn.GELU,
            "silu": torch.nn.SiLU,
        }

        activation_key = activation.lower()

        if activation_key not in activation_table:
            raise ValueError(
                f"Unsupported activation '{activation}'. "
                f"Choose from {tuple(activation_table)}."
            )

        activation_class = activation_table[activation_key]

        layers = [
            torch.nn.Linear(1, hidden_width),
            activation_class(),
        ]

        for _ in range(hidden_layers - 1):
            layers.extend(
                [
                    torch.nn.Linear(hidden_width, hidden_width),
                    activation_class(),
                ]
            )

        layers.append(
            torch.nn.Linear(hidden_width, output_size)
        )

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    @classmethod
    def unit_test_1d(
        cls,
        hidden_width=128,
        hidden_layers=3,
        activation="tanh",
        num_epochs=1000,
        num_samples=3000,
        batch_size=200,
        learning_rate=1.0e-3,
        device="cuda",
        save_path="MLP.gif",
        fps=8,
        frame_step=5,
        seed=1,
    ):
        """
        Train a traditional MLP to approximate sin(1/x) and save a GIF.

        Parameters
        ----------
        hidden_width : int
            Number of neurons in each hidden layer.

        hidden_layers : int
            Number of hidden layers.

        activation : str
            Activation function: "tanh", "relu", "gelu", or "silu".

        num_epochs : int
            Total number of training epochs.

        num_samples : int
            Number of training samples and test samples.

        batch_size : int
            Mini-batch size.

        learning_rate : float
            Adam learning rate.

        device : str
            Torch device, such as "cuda" or "cpu".

        save_path : str
            Output GIF path.

        fps : int
            GIF frames per second.

        frame_step : int
            Record one animation frame every `frame_step` epochs.

        seed : int
            NumPy and PyTorch random seed.
        """
        import numpy as np
        import torch
        from matplotlib import pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
        from matplotlib.patches import Rectangle

        if num_epochs < 1:
            raise ValueError("num_epochs must be positive.")

        if frame_step < 1:
            raise ValueError("frame_step must be positive.")

        if fps < 1:
            raise ValueError("fps must be positive.")

        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if device.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA is unavailable. Falling back to CPU.")
            device = "cpu"

        # -----------------------------------------------------------
        # Generate the same nonuniform noisy data used by Net
        # -----------------------------------------------------------
        min_x = 0.02
        max_x = 0.50

        nodes = 1.0 / (
            (1.0 / min_x - 1.0 / max_x)
            * np.random.rand(2 * num_samples, 1)
            + 1.0 / max_x
        )

        values = np.random.normal(
            loc=np.sin(1.0 / nodes),
            scale=0.5 * nodes**2,
            size=(2 * num_samples, 1),
        )

        x_train = np.asarray(
            nodes[:num_samples],
            dtype=np.float32,
        )
        y_train = np.asarray(
            values[:num_samples],
            dtype=np.float32,
        )
        x_test = np.asarray(
            nodes[num_samples:],
            dtype=np.float32,
        )
        y_test = np.asarray(
            values[num_samples:],
            dtype=np.float32,
        )

        x_train_tensor = torch.from_numpy(x_train)
        y_train_tensor = torch.from_numpy(y_train)
        x_test_tensor = torch.from_numpy(x_test)
        y_test_tensor = torch.from_numpy(y_test)

        dataset = torch.utils.data.TensorDataset(
            x_train_tensor,
            y_train_tensor,
        )

        generator = torch.Generator()
        generator.manual_seed(seed)

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )

        # -----------------------------------------------------------
        # Create the traditional MLP
        # -----------------------------------------------------------
        model = cls(
            hidden_width=hidden_width,
            hidden_layers=hidden_layers,
            activation=activation,
            output_size=1,
        ).to(device)

        num_trainable_params = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

        print("Creating traditional MLP")
        print(f"Number of trainable parameters: {num_trainable_params}")

        criterion = torch.nn.SmoothL1Loss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
        )

        # -----------------------------------------------------------
        # Dense curve for visualizing the learned function
        # -----------------------------------------------------------
        x_curve = np.linspace(
            min_x,
            max_x,
            1000,
            dtype=np.float32,
        )

        x_curve_tensor = torch.from_numpy(
            x_curve[:, None]
        ).to(device)

        y_reference = np.sin(1.0 / x_curve)

        prediction_frames = []
        frame_epochs = []
        loss_history = []
        mse_history = []

        def record_frame(epoch, loss_value, mse_value):
            model.eval()

            with torch.no_grad():
                prediction = (
                    model(x_curve_tensor)
                    .detach()
                    .cpu()
                    .numpy()
                    .flatten()
                )

            prediction_frames.append(prediction)
            frame_epochs.append(epoch)
            loss_history.append(loss_value)
            mse_history.append(mse_value)

        # Initial untrained frame.
        record_frame(
            epoch=0,
            loss_value=np.nan,
            mse_value=np.nan,
        )

        # -----------------------------------------------------------
        # Train the MLP
        # -----------------------------------------------------------
        print("Start training")

        for epoch in range(1, num_epochs + 1):
            model.train()

            epoch_loss = 0.0
            number_of_batches = 0

            for x_batch, y_batch in data_loader:
                inputs = x_batch.to(device)
                targets = y_batch.to(device)

                optimizer.zero_grad(set_to_none=True)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                number_of_batches += 1

            average_loss = (
                epoch_loss / max(number_of_batches, 1)
            )

            # Test MSE.
            model.eval()

            squared_error = 0.0
            number_of_values = 0

            with torch.no_grad():
                for start in range(
                    0,
                    x_test_tensor.size(0),
                    batch_size,
                ):
                    inputs = x_test_tensor[
                        start:start + batch_size
                    ].to(device)

                    targets = y_test_tensor[
                        start:start + batch_size
                    ].to(device)

                    outputs = model(inputs)

                    squared_error += torch.sum(
                        torch.square(outputs - targets)
                    ).item()

                    number_of_values += targets.numel()

            test_mse = (
                squared_error / max(number_of_values, 1)
            )

            print(
                f"Epoch [{epoch:4d}/{num_epochs}], "
                f"Loss: {average_loss:.6f}, "
                f"Test MSE: {test_mse:.6f}"
            )

            if (
                epoch % frame_step == 0
                or epoch == num_epochs
            ):
                record_frame(
                    epoch=epoch,
                    loss_value=average_loss,
                    mse_value=test_mse,
                )

        # -----------------------------------------------------------
        # Stable limits for all frames
        # -----------------------------------------------------------
        all_predictions = np.concatenate(prediction_frames)

        all_y = np.concatenate(
            [
                y_train.flatten(),
                y_reference,
                all_predictions,
            ]
        )

        y_range = np.ptp(all_y)

        if y_range == 0:
            y_range = 1.0

        y_padding = 0.08 * y_range

        x_plot_min = 0.0
        x_plot_max = max_x
        y_plot_min = np.min(all_y) - y_padding
        y_plot_max = np.max(all_y) + y_padding

        # -----------------------------------------------------------
        # Animation
        # -----------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8.2, 5.2))

        def draw_background():
            ax.clear()

            ax.set_xlim(x_plot_min, x_plot_max)
            ax.set_ylim(y_plot_min, y_plot_max)

            ax.add_patch(
                Rectangle(
                    (x_plot_min, y_plot_min),
                    x_plot_max - x_plot_min,
                    y_plot_max - y_plot_min,
                    facecolor="#b9c9f2",
                    edgecolor="none",
                    alpha=0.55,
                    zorder=0,
                )
            )

            for x_value in np.linspace(
                x_plot_min,
                x_plot_max,
                11,
            ):
                ax.axvline(
                    x_value,
                    color="white",
                    linewidth=0.65,
                    alpha=0.9,
                    zorder=1,
                )

            for y_value in np.linspace(
                y_plot_min,
                y_plot_max,
                9,
            ):
                ax.axhline(
                    y_value,
                    color="white",
                    linewidth=0.65,
                    alpha=0.9,
                    zorder=1,
                )

            ax.scatter(
                x_train.flatten(),
                y_train.flatten(),
                s=11,
                color="#4f8662",
                edgecolors="none",
                alpha=0.62,
                label="Training samples",
                zorder=2,
            )

            ax.plot(
                x_curve,
                y_reference,
                color="#6c6c6c",
                linewidth=1.15,
                linestyle="--",
                alpha=0.8,
                label=r"$\sin(1/x)$",
                zorder=3,
            )

            arrow_style = {
                "arrowstyle": "->",
                "color": "black",
                "linewidth": 1.2,
                "shrinkA": 0,
                "shrinkB": 0,
            }

            ax.annotate(
                "",
                xy=(x_plot_max, 0.0),
                xytext=(x_plot_min, 0.0),
                arrowprops=arrow_style,
                zorder=5,
            )

            ax.annotate(
                "",
                xy=(0.0, y_plot_max),
                xytext=(0.0, y_plot_min),
                arrowprops=arrow_style,
                zorder=5,
            )

            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_visible(False)

        def update(frame_index):
            draw_background()

            learned_line, = ax.plot(
                x_curve,
                prediction_frames[frame_index],
                color="#b35a5a",
                linewidth=2.0,
                label="Learned function",
                zorder=4,
            )

            epoch = frame_epochs[frame_index]

            if epoch == 0:
                legend_title = (
                    "Traditional MLP\n"
                    f"Width = {hidden_width}\n"
                    f"Hidden layers = {hidden_layers}\n"
                    f"Parameters = {num_trainable_params}\n"
                    f"Epoch = 0/{num_epochs}\n"
                    "Loss = N/A\n"
                    "Test MSE = N/A"
                )
            else:
                legend_title = (
                    "Traditional MLP\n"
                    f"Width = {hidden_width}\n"
                    f"Hidden layers = {hidden_layers}\n"
                    f"Parameters = {num_trainable_params}\n"
                    f"Epoch = {epoch}/{num_epochs}\n"
                    f"Loss = {loss_history[frame_index]:.4f}\n"
                    f"Test MSE = {mse_history[frame_index]:.4f}"
                )

            legend = ax.legend(
                loc="lower right",
                frameon=True,
                fontsize=9,
                title=legend_title,
                title_fontsize=9,
            )

            legend.get_frame().set_facecolor("white")
            legend.get_frame().set_alpha(0.85)
            legend.get_frame().set_edgecolor("none")

            return [learned_line]

        animation = FuncAnimation(
            fig,
            update,
            frames=len(prediction_frames),
            interval=1000 / fps,
            blit=False,
            repeat=True,
        )

        animation.save(
            save_path,
            writer=PillowWriter(fps=fps),
            dpi=120,
        )

        print(f"GIF saved to {save_path}")

        # -------------------------------------------------------
        # Save the final frame
        # -------------------------------------------------------
        update(len(prediction_frames) - 1)

        png_path = save_path.replace(".gif", ".png")

        fig.savefig(
            png_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02,
        )

        # -------------------------------------------------------
        # Overlay a play icon onto the same PNG
        # -------------------------------------------------------
        from PIL import Image, ImageDraw

        img = Image.open(png_path).convert("RGBA")

        overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        w, h = img.size
        cx, cy = w // 2, h // 2

        radius = int(min(w, h) * 0.075)

        # Semi-transparent black circle
        draw.ellipse(
            (
                cx - radius,
                cy - radius,
                cx + radius,
                cy + radius,
            ),
            fill=(0, 0, 0, 110),
        )

        # White play triangle
        triangle = [
            (cx - radius * 0.28, cy - radius * 0.45),
            (cx - radius * 0.28, cy + radius * 0.45),
            (cx + radius * 0.45, cy),
        ]

        draw.polygon(
            triangle,
            fill=(255, 255, 255, 255),
        )

        # Overwrite the original PNG
        Image.alpha_composite(img, overlay).convert("RGB").save(png_path)

        print(f"Final frame with play icon saved to {png_path}")

        plt.show()


if __name__ == "__main__":
    # Triangulation.unit_test_1d()
    # Triangulation.unit_test_2d()
    # FiniteElement.unit_test_1()
    # FiniteElement.unit_test_2()
    # Net.unit_test_1d(dof=128, device="cuda")
    # Net.unit_test_2d(dof=512, device="cuda")
    MLP.unit_test_1d(hidden_width=128, hidden_layers=3, device="cuda")
