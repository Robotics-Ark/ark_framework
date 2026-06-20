from pathlib import Path
import shutil
import subprocess
from dataclasses import dataclass

from scipy.spatial.transform import RigidTransform

WORLD_FRAME_NAME = "world"


@dataclass(frozen=True)
class Frame:
    parent: str
    tf: RigidTransform
    is_static: bool


class FrameForest:
    """Registry for static and dynamic transforms across disconnected trees."""

    def __init__(self):
        self._frames: set[str] = {WORLD_FRAME_NAME}
        self._edges: dict[str, Frame] = {}
        # Memoized (root, T_root_frame) for frames whose full ancestor chain is static.
        self._static_root_cache: dict[str, tuple[str, RigidTransform]] = {}

    def register_static_transform(
        self, parent: str, child: str, tf: RigidTransform
    ) -> None:
        """Register an immutable transform T_parent_child in the forest."""
        self._register_transform(parent, child, tf, is_static=True)

    def register_dynamic_transform(
        self, parent: str, child: str, tf: RigidTransform
    ) -> None:
        """Register a dynamic transform T_parent_child in the forest."""
        self._register_transform(parent, child, tf, is_static=False)

    def update_dynamic_transform(self, child: str, tf: RigidTransform) -> None:
        """Replace the transform for an existing dynamic frame."""
        if not child:
            raise ValueError("Child frame name cannot be empty.")
        edge = self._edges.get(child)
        if edge is None:
            raise ValueError(f"Dynamic frame '{child}' is not registered.")
        if edge.is_static:
            raise ValueError(f"Frame '{child}' is static and cannot be updated.")
        self._edges[child] = Frame(parent=edge.parent, tf=tf, is_static=False)

    def has_frame(self, frame: str) -> bool:
        """Return whether the frame is known to the forest."""
        return frame in self._frames

    def can_transform(self, target: str, source: str) -> bool:
        """Return True if lookup_transform(target, source) would succeed."""
        try:
            self.lookup_transform(target, source)
            return True
        except ValueError:
            return False

    def get_all_frames(self) -> set[str]:
        """Return all registered frame names."""
        return set(self._frames)

    def get_parent(self, frame: str) -> str | None:
        """Return the parent frame name, or None if frame is a root."""
        if frame not in self._frames:
            raise ValueError(f"Unknown frame '{frame}'.")
        edge = self._edges.get(frame)
        return edge.parent if edge is not None else None

    def get_children(self, frame: str) -> set[str]:
        """Return all immediate children of frame."""
        if frame not in self._frames:
            raise ValueError(f"Unknown frame '{frame}'.")
        return {child for child, edge in self._edges.items() if edge.parent == frame}

    def lookup_transform(self, target: str, source: str) -> RigidTransform:
        """Return T_target_source for two frames in the same tree."""
        if target not in self._frames:
            raise ValueError(f"Unknown target frame '{target}'.")
        if source not in self._frames:
            raise ValueError(f"Unknown source frame '{source}'.")

        # O(1) fast path: both frames have a cached all-static root transform.
        t_entry = self._static_root_cache.get(target)
        s_entry = self._static_root_cache.get(source)
        if t_entry is not None and s_entry is not None:
            root_t, T_root_target = t_entry
            root_s, T_root_source = s_entry
            if root_t != root_s:
                raise ValueError(
                    f"Frames '{target}' and '{source}' are in disconnected trees "
                    f"('{root_t}' and '{root_s}')."
                )
            return T_root_target.inv() * T_root_source

        # General case: LCA traversal — walk only to the nearest common ancestor.
        target_chain, target_root = self._build_ancestor_chain(target)

        current = source
        tf_current_source = RigidTransform.identity()  # T_current_source
        all_static_source = True
        visited: set[str] = set()

        while True:
            if current in target_chain:
                T_lca_target = target_chain[current]
                # Cache source's root transform when LCA is the root and path is all-static.
                if (
                    all_static_source
                    and source not in self._static_root_cache
                    and self._edges.get(current) is None
                ):
                    self._static_root_cache[source] = (current, tf_current_source)
                return T_lca_target.inv() * tf_current_source

            if current in visited:
                raise ValueError(f"Cycle detected in frames near '{current}'.")
            visited.add(current)

            edge = self._edges.get(current)
            if edge is None:
                raise ValueError(
                    f"Frames '{target}' and '{source}' are in disconnected trees "
                    f"('{target_root}' and '{current}')."
                )

            if not edge.is_static:
                all_static_source = False
            tf_current_source = edge.tf * tf_current_source
            current = edge.parent

    def world_pose(self, frame: str) -> RigidTransform:
        """Return T_world_frame for a frame connected to the world tree."""
        return self.lookup_transform(WORLD_FRAME_NAME, frame)

    def resolve_pose(
        self,
        parent: str,
        tf: RigidTransform,
        target: str = WORLD_FRAME_NAME,
    ) -> RigidTransform:
        """Resolve a parent-relative pose into the requested target frame."""
        return self.lookup_transform(target, parent) * tf

    def to_image(self, path: str | Path) -> None:
        """Render the frame forest to an image using Graphviz."""
        output_path = Path(path).expanduser()
        image_format = output_path.suffix.lstrip(".")
        if not image_format:
            raise ValueError("Image path must include a file extension.")

        executable = shutil.which("dot")
        if executable is None:
            raise RuntimeError("Graphviz executable 'dot' was not found in PATH.")

        try:
            subprocess.run(
                [executable, f"-T{image_format}", "-o", str(output_path)],
                input=self._dot_source(),
                text=True,
                capture_output=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip()
            if stderr:
                raise RuntimeError(f"Graphviz rendering failed: {stderr}") from exc
            raise RuntimeError("Graphviz rendering failed.") from exc

    def _register_transform(
        self,
        parent: str,
        child: str,
        tf: RigidTransform,
        *,
        is_static: bool,
    ) -> None:
        if not parent:
            raise ValueError("Parent frame name cannot be empty.")
        if not child:
            raise ValueError("Child frame name cannot be empty.")
        if child == WORLD_FRAME_NAME:
            raise ValueError(f"Cannot register '{WORLD_FRAME_NAME}' as a child frame.")
        if parent == child:
            raise ValueError(f"Frame '{child}' cannot have itself as a parent.")
        if child in self._edges:
            kind = "static" if self._edges[child].is_static else "dynamic"
            raise ValueError(
                f"Frame '{child}' is already registered as a {kind} frame."
            )
        if self._introduces_cycle(parent, child):
            raise ValueError(
                f"Registering '{child}' under '{parent}' would create a cycle."
            )
        self._frames.add(parent)
        self._frames.add(child)
        self._edges[child] = Frame(parent=parent, tf=tf, is_static=is_static)

    def _build_ancestor_chain(self, frame: str) -> tuple[dict[str, RigidTransform], str]:
        """Walk from frame toward root, return ({ancestor: T_ancestor_frame}, root_name).

        Short-circuits at cached static ancestors. Populates the static root cache
        when the full ancestor path is all-static.
        """
        chain: dict[str, RigidTransform] = {}
        current = frame
        tf = RigidTransform.identity()  # T_current_frame
        all_static = True
        root = frame
        visited: set[str] = set()

        while True:
            if current in visited:
                raise ValueError(f"Cycle detected in frames near '{current}'.")
            visited.add(current)
            chain[current] = tf

            # Short-circuit at a cached static ancestor (skip remaining walk to root).
            if current != frame and current in self._static_root_cache:
                cached_root, T_root_current = self._static_root_cache[current]
                chain[cached_root] = T_root_current * tf  # T_root_frame
                root = cached_root
                if all_static:
                    self._static_root_cache[frame] = (cached_root, T_root_current * tf)
                break

            edge = self._edges.get(current)
            if edge is None:
                root = current
                if all_static:
                    self._static_root_cache[frame] = (current, tf)
                break

            if not edge.is_static:
                all_static = False
            tf = edge.tf * tf
            current = edge.parent

        return chain, root

    def _introduces_cycle(self, parent: str, child: str) -> bool:
        current = parent
        visited: set[str] = set()
        while True:
            if current == child:
                return True
            if current in visited:
                return True
            visited.add(current)
            edge = self._edges.get(current)
            if edge is None:
                return False
            current = edge.parent

    def _dot_source(self) -> str:
        roots = {frame for frame in self._frames if frame not in self._edges}
        parents = {edge.parent for edge in self._edges.values()}
        leaves = {frame for frame in self._frames if frame not in parents}
        lines = [
            "digraph FrameForest {",
            '  rankdir="RL";',
            '  node [shape="box", style="filled", penwidth="1.5"];',
        ]

        for frame in sorted(self._frames):
            if frame == WORLD_FRAME_NAME:
                attrs = [
                    'shape="doublecircle"',
                    'fillcolor="black"',
                    'fontcolor="white"',
                    'color="black"',
                ]
            elif frame in roots:
                attrs = [
                    'shape="circle"',
                    'fillcolor="black"',
                    'fontcolor="white"',
                    'color="black"',
                ]
            else:
                edge = self._edges[frame]
                if not edge.is_static:
                    # dynamic frames: blue + rounded, leaf or not
                    attrs = [
                        'shape="box"',
                        'style="filled,rounded"',
                        'fillcolor="lightskyblue"',
                        'color="royalblue3"',
                    ]
                elif frame in leaves:
                    attrs = ['shape="box"', 'fillcolor="palegreen3"', 'color="darkgreen"']
                else:
                    attrs = ['shape="box"', 'fillcolor="indianred1"', 'color="firebrick3"']

            lines.append(f'  "{self._escape_label(frame)}" [{", ".join(attrs)}];')

        for child in sorted(self._edges):
            edge = self._edges[child]
            style = "solid" if edge.is_static else "dashed"
            color = "firebrick3" if edge.is_static else "royalblue3"
            lines.append(
                f'  "{self._escape_label(child)}" -> "{self._escape_label(edge.parent)}" '
                f'[style="{style}", color="{color}", penwidth="1.5"];'
            )

        lines.append("}")
        return "\n".join(lines)

    @staticmethod
    def _escape_label(label: str) -> str:
        return label.replace("\\", "\\\\").replace('"', '\\"')
