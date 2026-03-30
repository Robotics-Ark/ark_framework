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
        """Update an already-registered dynamic transform."""
        if not child:
            raise ValueError("Child frame name cannot be empty.")

        edge = self._edges.get(child)
        if edge is None:
            raise ValueError(f"Dynamic frame '{child}' is not registered.")
        if edge.is_static:
            raise ValueError(f"Frame '{child}' is static and cannot be updated.")

        self._edges[child] = Frame(
            parent=edge.parent,
            tf=tf,
            is_static=False,
        )

    def has_frame(self, frame: str) -> bool:
        """Return whether the frame is known to the forest."""
        return frame in self._frames

    def lookup_transform(self, target: str, source: str) -> RigidTransform:
        """Return T_target_source for two frames in the same tree."""
        if target not in self._frames:
            raise ValueError(f"Unknown target frame '{target}'.")
        if source not in self._frames:
            raise ValueError(f"Unknown source frame '{source}'.")

        target_root, tf_root_target = self._pose_from_root(target)
        source_root, tf_root_source = self._pose_from_root(source)
        if target_root != source_root:
            raise ValueError(
                f"Frames '{target}' and '{source}' are in disconnected trees "
                f"('{target_root}' and '{source_root}')."
            )

        return tf_root_target.inv() * tf_root_source

    def world_pose(self, frame: str) -> RigidTransform:
        """Return T_world_frame for a frame connected to the world tree."""
        try:
            return self.lookup_transform(WORLD_FRAME_NAME, frame)
        except ValueError as exc:
            if frame in self._frames:
                root, _ = self._pose_from_root(frame)
                if root != WORLD_FRAME_NAME:
                    raise ValueError(
                        f"Frame '{frame}' is not connected to '{WORLD_FRAME_NAME}'."
                    ) from exc
            raise

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
            raise ValueError(
                f"Cannot register '{WORLD_FRAME_NAME}' as a child frame."
            )
        if parent == child:
            raise ValueError(f"Frame '{child}' cannot have itself as a parent.")
        if child in self._edges:
            current_kind = "static" if self._edges[child].is_static else "dynamic"
            raise ValueError(
                f"Frame '{child}' is already registered as a {current_kind} frame."
            )
        if self._introduces_cycle(parent, child):
            raise ValueError(
                f"Registering '{child}' under '{parent}' would create a cycle."
            )

        self._frames.add(parent)
        self._frames.add(child)
        self._edges[child] = Frame(
            parent=parent,
            tf=tf,
            is_static=is_static,
        )

    def _pose_from_root(self, frame: str) -> tuple[str, RigidTransform]:
        current = frame
        tf_root_frame = RigidTransform.identity()
        visited: list[str] = []

        while True:
            if current in visited:
                cycle = " -> ".join([*visited, current])
                raise ValueError(f"Detected cycle in frames: {cycle}")
            visited.append(current)

            edge = self._edges.get(current)
            if edge is None:
                return current, tf_root_frame

            tf_root_frame = edge.tf * tf_root_frame
            current = edge.parent

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
                attrs = [
                    'shape="box"',
                    'fillcolor="palegreen3"' if frame in leaves else (
                        'fillcolor="indianred1"'
                        if edge.is_static
                        else 'fillcolor="lightskyblue"'
                    ),
                    'color="darkgreen"' if frame in leaves else (
                        'color="firebrick3"'
                        if edge.is_static
                        else 'color="royalblue3"'
                    ),
                ]
                if not edge.is_static:
                    attrs.append('style="filled,rounded"')

            lines.append(
                f'  "{self._escape_label(frame)}" [{", ".join(attrs)}];'
            )

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
