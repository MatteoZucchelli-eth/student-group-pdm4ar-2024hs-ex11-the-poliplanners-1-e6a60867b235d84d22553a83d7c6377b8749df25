import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import translate, rotate
from dg_commons import SE2Transform
from matplotlib.animation import FuncAnimation


class CollisionChecker:
    def __init__(self, portion_of_trajectory: float, my_name: str) -> None:
        self.geometries = {}
        self.portion_of_trajectory = portion_of_trajectory
        self.my_name = my_name
        self.orientation = None

    def add_other_geometry(self, name, other_geometry):
        self.geometries[name] = other_geometry

    def collision_checking(
        self,
        my_positions: list[SE2Transform],
        other_positions_dict: dict[str, list[SE2Transform]],
        orientation: float,
    ) -> dict[str, list[int]]:
        """
        This function checks if our current trajectory is in collision with other agents' trajectories.
        :param my_positions: List of tuples containing the current agent's positions
        :param other_positions: List of tuples containing the other agents' positions
        :param my_radius: Integer indicating the radius of our agent
        :param other_radius: Integer indicating the radius of the other agents
        :param portion_of_trajectory: Float indicating the portion of the trajectory to consider for the collision checking
        :return: List of integers indicating the time steps in which a collision occurs
        """
        self.orientation = orientation

        # Check on the lenght of the lists
        if len(my_positions) == 0 or all(
            int(len(my_positions) * self.portion_of_trajectory) != len(other_positions)
            for other_positions in other_positions_dict.values()
        ):
            return {}

        # Create the shapely points buffered for our car and the other agent
        my_points = []
        for my_position in my_positions[0 : int(len(my_positions) * self.portion_of_trajectory)]:
            my_points.append(self.compute_car_position(my_position, self.my_name))

        collision_indexes = {}
        other_points_dict = {}
        for other_name, other_positions in other_positions_dict.items():
            other_points_dict[other_name] = []
            for other_position in other_positions:
                other_points_dict[other_name].append(self.compute_car_position(other_position, other_name))

            # Check if corrisponding points are in collision
            collision_indexes[other_name] = []
            for my_point, other_point in zip(my_points, other_points_dict[other_name]):
                if my_point.intersects(other_point):
                    collision_indexes[other_name].append(my_points.index(my_point))

        # self.plot_trajectories(my_points, other_points_dict)

        return collision_indexes

    def compute_car_position(self, position: SE2Transform, name: str) -> Polygon:
        """
        This function computes the car position given the current position and the car geometry.
        :param position: Tuple containing the current position of the car
        :return: Shapely polygon representing the car around the given position
        """
        x, y = position.p
        theta = position.theta
        geometry = self.geometries[name]

        if isinstance(geometry, Polygon):
            # Translate the car geometry to the current position
            current_center = geometry.centroid
            translation = (x - current_center.x, y - current_center.y)
            angle = self.orientation - theta
            translated_geometry = rotate(
                translate(geometry, xoff=translation[0], yoff=translation[1]), angle, use_radians=True
            )

        # Consider the case of my car
        else:
            # Rectangle vertices
            half_width, half_height = geometry.length / 2 * 1.5, geometry.w_half * 1.5
            vertices = [
                (-half_width, -half_height),  # In basso a sinistra
                (half_width, -half_height),  # In basso a destra
                (half_width, half_height),  # In alto a destra
                (-half_width, half_height),  # In alto a sinistra
            ]

            # Rotate the vertices
            rotated_vertices = []
            for dx, dy in vertices:
                x_rot = x + (dx * np.cos(theta) - dy * np.sin(theta))
                y_rot = y + (dx * np.sin(theta) + dy * np.cos(theta))
                rotated_vertices.append((x_rot, y_rot))

            translated_geometry = Polygon(rotated_vertices)

        return translated_geometry

    def plot_trajectories(self, my_points: list[Polygon], other_points: dict[str, list[Polygon]]):
        """
        This function plots all the polygons that compose the trajectories of my_points and other_points.
        :param my_points: List of shapely polygons representing the current agent's trajectory
        :param other_points: Dictionary with keys as agent names and values as lists of shapely polygons representing other agents' trajectories
        """
        plt.figure()

        color = "red"
        # Plot my_points
        for poly in my_points:
            x, y = poly.exterior.xy
            plt.fill(x, y, alpha=0.5, fc=color, ec="black", label=self.my_name)

        colors = ["blue", "green", "orange", "purple", "red"]
        for i, (group_name, polygons) in enumerate(other_points.items()):
            color = colors[i % len(colors)]  # Usa un colore ciclico
            for poly in polygons:
                x, y = poly.exterior.xy  # Estrai le coordinate del contorno
                plt.fill(x, y, alpha=0.5, fc=color, ec="black", label=group_name if poly == polygons[0] else "")

        # Configura il grafico
        plt.gca().set_aspect("equal")  # Mantieni le proporzioni corrette
        plt.title("Cars trajectories")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.savefig("trajectories_collision.png")

        polygons_dict = other_points
        polygons_dict[self.my_name] = my_points

        # Numero totale di frame (uguale alla lunghezza delle liste dei poligoni)
        num_frames = len(next(iter(polygons_dict.values())))

        # Funzione per aggiornare il frame
        def update(frame):
            plt.cla()  # Cancella il grafico precedente
            plt.title(f"Istante: {frame}")
            plt.xlabel("Asse X")
            plt.ylabel("Asse Y")
            plt.grid(True)
            plt.axis("equal")

            # Disegna i poligoni al frame corrente
            for name, polygons in polygons_dict.items():
                poly = polygons[frame]
                x, y = poly.exterior.xy
                plt.fill(x, y, alpha=0.5, label=name)

            # Evita ripetizioni di etichette nella legenda
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())

        # Crea l'animazione
        fig = plt.figure(figsize=(8, 8))
        animation = FuncAnimation(fig, update, frames=num_frames, interval=500)  # type: ignore # Intervallo in millisecondi

        # Salva il video (richiede ffmpeg o imagemagick)
        animation.save("polygons_animation.mp4", fps=2, extra_args=["-vcodec", "libx264"])
