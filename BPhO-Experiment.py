import csv
from typing import Any, Callable, NamedTuple, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

ball_data = "ball_data.csv"
run_data = "run_data.csv"


class BallMassDiameter(NamedTuple):
    mass: float
    mean_diameter: float


class SingleRunData(NamedTuple):
    velocity: float
    diameter1: float
    diameter2: float
    ejecta: str
    level_height: float
    ejecta_height: Optional[float]
    lowest_height: float


class BallRunsData(NamedTuple):
    ball_mass_diameter: BallMassDiameter
    run_data: list[SingleRunData]


def ln(function: Callable[[BallMassDiameter, SingleRunData], float]):
    def wrapper(ball_mass_diameter: BallMassDiameter, run_data: SingleRunData) -> float:
        return np.log(function(ball_mass_diameter, run_data))

    return wrapper


@ln
def hole_diameter(_: BallMassDiameter, run_data: SingleRunData) -> float:
    return (run_data.diameter1 + run_data.diameter2) / 2


def hole_diameter_minus_ball_diameter(
    ball_mass_diameter: BallMassDiameter, run_data: SingleRunData
) -> float:
    return (
        (run_data.diameter1 + run_data.diameter2) / 2
    ) - ball_mass_diameter.mean_diameter


@ln
def kinetic_energy(
    ball_mass_diameter: BallMassDiameter, run_data: SingleRunData
) -> float:
    return run_data.velocity**2 * 1 / 2 * ball_mass_diameter.mass


@ln
def crater_depth(_: BallMassDiameter, run_data: SingleRunData) -> float:
    return run_data.level_height - run_data.lowest_height


def skip(ball_name: str, run: SingleRunData) -> bool:
    return ball_name == "Medium Metal" and run.diameter1 == 30 and run.diameter2 == 35


def plot_two_function_of_ball_against_each_other(
    ax: plt.Axes,
    runs_data: dict[str, BallRunsData],
    x_axis_function: Callable[[BallMassDiameter, SingleRunData], float],
    y_axis_function: Callable[[BallMassDiameter, SingleRunData], float],
    x_axis_label: str,
    y_axis_label: str,
    title: str,
) -> None:
    for ball_name, run_data in runs_data.items():
        ball_mass_diameter = run_data.ball_mass_diameter
        lists_to_plot = (
            [
                x_axis_function(ball_mass_diameter, run)
                for run in run_data.run_data
                if not skip(ball_name, run)
            ],
            [
                y_axis_function(ball_mass_diameter, run)
                for run in run_data.run_data
                if not skip(ball_name, run)
            ],
        )
        ax.plot(*lists_to_plot, "o", label=ball_name)
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)
    ax.set_title(title)
    ax.legend()


def linear_regression(
    runs_data: dict[str, BallRunsData],
    x_axis_function: Callable[[BallMassDiameter, SingleRunData], float],
    y_axis_function: Callable[[BallMassDiameter, SingleRunData], float],
) -> list[tuple[LinearRegression, float]]:
    ans: list[tuple[LinearRegression, float]] = []
    for ball_name, run_data in runs_data.items():
        ball_mass_diameter = run_data.ball_mass_diameter
        x = np.array(
            [
                x_axis_function(ball_mass_diameter, run)
                for run in run_data.run_data
                if not skip(ball_name, run)
            ]
        ).reshape((-1, 1))
        y = np.array(
            [
                y_axis_function(ball_mass_diameter, run)
                for run in run_data.run_data
                if not skip(ball_name, run)
            ]
        )
        model = LinearRegression().fit(x, y)
        ans.append((model, model.score(x, y)))
    return ans


def linear_regression_on_all_data(
    runs_data: dict[str, BallRunsData],
    x_axis_function: Callable[[BallMassDiameter, SingleRunData], float],
    y_axis_function: Callable[[BallMassDiameter, SingleRunData], float],
) -> tuple[LinearRegression, float]:
    x: list[float] = []
    y: list[float] = []
    for ball_name, run_data in runs_data.items():
        ball_mass_diameter = run_data.ball_mass_diameter
        for run in run_data.run_data:
            if skip(ball_name, run):
                continue
            x.append(x_axis_function(ball_mass_diameter, run))
            y.append(y_axis_function(ball_mass_diameter, run))
    x = np.array(x).reshape((-1, 1))
    y = np.array(y)
    model = LinearRegression().fit(x, y)
    return model, model.score(x, y)


def read_in_data() -> dict[str, BallRunsData]:
    data: dict[str, BallRunsData] = {}
    with open(ball_data, "r") as f:
        for row in csv.DictReader(f):
            name = row["ï»¿Ball"]
            mass = float(row["Mass / g"])
            mean_diameter = (
                float(row["Diameter1/ mm"])
                + float(row["Diameter2/ mm"])
                + float(row["Diameter3/ mm"])
            ) / 3
            data[name] = BallRunsData(
                BallMassDiameter(mass=mass, mean_diameter=mean_diameter), []
            )

    with open(run_data, "r") as f:
        for row in csv.DictReader(f):
            data[row["ï»¿Ball"]].run_data.append(
                SingleRunData(
                    velocity=float(row["Velocity / ms^-1"]),
                    diameter1=float(
                        row["Impact crater diameter (edge of cocoa powder) / mm"]
                    ),
                    diameter2=float(row["Impact crater other way / mm"]),
                    ejecta=row["Presence of ejecta?"],
                    level_height=float(row["Level height / mm"]),
                    ejecta_height=(
                        None
                        if row["Ejecta height / mm"] == "N/A"
                        else float(row["Ejecta height / mm"])
                    ),
                    lowest_height=float(row["Lowest height / mm"]),
                )
            )
    return data


def main():
    data = read_in_data()
    fig, axes = plt.subplots(1, 2)  # type: ignore
    plot_two_function_of_ball_against_each_other(
        axes[0],
        data,
        kinetic_energy,
        hole_diameter,
        "ln(Kinetic Energy / mJ)",
        "ln(Mean Hole Diameter / mm)",
        "Mean diameter of crater against kinetic energy (log-log plot)",
    )
    ke_diameter_regressions = linear_regression(data, kinetic_energy, hole_diameter)
    for ball_name in data.keys():
        print(ball_name)
    print(*(model[0].coef_ for model in ke_diameter_regressions))
    print(*(model[1] for model in ke_diameter_regressions))
    all_regression = linear_regression_on_all_data(data, kinetic_energy, hole_diameter)
    print(all_regression[0].coef_)
    print(all_regression[1])
    plot_two_function_of_ball_against_each_other(
        axes[1],
        data,
        kinetic_energy,
        crater_depth,
        "ln(Kinetic Energy / mJ)",
        "ln(Hole Depth / mm)",
        "Depth of crater against kinetic energy (log-log plot)",
    )
    ke_depth_regressions = linear_regression(data, kinetic_energy, crater_depth)
    for ball_name in data.keys():
        print(ball_name)
    print(*(model[0].coef_ for model in ke_depth_regressions))
    print(*(model[1] for model in ke_depth_regressions))
    all_regression = linear_regression_on_all_data(data, kinetic_energy, crater_depth)
    print(all_regression[0].coef_)
    print(all_regression[1])
    fig.set_figwidth(15)
    plt.show()


if __name__ == "__main__":
    main()
