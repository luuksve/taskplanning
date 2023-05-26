"""Module to plan tasks for mechanics"""


from typing import Dict, List, Tuple

import gurobipy as gp
from pydantic import validate_arguments


class TaskPlanningModel:
    """Class containing all data and the model to be solved
    Allows for setting the data, creating the model and solving it

    Args:
        airplanes (List[str]): List of string airplane indices.
        mechanics (List[str]): List of string mechanic indices.
        tasks (List[str]): List of string task indices.
        tasks_per_airplane (Dict[str, Tuple[str, ...]]): Dict of string airplane
            indices (keys) and a tuple of variable length of string task indices.
        qualifications (Dict[Tuple[str, str], float]): Dict of two-dimensional
            string,string type keys that indicate mechanics and tasks respectively and float values.
            The floats should be either 0.0 or 1.0, acting as binary qualification indicators.
        duration_per_task (Dict[str, float]): Dict of string task indicators and float duration values.
    """

    @validate_arguments
    def __init__(
        self,
        airplanes: List[str],
        mechanics: List[str],
        tasks: List[str],
        tasks_per_airplane: Dict[str, Tuple[str, ...]],
        qualifications: Dict[Tuple[str, str], float],
        duration_per_task: Dict[str, float],
    ):
        self.airplanes = airplanes
        self.mechanics = mechanics
        self.tasks = tasks
        self.tasks_per_airplane = tasks_per_airplane
        self.qualifications = qualifications
        self.duration_per_task = duration_per_task
        self.model = gp.Model("TaskPlanningModel")

    def create_model(self):
        """Creates the task planning model"""

        # Define decision variables
        x = self.model.addVars(
            self.mechanics, self.tasks, name="x", vtype=gp.GRB.BINARY
        )
        y = self.model.addVars(
            self.tasks, name="y", vtype=gp.GRB.BINARY
        )
        z = self.model.addVars(
            self.airplanes, name="z", vtype=gp.GRB.BINARY
        )

        # Update model so constraints can use variables
        self.model.update()

        # Add constraints
        self.model.addConstrs(
            (z[a] <= y[t] for a in self.airplanes for t in self.tasks_per_airplane[a]),
            "c0",
        )
        self.model.addConstrs(
            (y[t] <= gp.quicksum(x[m, t] for m in self.mechanics) for t in self.tasks),
            "c1",
        )
        self.model.addConstrs(
            (
                gp.quicksum(x[m, t] * self.duration_per_task[t] for t in self.tasks)
                <= 8
                for m in self.mechanics
            ),
            "c2",
        )
        self.model.addConstrs(
            (
                x[m, t] <= self.qualifications[m, t]
                for m in self.mechanics
                for t in self.tasks
            ),
            "c3",
        )

        # Set objective function
        self.model.setObjective(
            gp.quicksum(z[a] for a in self.airplanes), gp.GRB.MAXIMIZE
        )

        # Update model
        self.model.update()

    def solve_model(self):
        """Solves the model and returns the solution"""

        # Optimize model
        self.model.optimize()

        # Return instance of solution class
        x = [v for v in self.model.getVars() if v.startswith("x")]
        return Solution(status=self.model.status, x=x, obj_val=self.model.ObjVal)


class Solution:
    """Solution class

    Args:
        status (int): Model status
        x (List[gp.Var]): Main decision variables
        obj_val (float): Objective value
    """

    def __init__(self, status: int, x: List[gp.Var], obj_val: float):
        self.status = status
        self.x = x
        self.obj_val = obj_val

    def __str__(self):
        if self.status != gp.GRB.Status.OPTIMAL:
            return "Model status is not optimal"
        return_string = "Model status is optimal\n"

        return_string += f"Number of planes ready: {str(self.obj_val)}\n"
        for key in self.x:
            if self.x[key].x == 1.0:
                return_string += f"Mechanic {str(key[0])} works on task {str(key[1])}\n"
        return return_string


if __name__ == "__main__":
    model_inputs = {
        "airplanes": ["ap1", "ap2"],
        "mechanics": ["john", "jack"],
        "tasks": ["a", "b", "c"],
        "tasks_per_airplane": {"ap1": ("a", "b"), "ap2": ("b", "c")},
        "qualifications": {
            ("john", "a"): 1.0,
            ("john", "b"): 0.0,
            ("john", "c"): 1.0,
            ("jack", "a"): 0.0,
            ("jack", "b"): 1.0,
            ("jack", "c"): 1.0,
        },
        "duration_per_task": {"a": 2.0, "b": 6.0, "c": 4.0},
    }
    model = TaskPlanningModel(**model_inputs)
    model.create_model()
    print(model.solve_model)
