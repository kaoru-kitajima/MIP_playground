import os
from abc import ABCMeta, abstractmethod,ABC
import pulp

class AbstractMILP(ABC):
    def __init__(self):
        self.prob = pulp.LpProblem()
        self.objectives = []

    def set_settings(self, settings):
        self.settings = settings
    def get_settings(self):
        return self.settings
    
    def write_mps(self):
        self.prob.writeMPS("{}.mps".format(self.problem_name))

    def write_lp(self):
        self.prob.writeLP("{}.lp".format(self.problem_name))


    def set_solver(self, solver:str, timelimit:int=600, msg=False):
        self.solver = pulp.PULP_CBC_CMD(timeLimit = timelimit)
        solver_list = pulp.listSolvers(onlyAvailable=True)
        print(solver_list) # list up available solvers
        if solver not in solver_list :
            solvers = ', '.join(solver_list)
            raise Exception(f"The specified solver is unavailable. selectSolver:{solver}, availableSolver:{solvers}")
        else :
            if solver == "PULP_CBC_CMD" :
                self.solver = pulp.PULP_CBC_CMD(timeLimit = timelimit, msg=msg)
            if solver == "GUROBI" :
                self.solver = pulp.GUROBI(timeLimit = timelimit, msg=msg)
            if solver == "CPLEX" :
                self.solver = pulp.CPLEX(timeLimit = timelimit, msg=msg)
            if solver == "GLPK_CMD" :
                self.solver = pulp.GLPK_CMD(timeLimit = timelimit, msg=msg)
            if solver == "XPRESS" :
                self.solver = pulp.XPRESS(timeLimit = timelimit, msg=msg)
            if solver == "SCIP_CMD" :
                self.solver = pulp.SCIP_CMD(timeLimit = timelimit, msg=True) # Maybe no -q option available in SCIP_CMD? 
            if solver == "FSCIP_CMD" :
                self.solver = pulp.FSCIP_CMD(timeLimit = timelimit, msg=True) # No -q option available in FSCIP_CMD
            if solver == "COIN_CMD" :
                self.solver = pulp.COIN_CMD(timeLimit = timelimit, threads=os.cpu_count(), msg=msg) # parallel-processing by CBC is available if built with proper args. otherwise, threads arg is ignored.
            if solver == "XPRESS_PY":
                self.solver = pulp.XPRESS_PY(timeLimit = timelimit, msg=msg)
            if solver == "SCIP_PY":
                self.solver = pulp.SCIP_PY(timeLimit = timelimit, msg=msg)
        self.prob.setSolver(self.solver)

    def solve(self):
        self.declare_variables()
        self.declare_objectives()
        self.declare_constraints()
        self.statuses = self.sequential_solve()

    def sequential_solve(self):
        statuses = []
        for i, obj in enumerate(self.objectives):
            assert isinstance(obj, MilpObjective), f"obj must be MilpObjective, but got {type(obj)}"
            self.prob.objective = obj.val
            self.prob.sense = obj.sense
            self.prob.objective.name = obj.name
            rel = obj.relativeTol
            absol = obj.absoluteTol
            status = self.prob.solve()
            statuses.append(status)
            print(f"{obj.name} is {pulp.LpStatus[status]}.")
            if self.prob.sense == pulp.const.LpMinimize:
                self.prob += obj.val <= pulp.value(obj.val) * rel + absol, f"Sequence_Objective_{i}"
            elif self.prob.sense == pulp.const.LpMaximize:
                self.prob += obj.val >= pulp.value(obj.val) * rel + absol, f"Sequence_Objective_{i}"
        return statuses

    def add_objective(self, obj, sense, name, absoluteTol=0, relativeTol=1):
        self.objectives.append(MilpObjective(val=obj, sense=sense, name=name, absoluteTol=absoluteTol, relativeTol=relativeTol))
        
    @abstractmethod
    def declare_variables(self):
        pass

    @abstractmethod
    def declare_objectives(self):
        pass

    @abstractmethod
    def declare_constraints(self):
        pass

class MilpObjective:
    def __init__(self, val:pulp.LpAffineExpression, name:str, sense:int, absoluteTol=0, relativeTol=1):
        assert isinstance(val, pulp.LpAffineExpression), f"val must be LpAffineExpression, but got {type(val)}"
        assert isinstance(name, str), f"name must be str, but got {type(name)}"
        assert sense in (pulp.const.LpMinimize,pulp.const.LpMaximize), f"sense must be LpMinimize or LpMaximize, but got {type(sense)}"
        assert isinstance(absoluteTol, int) or isinstance(absoluteTol, float), f"absoluteTol must be int or float, but got {type(absoluteTol)}"
        assert isinstance(relativeTol, int) or isinstance(relativeTol, float), f"relativeTol must be int or float, but got {type(relativeTol)}"
        self.val = val
        self.name = name
        self.sense = sense
        self.absoluteTol = absoluteTol
        self.relativeTol = relativeTol
        
