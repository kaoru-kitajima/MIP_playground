import numpy as np
import sys
import os
import pulp
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import pulp
import relax_problem
from abstract_milp import AbstractMILP
class SimpleCAPP(AbstractMILP):
    """Simple Computer-Aided Process Planning (CAPP) class"""
    def __init__(self, products, product_tasks, main_component_sets, aux_component_sets, initial_costs, aux_initial_costs, labor_costs, aux_labor_costs, operator_counts, task_times, aux_requirements, amount, time_period, stations, changeover_times, import_times, export_times, transport_times, transport_can_be_bottleneck, problem_name="Production_Line_Optimization"):
        super().__init__()
        self.products = products
        self.product_tasks = product_tasks
        self.main_component_sets = main_component_sets
        self.aux_component_sets = aux_component_sets
        self.stations = stations
        self.initial_costs = initial_costs
        self.aux_initial_costs = aux_initial_costs
        self.labor_costs = labor_costs
        self.aux_labor_costs = aux_labor_costs
        self.aux_requirements = aux_requirements
        self.operator_counts = operator_counts
        self.task_times = task_times
        self.amount = amount
        self.time_period = time_period
        self.changeover_times = changeover_times
        self.import_times = import_times
        self.export_times = export_times
        self.transport_times = transport_times
        self.transport_can_be_bottleneck = transport_can_be_bottleneck
        self.problem_name = problem_name
        # Define the problem
        self.prob = pulp.LpProblem(self.problem_name, pulp.LpMinimize)
        self.M = 1000000000 # a large number

    def declare_variables(self):
        # The variables dict will hold the decision variables
        self.assigned = pulp.LpVariable.dicts("Assignment", [(p,t,c,s) for p in self.products for t in self.product_tasks[p] for c in self.main_component_sets for s in self.stations], cat=pulp.LpBinary)
        for p in self.products:
            for t in self.product_tasks[p]:
                for c in self.main_component_sets:
                    for s in self.stations:
                        self.assigned[p,t,c,s].setInitialValue(0)
        self.equipped = pulp.LpVariable.dicts("Equipment", [(c,s) for c in self.main_component_sets for s in self.stations], cat=pulp.LpBinary)
        for c in self.main_component_sets:
            for s in self.stations:
                self.equipped[c,s].setInitialValue(0)
        self.aux_equipped = pulp.LpVariable.dicts("AuxiliaryEquipment", [(ac,s) for ac in self.aux_component_sets for s in self.stations], cat=pulp.LpBinary)
        for ac in self.aux_component_sets:
            for s in self.stations:
                self.aux_equipped[ac,s].setInitialValue(0)
        self.assigned_same_station = pulp.LpVariable.dicts("AssignmentSameStation", [(p,t_1,t_2,c,s) for p in self.products for t_1, t_2 in zip(self.product_tasks[p], self.product_tasks[p][1:]) for c in self.main_component_sets for s in self.stations], cat=pulp.LpBinary)
        self.cycletime = pulp.LpVariable.dicts("StationCycleTime", [(p,s) for p in self.products for s in self.stations], lowBound=0, cat=pulp.LpContinuous)
        # operator worktime for each component set in each station for each product. effected by settings[labor_cost_on_exact_cycletime]. 
        self.operator_paid_worktime = pulp.LpVariable.dicts("WorkerPaidWorkTime", [(p,c,s) for p in self.products for c in self.main_component_sets for s in self.stations], lowBound=0, cat=pulp.LpContinuous)
        self.aux_operator_paid_worktime = pulp.LpVariable.dicts("AuxWorkerPaidWorkTime", [(p,ac,s) for p in self.products for ac in self.aux_component_sets for s in self.stations], lowBound=0, cat=pulp.LpContinuous)
        self.overall_cycle_time = pulp.LpVariable.dicts("OverallCycleTime", [p for p in self.products], lowBound=0, cat=pulp.LpContinuous)
        # trans is 1 if the task t_1 and next task t_2 are assigned to the different stations s_1 and s_2. ! jobshopがなければtransとassigned_same_stationは片方だけでいいと思われる。!
        self.trans = pulp.LpVariable.dicts("Transportation", [(p,t_1,t_2,s_1,s_2) for p in self.products for t_1, t_2 in zip(self.product_tasks[p], self.product_tasks[p][1:]) for s_1 in self.stations for s_2 in self.stations], cat=pulp.LpBinary)
    
    def __total_cost(self):
        return self.__total_initial_cost() + self.__total_labor_cost()
        
    def __total_initial_cost(self):
        if self.settings['depreciation']:
            return pulp.lpSum([self.initial_costs[c]*self.time_period/self.settings['depreciation_timespan']*self.equipped[c,s] for c in self.main_component_sets for s in self.stations]) \
              + pulp.lpSum([self.aux_initial_costs[ac]*self.time_period/self.settings['depreciation_timespan']*self.aux_equipped[ac,s] for ac in self.aux_component_sets for s in self.stations])
        else:
            return pulp.lpSum([self.initial_costs[c]*self.equipped[c,s] for c in self.main_component_sets for s in self.stations]) \
              + pulp.lpSum([self.aux_initial_costs[ac]*self.aux_equipped[ac,s] for ac in self.aux_component_sets for s in self.stations])
        
    def __total_labor_cost(self):
        return pulp.lpSum([self.labor_costs[c]*self.operator_paid_worktime[p,c,s] for p in self.products for c in self.main_component_sets for s in self.stations]) \
              + pulp.lpSum([self.aux_labor_costs[ac]*self.aux_operator_paid_worktime[p,ac,s] for p in self.products for ac in self.aux_component_sets for s in self.stations])
    
    def __total_production_time(self):
        # TODO: １つ流しの混流生産ならば、製品ごとのoverall_cycle_timeではなく、ステーションごとのcycletime*amountとするべき。ボトルネック工程が被らなければ、overall_cycle_timeよりも短い期間ですむ。bottleneck_stationをlpvariableとして設定するべき。
        # TODO: バッチ生産ならば、製品ごとのoverall_timeであるべき。
        return pulp.lpSum([self.amount[p] * self.overall_cycle_time[p] for p in self.products])
    
    def __total_transport_time(self):
        # transport time from station 0 to first station of each product + transport time between each task-assigned stations
        return pulp.lpSum(self.amount[p] * pulp.lpSum(self.transport_times[self.stations[0], s] * self.assigned[p,self.product_tasks[p][0],c,s] for c in self.main_component_sets for s in self.stations) for p in self.products) \
         + pulp.lpSum([self.amount[p] * pulp.lpSum([self.transport_times[s_1,s_2]*self.trans[p,t_1,t_2,s_1,s_2] for t_1, t_2 in zip(self.product_tasks[p], self.product_tasks[p][1:]) for s_1 in self.stations for s_2 in self.stations]) for p in self.products])
    
    def declare_objectives(self):
        # 1st Objective function: Minimize the total cost
        self.add_objective(obj=self.__total_cost(), sense=pulp.LpMinimize, name="TotalCost")
        # 2nd Objective function: Minimize the total production time
        self.add_objective(obj=self.__total_production_time(), sense=pulp.LpMinimize, name="TotalProductionTime", relativeTol=1.1) # 10% tolerance for total production time constraint
        # 3rd Objective function: Minimize the transport Time
        self.add_objective(obj=self.__total_transport_time(), sense=pulp.LpMinimize, name="TotalTransportTime")
    
    def declare_constraints(self):
        self.__task_assignment_constraint()
        self.__task_order_constraint()
        self.__cycletime_constraint()
        self.__operator_constraint()

    def __task_assignment_constraint(self):
        """Basic task-component_set-station assignment constraints.
        - Each task step requires single component set in single station. 
        - Each each station has 0 or only one component set. 
        - Task cannot be assigned to station if the station is not equipped with the component set."""
        # Each task step requires single component set in single station
        for p in self.products:
            for t in self.product_tasks[p]:
                self.prob += pulp.lpSum([self.assigned[p,t,c,s] for c in self.main_component_sets for s in self.stations]) == 1, f"SingleComponentSetSingleStationForTask_{p}_{t}"
                
        # Each each station has 0 or only one component set
        for s in self.stations:
            self.prob += pulp.lpSum([self.equipped[c,s] for c in self.main_component_sets]) <= 1, f"SingleComponentSetForStation_{s}"

        # Task cannot be assigned to station if the station is not equipped with the component set
        for s in self.stations:
            for c in self.main_component_sets:
                for p in self.products:
                    for t in self.product_tasks[p]:
                        self.prob += self.assigned[p,t,c,s] <= self.equipped[c,s], f"Equipment_{c}_{s}_{t}_{p}"

        # Auxiliary component set has to be assigned to the station if the specific main component set and the task are assigned to the station
        for p in self.products:
            for t in self.product_tasks[p]:
                for c in self.main_component_sets:
                    for s in self.stations:
                        for ac in self.aux_component_sets:
                            if aux_requirements[(t,c)] == ac:
                                self.prob += self.aux_equipped[ac,s] >= self.assigned[p,t,c,s], f"AuxiliaryEquipment_LT_{ac}_{s}_{c}_{t}_{p}"

        # Component set cannot be assigned to task step if the lead time is less than 0
        for p in self.products:
            for t in self.product_tasks[p]:
                for c in self.main_component_sets:
                    if self.task_times[(t,c)] < 0:
                        for s in self.stations:
                            self.prob += self.assigned[p,t,c,s] == 0, f"LeadTimeNotNegative_{p}_{t}_{c}_{s}"
            for t_1, t_2 in zip(self.product_tasks[p], self.product_tasks[p][1:]):
                for c in self.main_component_sets:
                    if self.task_times[(t_1,c)] < 0 or self.task_times[(t_2,c)] < 0:
                        for s in self.stations:
                            self.prob += self.assigned_same_station[p,t_1,t_2,c,s] == 0, f"LeadTimeNotNegative_{p}_{t_1}_{t_2}_{c}_{s}"

    def __task_order_constraint(self):
        """Flowshop constraint
        - Former task must be assigned to former station than the latter task. """
        # Former task must be assigned to former station than the latter task
        if self.settings['flowshop']:
            for p in self.products:
                for t_1, t_2 in zip(self.product_tasks[p], self.product_tasks[p][1:]):
                    for s_1 in range(len(self.stations)):
                        # t_2がs_1に割り付けられているとき、t_1はs_1と同じかより前のステーションに割り付けられていなければならない。
                        self.prob += pulp.lpSum(self.assigned[p,t_2,c,self.stations[s_1]] for c in self.main_component_sets) <= pulp.lpSum(self.assigned[p,t_1,c,s] for s in self.stations[:s_1+1] for c in self.main_component_sets), f"TaskOrder_{t_1}to{t_2}_at_{s_1}"

    def __cycletime_constraint(self):
        # overall cycle time must be larger than the cycle time of each station
        for p in self.products:
            for s in self.stations:
                self.prob += self.cycletime[p,s] <= self.overall_cycle_time[p], f"OverallCycleTime_{p}_{s}"

        # If task t_1 and next task t_2 are assigned to station s_1, s_2, trans must be 1
        for p in self.products:
            for t_1, t_2 in zip(self.product_tasks[p], self.product_tasks[p][1:]):
                for s_1 in self.stations:
                    for s_2 in self.stations:
                        if s_1 != s_2:
                            self.prob += pulp.lpSum([self.assigned[p,t_1,c,s_1] for c in self.main_component_sets]) >= self.trans[p,t_1,t_2,s_1,s_2], f"Transport_{t_1}_{t_2}_{s_1}_{s_2}_must_be_0_if_{p}_{t_1}_not_assigned_to_{s_1}"
                            self.prob += pulp.lpSum([self.assigned[p,t_2,c,s_2] for c in self.main_component_sets]) >= self.trans[p,t_1,t_2,s_1,s_2], f"Transport_{t_1}_{t_2}_{s_1}_{s_2}_must_be_0_if_{p}_{t_2}_not_assigned_to_{s_2}"
                            self.prob += pulp.lpSum([self.assigned[p,t_1,c,s_1] for c in self.main_component_sets]) + pulp.lpSum([self.assigned[p,t_2,c,s_2] for c in self.main_component_sets]) - 1 <= self.trans[p,t_1,t_2,s_1,s_2], f"Transport_{t_1}_{t_2}_{s_1}_{s_2}_must_be_1_if_{p}_{t_1}_assigned_{s_1}_and_{p}_{t_2}_assigned_{s_2}"
        # if transport_can_be_bottleneck is True, overal cycle time must be larger than the transport time
        for p in self.products:
            for t_1, t_2 in zip(self.product_tasks[p], self.product_tasks[p][1:]):
                for s_1 in self.stations:
                    for s_2 in self.stations:
                        if s_1 != s_2:
                            if self.transport_can_be_bottleneck[(s_1,s_2)]:
                                self.prob += self.trans[p,t_1,t_2,s_1,s_2] * self.transport_times[(s_1,s_2)] <= self.overall_cycle_time[p], f"OverallCycleTime_{p}_GT_TransportTime_{p}_{t_1}_{t_2}_{s_1}_{s_2}"

        # If task t_1,t_2 are assigned to the same station
        for p in self.products:
            for t_1, t_2 in zip(self.product_tasks[p], self.product_tasks[p][1:]):
                for c in self.main_component_sets:
                    for s in self.stations:
                        self.prob += self.assigned[p,t_1,c,s] >= self.assigned_same_station[p,t_1,t_2,c,s], f"AssignedSameStation{t_1}and{t_2}_at_{s}_{c}LT{t_1}_{p}"
                        self.prob += self.assigned[p,t_2,c,s] >= self.assigned_same_station[p,t_1,t_2,c,s], f"AssignedSameStation{t_1}and{t_2}_at_{s}_{c}LT{t_2}_{p}"
                        self.prob += self.assigned[p,t_1,c,s] + self.assigned[p,t_2,c,s] - 1 <= self.assigned_same_station[p,t_1,t_2,c,s], f"AssignedSameStation{t_1}and{t_2}_at_{s}_{c}_{p}_GT"

        # cycle time constraint
        for p in self.products:
            for s in self.stations:
                # task time + import time of first task station + export time of last task station + import time of each import station + export time of each export station + changeover time <= cycle time
                self.prob += pulp.lpSum([self.assigned[p,t,c,s] * self.task_times[(t,c)] for t in self.product_tasks[p] for c in self.main_component_sets]) \
                 + (self.assigned[p,self.product_tasks[p][0],c,s] * self.import_times[c] for c in self.main_component_sets ) \
                 + (self.assigned[p,self.product_tasks[p][-1],c,s] * self.export_times[c] for c in self.main_component_sets ) \
                 + pulp.lpSum((self.assigned[p,t_2,c,s] - self.assigned_same_station[p,t_1,t_2,c,s]) * self.import_times[c] for c in self.main_component_sets for t_1, t_2 in zip(self.product_tasks[p], self.product_tasks[p][1:])) \
                 + pulp.lpSum((self.assigned[p,t_1,c,s] - self.assigned_same_station[p,t_1,t_2,c,s]) * self.export_times[c] for c in self.main_component_sets for t_1, t_2 in zip(self.product_tasks[p], self.product_tasks[p][1:])) \
                 + pulp.lpSum(self.assigned_same_station[p,t_1,t_2,c,s] * self.changeover_times[(c,t_1,t_2)] for t_1, t_2 in zip(self.product_tasks[p], self.product_tasks[p][1:]) for c in self.main_component_sets) <= self.cycletime[p,s], f"CycleTime_{s}_{p}"

        # The sum of overallcycletime*amount must be less than or equal to the time_period (all product must be produced to amount in time_period)
        self.prob += self.__total_production_time() <= self.time_period, "TotalCycleTime"

    def __operator_constraint(self):
        # operator worktime constraint
        for p in self.products:
            for c in self.main_component_sets:
                for s in self.stations:
                    self.prob += self.operator_paid_worktime[p,c,s] >= 0, f"WorkerPaidWorkTime_{p}_{c}_{s}_0"
                    self.prob += self.operator_paid_worktime[p,c,s] <= self.M * self.equipped[c,s], f"WorkerPaidWorkTime_{p}_{c}_{s}_M"
                    if self.settings['labor_cost_on_exact_cycletime']:
                        # worktime is calculated by the exact cycle time in each station. operator can do other jobs after the task is done.
                        self.prob += self.operator_paid_worktime[p,c,s] <= self.amount[p]*self.cycletime[p,s], f"WorkerPaidWorkTime_{p}_{c}_{s}"
                        self.prob += self.amount[p]*self.cycletime[p,s] - self.operator_paid_worktime[p,c,s] <= self.M * (1 - self.equipped[c,s]), f"WorkerPaidWorkTime_{p}_{c}_{s}_M2"
                    else:
                        # worktime is calculated by the overall cycle time of the product. operator cannot do any other job. 
                        self.prob += self.operator_paid_worktime[p,c,s] <= self.amount[p]*self.overall_cycle_time[p], f"WorkerPaidWorkTime_{p}_{c}_{s}"
                        self.prob += self.amount[p]*self.overall_cycle_time[p] - self.operator_paid_worktime[p,c,s] <= self.M * (1 - self.equipped[c,s]), f"WorkerPaidWorkTime_{p}_{c}_{s}_M2"

        # aux operator worktime constraint
        for p in self.products:
            for ac in self.aux_component_sets:
                for s in self.stations:
                    self.prob += self.aux_operator_paid_worktime[p,ac,s] >= 0, f"AuxWorkerPaidWorkTime_{p}_{ac}_{s}_0"
                    self.prob += self.aux_operator_paid_worktime[p,ac,s] <= self.M * self.aux_equipped[ac,s], f"AuxWorkerPaidWorkTime_{p}_{ac}_{s}_M"
                    if self.settings['labor_cost_on_exact_cycletime']:
                        # worktime is calculated by the exact cycle time in each station. operator can do other jobs after the task is done.
                        self.prob += self.aux_operator_paid_worktime[p,ac,s] <= self.amount[p]*self.cycletime[p,s], f"AuxWorkerPaidWorkTime_{p}_{ac}_{s}"
                        self.prob += self.amount[p]*self.cycletime[p,s] - self.aux_operator_paid_worktime[p,ac,s] <= self.M * (1 - self.aux_equipped[ac,s]), f"AuxWorkerPaidWorkTime_{p}_{ac}_{s}_M2"
                    else:
                        # worktime is calculated by the overall cycle time of the product. operator cannot do any other job. 
                        self.prob += self.aux_operator_paid_worktime[p,ac,s] <= self.amount[p]*self.overall_cycle_time[p], f"AuxWorkerPaidWorkTime_{p}_{ac}_{s}"
                        self.prob += self.amount[p]*self.overall_cycle_time[p] - self.aux_operator_paid_worktime[p,ac,s] <= self.M * (1 - self.aux_equipped[ac,s]), f"AuxWorkerPaidWorkTime_{p}_{ac}_{s}_M2"

        # operator count constraint
        if self.settings['max_operator_count'] >= 0:
            self.prob += pulp.lpSum([self.operator_counts[c]*self.equipped[c,s] for c in self.main_component_sets for s in self.stations]) <= self.settings['max_operator_count'], f"OperatorCount{self.settings['max_operator_count']}"

    def format_result(self):
        # Print the results
        print("Solver:", self.solver.name)
        print([f"{obj.name}:{pulp.LpStatus[stat]}" for obj,stat in zip(self.objectives, self.statuses)])

        if all(stat in (pulp.LpStatusOptimal, pulp.LpStatusNotSolved) for stat in self.statuses):
            for v in self.prob.variables():
                print(v.name, "=", v.varValue)
            print("Total Cost = ", pulp.value(self.__total_cost()))
            print("    Total Initial Cost = ", pulp.value(self.__total_initial_cost()))
            print("    Total Labor Cost = ", pulp.value(self.__total_labor_cost()))
            print("Total Production Time = ", pulp.value(self.__total_production_time()))
            print("Total Transportation Time = ", pulp.value(self.__total_transport_time()))
        else:
            print("The problem is infeasible or unbounded.")
            return
        result_assignment_data = []
        for p in self.products:
            for s in self.stations:
                for c in self.main_component_sets:
                    if any(self.assigned[p,t,c,s].varValue > 0 for t in self.product_tasks[p]):
                        result_assignment_data.append({'Product': p, 'Station': s, 'Task': ','.join([t for t in self.product_tasks[p] if self.assigned[p,t,c,s].varValue > 0]), 
                                                       'Component Set': c, 'Aux Component Set': ','.join([ac for ac in self.aux_component_sets if self.aux_equipped[ac,s].varValue > 0]), 'Cycle Time': self.cycletime[p,s].varValue, 
                                 'Prod Amount':self.amount[p], 'Operator Paid Work Time': self.operator_paid_worktime[p,c,s].varValue})
                                
        self.result_assignment_df = pd.DataFrame(result_assignment_data)
        result_workflow_data = []
        for p in self.products:
            for t_i, t in enumerate(self.product_tasks[p]):
                for s in self.stations:
                    for c in self.main_component_sets:
                        if self.assigned[p,t,c,s].varValue > 0:
                            if t_i < len(self.product_tasks[p])-1:
                                t_2 = self.product_tasks[p][t_i+1]
                                result_workflow_data.append({'Product': p, 'Task/Trans/Changeover': t, 'Station': s, 'Task Time': self.task_times[(t,c)], 
                                                         'Changeover Time': 0, 
                                                         'Import Time': self.import_times[c] if (t_i == 0 or self.assigned_same_station[p,self.product_tasks[p][t_i-1],t,c,s].varValue == 0) else 0, 
                                                         'Export Time': self.export_times[c] if (self.assigned_same_station[p,t,self.product_tasks[p][t_i+1],c,s].varValue == 0) else 0,
                                                         'Transport Time': 0})
                                if self.assigned_same_station[p,t,self.product_tasks[p][t_i+1],c,s].varValue > 0: 
                                    result_workflow_data.append({'Product': p, 'Task/Trans/Changeover': f"Changeover {t}->{t_2}", 
                                                         'Station': s, 'Task Time': 0, 
                                                             'Changeover Time': self.assigned_same_station[p,t,t_2,c,s].varValue * self.changeover_times[(c,t,t_2)], 
                                                             'Import Time': 0, 'Export Time': 0, 
                                                         'Transport Time': 0})
                                if self.assigned_same_station[p,t,self.product_tasks[p][t_i+1],c,s].varValue == 0: 
                                    result_workflow_data.append({'Product': p, 'Task/Trans/Changeover': f"Transport {s}->{[s_2 for s_2 in self.stations if s != s_2 and self.trans[p,t,t_2,s,s_2].varValue > 0][0]}", 
                                                         'Station': s, 'Task Time': 0, 
                                                             'Changeover Time': 0, 
                                                             'Import Time': 0, 'Export Time': 0, 
                                                         'Transport Time': sum(self.trans[p,t,t_2,s,s_2].varValue * self.transport_times[(s,s_2)] for s_2 in self.stations if s != s_2)})
                            else:
                                result_workflow_data.append({'Product': p, 'Task/Trans/Changeover': t, 'Station': s, 'Task Time': self.task_times[(t,c)], 
                                                         'Changeover Time': 0, 
                                                         'Import Time': 0, 
                                                         'Export Time': self.export_times[c],
                                                         'Transport Time': 0})
        self.result_workflow_df = pd.DataFrame(result_workflow_data)

    def print_result_pretty(self):
        print(self.result_assignment_df)
        print(self.result_workflow_df)

if __name__ == "__main__":
    # M-BOM
    products = ['inverter1', 'inverter2']
    task_steps = {'inverter1':['Block', 'SubAsmA', 'Cover'], 
                  'inverter2':['Block', 'SubAsmB', 'Cover']}
    stations = [f"station_{i}" for i in range(sum(len(tasks) for tasks in task_steps.values()))]
    # BOE(bill of equipment)
    main_component_sets = ['X', 'Y']
    # auxiliary component sets
    aux_component_sets = {'x','y','none'}
    # The invest cost of investing in each component set
    initial_costs = {'X': 3000, 'Y': 2000}
    aux_initial_costs = {'x': 1000, 'y': 500, 'none':0}
    # the labor cost of each component set
    labor_costs = {'X': 100, 'Y': 50}
    aux_labor_costs = {'x': 0, 'y': 0, 'none':0}
    # the energy cost of each component set
    energy_costs = {'X': 10, 'Y': 5}
    # the maintenance cost of each component set
    maintenance_costs = {'X': 50, 'Y': 30}
    # the operator count of each component set
    operator_counts = {'X': 2, 'Y': 1}
    # import/export a part from/to station of each component set
    import_times = {'X': 0.1, 'Y': 0.1}
    export_times = {'X': 0.1, 'Y': 0.1}
    # transport time between each station
    transport_times = {(s_1,s_2):0.1*abs(s_i_1-s_i_2) for s_i_1, s_1 in enumerate(stations) for s_i_2, s_2 in enumerate(stations)}
    # transportation of single part at once (ex. pick&place robot, not conveyer) can be a bottleneck of the line. 
    transport_can_be_bottleneck = {(s_1,s_2):0 for s_1 in stations for s_2 in stations}
    transport_can_be_bottleneck[('station_0','station_1')] = 1
    
    # The time for each task step with each component set
    task_times = {('Block', 'X'): 1, ('Block', 'Y'): -2, ('SubAsmA', 'X'): -2, ('SubAsmA', 'Y'): 2, ('SubAsmB', 'X'): 1, ('SubAsmB', 'Y'): 200, ('Cover', 'X'): 1, ('Cover', 'Y'): 1}
    # auxiliary component set for each task step with each component set. 
    aux_requirements = {('Block', 'X'): 'x', ('Block', 'Y'): 'x', ('SubAsmA', 'X'): 'none', ('SubAsmA', 'Y'): 'none', ('SubAsmB', 'X'): 'none', ('SubAsmB', 'Y'): 'none', ('Cover', 'X'): 'none', ('Cover', 'Y'): 'y'}
    # Changeover time between each task step with same component set
    changeover_times = {('X','Block','SubAsmA'): 0, ('X','SubAsmA','Cover'): 0, ('Y','Block','SubAsmA'): 1, ('Y','SubAsmA','Cover'): 1,
                        ('X','Block','SubAsmB'): 1, ('X','SubAsmB','Cover'): 1, ('Y','Block','SubAsmB'): 1, ('Y','SubAsmB','Cover'): 1}
    # Changeover time between each component set
    stations_change_times = ({('X','X'): 0, ('X','Y'): 1, ('Y','X'): 1, ('Y','Y'): 0})
    # production requirement
    # time period length of production
    time_period = 48
    # The production_amount of each product in time period
    amount = {'inverter1':3, 'inverter2':4}
    settings = {
        'flowshop': True,
        'depreciation': True,
        'depreciation_timespan': 200,
        'labor_cost_on_exact_cycletime': True,
        'max_operator_count': 10,
    }

    capp = SimpleCAPP(
        products=products, 
        product_tasks=task_steps, 
        stations=stations, 
        main_component_sets=main_component_sets, 
        aux_component_sets=aux_component_sets,
        initial_costs=initial_costs, 
        aux_initial_costs=aux_initial_costs,
        labor_costs=labor_costs, 
        aux_labor_costs=aux_labor_costs,
        operator_counts=operator_counts,
        task_times=task_times, 
        aux_requirements=aux_requirements,
        changeover_times=changeover_times, 
        amount=amount, 
        time_period=time_period,
        import_times=import_times,
        export_times=export_times,
        transport_times=transport_times,
        transport_can_be_bottleneck=transport_can_be_bottleneck)
    capp.set_settings(settings=settings)
    capp.set_solver('PULP_CBC_CMD')
    capp.solve()
    if all(stat in (pulp.LpStatusOptimal, pulp.LpStatusNotSolved) for stat in capp.statuses):
        capp.format_result()
        capp.print_result_pretty()

    # relaxed_prob, infeasible_constraints = relax_problem.relax_problem(capp.prob)
    # relaxed_prob.solve()
    # lpvars = {lpvar.name: lpvar.value() for lpvar in relaxed_prob.variables()}
    # print(f"infeasible constraints: {infeasible_constraints}")
    # print(f"{' Objective Value ':=^60}")
    # print(f"Status: {pulp.LpStatus[relaxed_prob.status]}")
    # print(f"Objective: {relaxed_prob.objective.value()}")
    # print(f"{' Variables Values ':=^60}")
    # for name, value in lpvars.items():
    #     print(f"{name}: {value}")
    
    # print(f"{' Problem Constraints ':=^60}")
    # for lpname, lpcons in relaxed_prob.constraints.items():
    #     c = _c = lpcons.__str__().replace('*', ' * ')
    #     for name, value in lpvars.items():
    #         c = c.replace(name, str(value))
    #     res = eval(c)
    #     print(f"{lpname}: {_c} -> {c} -> {res}")

    # productsは別の名前でなくてはならず、同じ製品ならば同じタスクであっても別の名前として登録する必要がある。
    # m4_screw_1, m4_screw_2など。ただし別の製品ならば同じ名前で良い。

    # component_setは事前に必要な組み合わせの数だけ列挙しておく必要がある。
    # taskA = ['component1', 'component2', 'component3'] （例えばロボット、ハンド、ストレージ）
    # taskB = ['component1', 'component4', 'component5']
    # このとき、taskA,taskBを両方できるcomponent_setは
    # component_set = [
    # ['component1', 'component2', 'component3'],
    # ['component1', 'component4', 'component5'],
    # ['component1', 'component2', 'component3', 'component4', 'component5'],
    # ]となる。
    # このとき、component_set[2]はtaskA,taskBを両方できるが、component_set[0]はtaskBを実行できない。
    # さらに複数のロボットで並列作業する場合、ロボットとハンドなど必要なコンポーネントをN個ずつ含むcomponent_setを列挙する。
    # 同一製品内のタスク間の段取りはchangeover_timeで定義する。異なる製品間のスケジューリングと段取り時間は別に定義する。
    # さらにこのとき、component_setはcellのcapacity以下である必要があるなど他の制約もある。
    # セル自身についてもcell.capacityを定義して、component_setに追加できる。例えばcomponent_set[0]=['component1', 'component2', 'component3','cell1']など。
    
    # task timeがすべてのcomponent_setで負の値を取ることはあってはならない。負の値を取る場合は、そのcomponent_setはそのtaskを実行できない。 

    # 搬送時間（ステーションsから搬出後、s+1搬入まで）とバッファの量はステーションごとのサイクルタイム、ボトルネックには関係ない(搬送時間＜作業時間ならば)。一方で、ラインのリードタイムに関係する。
    # ラインのリードタイムはsum(cycletime for each station) + sum(transport_time for each station) + sum(buffer_time for each station)となる。
    # ラインのリードタイムは最も短いことが仕掛在庫の低減につながる。

    # レイアウトについては、floor.capacityとcell.size, cell.capacityとcomp.sizeで考える。
    # 矩形詰め込み問題か搬送方向の距離のみの詰め込み問題として定式化する。

    # 複数ライン（複数工場）での分担を考慮できると、生産量の割付も含めて変数とできて、生産量割付含めて最適化できる。ラインビルド事業部からの引き合いはないけど顧客は考慮しているはず。

    # TODO: すべてのコンポーネントセットを候補とするとModeに比べて数が多く(インバータモックで4→54)、assignとequipの範囲が広くなりすぎ時間がかかる(3min→over60min)。taskごとのtoolReqなどによってModeに対応する部分を自動選別する必要がある。main component set, sub(aux) component set に分類して、mainが決まればcost,timeが決まるようにする。
    # メインとサブで分離することができるのはツール交換、治具交換で対応できる製品・作業が想定されるためであり、これらを別要素として考える必要はないのでは。
    # TODO: decompositionは複数製品の生産量制約の場合、作業間で分割してsum(overall_cycletime*目標量 for products)<期間の制約を残せば、対応可能。ただしこれまで通りdecompositionはmax_operator_countやmax_stationsが最初の方は意味を持たず、後半で不可能となりうる。分割を辞めるのではなく分割後に部分ごとに数値設定する方法を取ることも考えるべき。
    # TODO: decompositionの中で、すべてのタスクを行えないコンポーネントセット（すべてtasktimeが負の数）の除外を行う。計算時間を大きく減らせる。
    # TODO: decompositionは最適化の中ではなく前処理として行うことで、分割を確認できるし計算時間上限などを管理しやすい。
    # TODO: 最大stationsを自動算出する機能も移植が必要。最も長いtask timeの設備をすべて選択したときに収まる最小のstations数を算出する。
    # TODO: 既存設備に付いて考慮する。
    # TODO: ステーションが製品を作ったあと次の製品に合わせてchangeoverする時間を考慮せず、import/exportしか考慮していない。複数製品で考慮するなら生産計画のように次の製品の計画が必要。
    # TODO: セルずらしの機能。
    # TODO: バックエンドとして、FlaskなどでAPIを整備しておくこと。FastAPIだとドキュメントも自動生成できてかっこいいかも。
    # TODO: 最適化の入力としてのCSV,jsonからのデータ整形の実装。
    # TODO: 作業ライブラリ、コンポーネントライブラリの実装。
    # 居室のlinuxPCにGPC,居室windowsPCのブラウザからアクセスできたので、streamlitなども同様に動作するはず。
