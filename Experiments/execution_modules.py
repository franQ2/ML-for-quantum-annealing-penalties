import numpy as np
import json


UNIFORM_PENALTIES = [10**9 for i in range(4)]

def get_Q(data, penalties=UNIFORM_PENALTIES):
  """
  Unless otherwise specified, penalties will be the same for all restrictions
  """
  # Penalties
  Pmin1, Pmin2, Pmax1, Pmax2 = penalties

  ## Electricity price:
  c_g = json.loads(data['grid_buying_price_values'])
  c_g = [round(x, 2) for x in c_g]

  # Batery:
  Cap_b = round(json.loads(data['e_bat_capacity_values'])[0], 2)
  Watt_b = round(json.loads(data['e_bat_charging_power_values'])[0], 2)
  Alpha = Watt_b/Cap_b

  SOC_0 = data['e_bat_SOC_ini']
  SOC_min = data['e_bat_SOC_min']
  SOC_max = data['e_bat_SOC_max']

  ## Matrix A.
  A = np.zeros((12, 12))

  A[0][0] = 2*Watt_b*c_g[0]/3
  A[1][1] = 4*Watt_b*c_g[0]/3

  ## Matrix B.
  B = np.zeros((12, 12))

  B[2][2] = 2*Watt_b*c_g[1]/3
  B[3][3] = 4*Watt_b*c_g[1]/3

  ## Matriz C.

  C = np.zeros((12, 12))
  beta_c = SOC_min - SOC_0 + Alpha

  C[0][0] = Pmin1*4/9**Alpha**2 -4/3*beta_c*Alpha
  C[1][1] = Pmin1*16/9**Alpha**2 -8/3*beta_c*Alpha
  C[4][5] = Pmin1*4/9
  C[4][0] = Pmin1*C[5][0] - 4/9*Alpha
  C[4][1] = - Pmin1*8/9*Alpha
  C[0][1] = Pmin1*16/9*Alpha**2
  C[4][4] = Pmin1*1/9 + Pmin1*2*beta_c/3
  C[5][5] = Pmin1*4/9 + Pmin1*4*beta_c/3

  ## Matrix D.

  D = np.zeros((12, 12))

  beta_d = SOC_min - SOC_0 + 2*Alpha

  D[0][0] = D[2][2] = Pmin2*4/3*Alpha*(Alpha/3 - beta_d)
  D[1][1] = D[3][3] = Pmin2*8/3*Alpha*(2*Alpha/3 - beta_d)

  D[6][6] = Pmin2*1/9 + Pmin2*2*beta_d/3
  D[7][7] = Pmin2*4/9 + Pmin2*4*beta_d/3

  D[0][1] = D[0][3] = D[1][2] = D[2][3] = Pmin2*16/9*Alpha**2
  D[1][3] = Pmin2*32/9*Alpha**2
  D[0][2] = Pmin2*8/9*Alpha**2

  D[3][7] = D[1][7] = - Pmin2*16/9*Alpha
  D[0][6] = D[2][6] = - Pmin2*4/9*Alpha
  D[0][7] = D[1][6] = D[2][7] = D[3][6] = - Pmin2*8/9*Alpha

  D[6][7] = Pmin2*4/9

  ## Matrix E.

  E = np.zeros((12, 12))

  beta_e = SOC_0 - SOC_max - Alpha

  E[0][0] = Pmax1*4/9**Alpha**2 + 4/3*beta_c*Alpha
  E[1][1] = Pmax1*16/9**Alpha**2 + 8/3*beta_c*Alpha
  E[8][8] = Pmax1*1/9 + Pmax1*2*beta_c/3
  E[9][9] = Pmax1*4/9 + Pmax1*4*beta_c/3
  E[0][1] = Pmax1*16/9*Alpha**2
  E[0][8] = Pmax1*4/9*Alpha
  E[8][9] = Pmax1*4/9
  E[0][9] = E[1][8] = Pmax1*8/9*Alpha
  E[1][9] = Pmax1*16/9*Alpha

  ## MatriX F.

  F = np.zeros((12, 12))

  beta_f = SOC_0 - 2*Alpha - SOC_max

  F[0][0] = F[2][2] = Pmax2*4/3*Alpha*(Alpha/3 + beta_f)
  F[1][1] = F[3][3] = Pmax2*8/3*Alpha*(2*Alpha/3 + beta_f)
  F[10][10] = Pmax2*1/9 + Pmax2*2*beta_f/3
  F[11][11] = Pmax2*4/9 + Pmax2*4*beta_f/3

  F[0][1] = F[0][3] = F[1][2] = F[2][3] = Pmax2*16/9*Alpha**2
  F[0][2] = Pmax2*8/9*Alpha**2
  F[1][3] = Pmax2*32/9*Alpha**2

  F[0][10] = F[2][10] = Pmax2*4/9*Alpha
  F[0][11] = F[1][10] = F[2][11] = F[3][10] = Pmax2*8/9*Alpha
  F[3][11] = F[1][11] = Pmax2*16/9*Alpha

  F[10][11] = Pmax2*4/9

  Q = A + B + C + D + E + F

  return Q


from pyomo.environ import *



def get_MINLP_solution(Q):
  model = ConcreteModel()
  # Define the set of indices
  n = 12
  model.I = RangeSet(n)

  # Define the binary variables
  model.x = Var(model.I, within=Binary)

  def rule(model):
    array = [model.x[i] for i in model.I]
    return np.dot(np.dot(array, Q), array)

  model.objective = Objective(rule=rule, sense=minimize)
  solver_factory = SolverFactory('mindtpy')
  solver_factory.solve(model, mip_solver='glpk', nlp_solver='ipopt', mip_solver_args={'timelimit': 6000}, time_limit=120)
  
  # Displaying the final variable values and the minimum found by the optimizer for the objective function. As it can be noticed, they only take values 0 or 1.
  # model.display()
  # model.pprint()
  return model


def check_two_step_solution(solution, data):
    """
    solution is an array of binary values
    data is a DataFrame row
    """
    Cap_b = json.loads(data['e_bat_capacity_values'])[0]
    Watt_b = json.loads(data['e_bat_charging_power_values'])[0]
    alpha = Watt_b/Cap_b

    SOC_0 = data['e_bat_SOC_ini']
    SOC_min = data['e_bat_SOC_min']
    SOC_max = data['e_bat_SOC_max']
    u_b_1 = 2*solution[0]/3 + 4*solution[1]/3 - 1
    u_b_2 = 2*solution[2]/3 + 4*solution[3]/3 - 1

    SOC_1 = SOC_0 + alpha*u_b_1
    SOC_2 = SOC_1 + alpha*u_b_2

    respects_conditions = True
    if SOC_min > SOC_1 or SOC_1 > SOC_max:
        respects_conditions = False

    if SOC_min > SOC_2 or SOC_2 > SOC_max:
        respects_conditions = False
    
    return respects_conditions


from dwave.system import LeapHybridSampler
from dimod import ExactSolver

def get_annealer_result(Q, solver = "exact"):
    """
    Use exact sampler for testing
    Dwave credentials must be updated before using hybrid solver
    """
    if solver=="hybrid":
      print("\nSending problem to hybrid sampler...")
      sampler = LeapHybridSampler(token='')
    else:
      sampler = ExactSolver()

    results_annealer = sampler.sample_qubo(Q)
    
    return results_annealer