# ===================================================================
# CÓDIGO 1: Importación de bibliotecas, fijación de semillas y selección de dispositivo
# (lst:libraries_seeds_device)
# ===================================================================
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import qmc

np.random.seed(23)
torch.manual_seed(42)

dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===================================================================
# CÓDIGO 2: Parámetros del dominio y definición de la difusividad hidráulica
# (lst:hall_setup_model)
# ===================================================================
T = 1.0     # en minutos
X = 13.0    # en milimetros
theta_0 = 0.5
theta_s = 1.0

def D(theta: torch.Tensor) -> torch.Tensor:
    """Difusividad hidráulica D(theta) = 247.1 * theta^4."""
    return 247.1 * (theta**4)

# ===================================================================
# CÓDIGO 3: Inicializacion de los pesos de la función de pérdida y esquema adaptativo
# (lst:hall_loss_weights)
# ===================================================================
# --- Pesos iniciales de la funcion de perdida ---
lambda_R    = 1.0
lambda_IC   = 1.0
lambda_BC_0 = 1.0
lambda_BC_X = 1.0

# --- Funciones para actualizar los pesos ---
def set_loss_weights(l_R=1.0, l_IC=1.0, l_BC_0=1.0, l_BC_X=1.0):
    global lambda_R, lambda_IC, lambda_BC_0, lambda_BC_X
    lambda_R    = l_R
    lambda_IC   = l_IC
    lambda_BC_0 = l_BC_0
    lambda_BC_X = l_BC_X
    
def reset_loss_weights():
    """Reinicia los pesos a su valor uniforme."""
    set_loss_weights(1.0, 1.0, 1.0, 1.0)

# --- Parametros del esquema adaptativo ---
adaptive_weights = True
adapt_every      = 200    # frecuencia de actualizacion
clamp_min        = 0.1    # limite inferior
clamp_max        = 4.0    # limite superior
target_sum       = 4.0    # suma fija de los pesos

# ===================================================================
# CÓDIGO 4: Implementación de la arquitectura PINN
# (lst:hall_pinn_model)
# ===================================================================
# --- Definicion del modelo PINN ---
class PINN(nn.Module):
    def __init__(self, capas_ocultas):
        super().__init__()
        self.activacion = torch.tanh
        dims = [2] + list(capas_ocultas) + [1]
        capas = []
        for i in range(len(dims) - 1):
            lin = nn.Linear(dims[i], dims[i+1])
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
            capas.append(lin)
        self.capas = nn.ModuleList(capas)
        
    def forward(self, entrada):
        z = entrada
        for i in range(len(self.capas) - 1):
            z = self.activacion(self.capas[i](z))
        return self.capas[-1](z)

# --- Arquitectura de las capas ocultas ---
capas_ocultas = [50] * 5
modelo = PINN(capas_ocultas).to(dispositivo)

# ===================================================================
# CÓDIGO 5: Implementación del residuo PDE mediante diferenciación automática
# (lst:hall_residuo_pde)
# ===================================================================
def residuo_pde(modelo, t, x):
    t = t.clone().requires_grad_(True)
    x = x.clone().requires_grad_(True)

    TX = torch.cat((t, x), dim=1)
    theta = modelo(TX)

    grad_theta = torch.autograd.grad(
        theta, TX,
        grad_outputs=torch.ones_like(theta),
        create_graph=True
    )[0]
    theta_t = grad_theta[:, 0:1]
    theta_x = grad_theta[:, 1:2]

    # q = D(theta) * theta_x
    q = D(theta) * theta_x
    grad_q = torch.autograd.grad(
        q, TX,
        grad_outputs=torch.ones_like(q),
        create_graph=True
    )[0]
    q_x = grad_q[:, 1:2]

    return theta_t - q_x

# ===================================================================
# CÓDIGO 6: Muestreo de puntos de entrenamiento
# (lst:hall_sampling)
# ===================================================================
# --- Puntos interiores (dominio de la PDE) ---
n_interior = 10_000
lhs = qmc.LatinHypercube(d=2).random(n_interior)
t_in = torch.tensor(T * lhs[:, 0:1], dtype=torch.float32, device=dispositivo)
x_in = torch.tensor(X * lhs[:, 1:2], dtype=torch.float32, device=dispositivo)

# --- Puntos de frontera en x = 0 (BC_0) ---
n_bc0 = 1_000
t_0 = torch.rand((n_bc0, 1), dtype=torch.float32, device=dispositivo) * T
x_0 = torch.zeros((n_bc0, 1), dtype=torch.float32, device=dispositivo)
theta_data_0 = torch.full((n_bc0, 1), theta_s, dtype=torch.float32, device=dispositivo)

# --- Puntos de frontera en x = X (BC_X) ---
n_bcx = 500
t_X = torch.rand((n_bcx, 1), dtype=torch.float32, device=dispositivo) * T
x_X = torch.full((n_bcx, 1), X, dtype=torch.float32, device=dispositivo)
theta_data_X = torch.full((n_bcx, 1), theta_0, dtype=torch.float32, device=dispositivo)

# --- Puntos de condicion inicial en t = 0 (IC) ---
n_ic = 1_000
eps = 0.01 * X
x_IC = torch.rand((n_ic, 1), dtype=torch.float32, device=dispositivo) * (X - 2*eps) + eps
t_IC = torch.zeros((n_ic, 1), dtype=torch.float32, device=dispositivo)
theta_IC_data = torch.full((n_ic, 1), theta_0, dtype=torch.float32, device=dispositivo)

# --- Seleccion aleatoria de indices (mini-batching) ---
def rand_idx(n, B):
    B = min(B, n)
    return torch.randperm(n, device=dispositivo)[:B]

# ===================================================================
# CÓDIGO 7: Funciones de pérdida total y por minibatches
# (lst:hall_losses)
# ===================================================================
# --- Funcion de perdida (full-batch) ---
def loss_full():
    R_res = residuo_pde(modelo, t_in, x_in)
    L_R = torch.mean(R_res**2)

    theta_IC_pred = modelo(torch.cat((t_IC, x_IC), dim=1))
    L_IC = torch.mean((theta_IC_pred - theta_IC_data)**2)

    theta_0_pred = modelo(torch.cat((t_0, x_0), dim=1))
    L_BC0 = torch.mean((theta_0_pred - theta_data_0)**2)

    theta_X_pred = modelo(torch.cat((t_X, x_X), dim=1))
    L_BCX = torch.mean((theta_X_pred - theta_data_X)**2)

    L_tot = (
        lambda_R    * L_R +
        lambda_IC   * L_IC +
        lambda_BC_0 * L_BC0 +
        lambda_BC_X * L_BCX
    )
    return L_tot, L_R, L_IC, L_BC0, L_BCX

# --- Funcion de perdida con minibatches ---
def loss_minibatch(ii, i0, iX, io):
    R_res = residuo_pde(modelo, t_in[ii], x_in[ii])
    L_R = torch.mean(R_res**2)

    theta_IC_pred = modelo(torch.cat((t_IC[io], x_IC[io]), dim=1))
    L_IC = torch.mean((theta_IC_pred - theta_IC_data[io])**2)

    theta_0_pred = modelo(torch.cat((t_0[i0], x_0[i0]), dim=1))
    L_BC0 = torch.mean((theta_0_pred - theta_data_0[i0])**2)

    theta_X_pred = modelo(torch.cat((t_X[iX], x_X[iX]), dim=1))
    L_BCX = torch.mean((theta_X_pred - theta_data_X[iX])**2)

    L_tot = (
        lambda_R    * L_R +
        lambda_IC   * L_IC +
        lambda_BC_0 * L_BC0 +
        lambda_BC_X * L_BCX
    )
    return L_tot, L_R, L_IC, L_BC0, L_BCX

# ===================================================================
# CÓDIGO 8: Fase de entrenamiento con Adam y balanceo adaptativo
# (lst:hall_adam_training)
# ===================================================================
# --- Fase 1: entrenamiento con Adam (mini-batch) ---
epocas_adam = 15_000
B_int = 1_000
B_bc0 = 100
B_bcx = 50
B_ic  = 100

opt = optim.Adam(modelo.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=(0.5)**(1/5000))

print(f"Iniciando entrenamiento Adam en {dispositivo}...")

for ep in range(epocas_adam):
    modelo.train()
    opt.zero_grad()

    # Indices aleatorios para los distintos conjuntos de puntos
    ii = rand_idx(n_interior, B_int)
    i0 = rand_idx(n_bc0,      B_bc0)
    iX = rand_idx(n_bcx,      B_bcx)
    io = rand_idx(n_ic,       B_ic)

    # Perdida total y terminos elementales (mini-batch)
    L_tot, L_R, L_IC, L_BC0, L_BCX = loss_minibatch(ii, i0, iX, io)
    L_tot.backward()
    opt.step()
    scheduler.step()

    # Balanceo adaptativo de los terminos de perdida
    if adaptive_weights:
        # Reinicio periodico de los pesos
        if (ep > 0) and (ep % 1000 == 0):
            reset_loss_weights()

        # Reajuste de pesos cada 'adapt_every' epocas
        if ep % adapt_every == 0:
            R_val  = float(L_R.detach().item())
            IC_val = float(L_IC.detach().item())
            B0_val = float(L_BC0.detach().item())
            BX_val = float(L_BCX.detach().item())

            eps_num = 1e-12
            inv = np.array(
                [1.0/(R_val+eps_num),
                 1.0/(IC_val+eps_num),
                 1.0/(B0_val+eps_num),
                 1.0/(BX_val+eps_num)],
                dtype=np.float64
            )
            # Normalizacion, acotacion y reescalado a 'target_sum'
            inv = inv / (inv.mean() + eps_num)
            inv = np.clip(inv, clamp_min, clamp_max)
            inv = inv * (target_sum / (inv.sum() + eps_num))

            set_loss_weights(*inv.tolist())
    
    # Reporte de progreso (opcional)
    if ep % 1000 == 0 or ep == epocas_adam - 1:
        L_full, _, _, _, _ = loss_full()
        print(f"Época {ep}/{epocas_adam}: Pérdida Total = {L_full.item():.6e} | Pesos: R={lambda_R:.2f}, IC={lambda_IC:.2f}, BC0={lambda_BC_0:.2f}, BCX={lambda_BC_X:.2f}")

# ===================================================================
# CÓDIGO 9: Fase de entrenamiento con L-BFGS y reajuste adaptativo
# (lst:hall_lbfgs_training)
# ===================================================================
# --- Fase 2: entrenamiento con L-BFGS (full-batch) ---
NUM_LBFGS_ROUNDS   = 1
MAX_ITER_PER_ROUND = 15_000

print("\nIniciando entrenamiento L-BFGS...")

for rnd in range(1, NUM_LBFGS_ROUNDS + 1):
    opt_lbfgs = optim.LBFGS(
        modelo.parameters(),
        max_iter=MAX_ITER_PER_ROUND,
        tolerance_grad=1e-12,
        tolerance_change=1e-12,
        history_size=200,
        line_search_fn='strong_wolfe'
    )

    def closure():
        opt_lbfgs.zero_grad()
        L_tot, L_R, L_IC, L_BC0, L_BCX = loss_full()
        L_tot.backward()
        return L_tot

    L_final = opt_lbfgs.step(closure)
    print(f"Ronda L-BFGS {rnd}/{NUM_LBFGS_ROUNDS} finalizada. Pérdida: {L_final.item():.6e}")

    # Balanceo adaptativo despues de cada ronda L-BFGS
    if adaptive_weights:
        Ltot, L_R, L_IC, L_BC0, L_BCX = loss_full()
        R_val  = float(L_R.detach().item())
        IC_val = float(L_IC.detach().item())
        B0_val = float(L_BC0.detach().item())
        BX_val = float(L_BCX.detach().item())

        reset_loss_weights()
        eps_num = 1e-12
        inv = np.array(
            [1.0/(R_val+eps_num),
             1.0/(IC_val+eps_num),
             1.0/(B0_val+eps_num),
             1.0/(BX_val+eps_num)],
            dtype=np.float64
        )
        inv = inv / (inv.mean() + eps_num)
        inv = np.clip(inv, clamp_min, clamp_max)
        inv = inv * (target_sum / (inv.sum() + eps_num))

        set_loss_weights(*inv.tolist())
        print(f"Pesos L-BFGS reajustados: R={lambda_R:.2f}, IC={lambda_IC:.2f}, BC0={lambda_BC_0:.2f}, BCX={lambda_BC_X:.2f}")
