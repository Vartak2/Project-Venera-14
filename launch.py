import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd

# ============================================================================
# ФИЗИЧЕСКИЕ КОНСТАНТЫ ПЛАНЕТЫ EVE
# ============================================================================

# Параметры планеты Eve
R_eve = 700000.0          # Радиус планеты Eve, м
g0_eve = 16.7             # Ускорение свободного падения на поверхности, м/с²

# Параметры атмосферы Eve
rho0_eve = 5.0            # Плотность на поверхности, кг/м³
H_atm_eve = 7000.0        # Шкала высоты атмосферы, м

# ============================================================================
# ПАРАМЕТРЫ СПУСКАЕМОГО АППАРАТА
# ============================================================================

m0 = 2000.0               # Масса аппарата, кг
A_base = 2.0              # Площадь поперечного сечения аппарата, м²
Cd_base = 1.2             # Коэффициент лобового сопротивления аппарата

# Параметры парашюта
A_parachute = 100.0       # Площадь парашюта в раскрытом состоянии, м²
Cd_parachute = 2.2        # Коэффициент сопротивления парашюта
v_parachute_deploy = 500.0  # Скорость раскрытия парашюта, м/с

# Начальные условия входа в атмосферу
h0 = 100000.0             # Начальная высота, м
v0 = 5000.0               # Начальная скорость, м/с
theta0 = -30.0 * np.pi / 180.0  # Угол входа

# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def load_experimental_data():
    """Загрузка экспериментальных данных из CSV-файлов"""
    try:
        # Загрузка данных по высоте
        height_df = pd.read_csv('height_data_eve.csv')
        time_height = height_df.iloc[:, 0].values
        height_exp = height_df.iloc[:, 1].values / 1000
        
        # Загрузка данных по скорости
        speed_df = pd.read_csv('speed_data_eve.csv')
        time_speed = speed_df.iloc[:, 0].values
        speed_exp = speed_df.iloc[:, 1].values
        
        return time_height, height_exp, time_speed, speed_exp
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        return None, None, None, None

def gravity(h):
    """Ускорение свободного падения на высоте h над поверхностью Eve"""
    r = R_eve + h
    return g0_eve * (R_eve / r)**2

def atmospheric_density(h):
    """Плотность атмосферы Eve на высоте h"""
    if h < 0:
        return rho0_eve
    return rho0_eve * np.exp(-h / H_atm_eve)

# ============================================================================
# СИСТЕМА ДИФФЕРЕНЦИАЛЬНЫХ УРАВНЕНИЙ
# ============================================================================

class DescentModel:
    """Модель спуска аппарата на планету Eve"""
    
    def __init__(self):
        self.parachute_deployed = False
        
    def descent_equations(self, t, y):
        """
        Система дифференциальных уравнений движения спускаемого аппарата
        y = [h, x, v_y, v_x]
        """
        h, x, v_y, v_x = y
        
        # Скорость
        v = np.sqrt(v_x**2 + v_y**2)
        
        # Проверка условия раскрытия парашюта
        if not self.parachute_deployed and v <= v_parachute_deploy:
            self.parachute_deployed = True
        
        # Определение аэродинамических параметров
        if self.parachute_deployed:
            A_eff = A_base + A_parachute
            Cd_eff = 0.5 * Cd_base + 0.5 * Cd_parachute
        else:
            A_eff = A_base
            Cd_eff = Cd_base
        
        # Сила тяжести
        g = gravity(h)
        
        # Атмосферное сопротивление
        if v > 0:
            rho = atmospheric_density(h)
            drag_force = 0.5 * rho * Cd_eff * A_eff * v**2
            
            # Компоненты силы сопротивления
            drag_y = drag_force * (v_y / v)
            drag_x = drag_force * (v_x / v)
        else:
            drag_y = 0.0
            drag_x = 0.0
        
        # Уравнения движения
        dhdt = v_y
        dxdt = v_x
        dv_ydt = -drag_y / m0 - g
        dv_xdt = -drag_x / m0
        
        return [dhdt, dxdt, dv_ydt, dv_xdt]

# ============================================================================
# ФУНКЦИЯ МОДЕЛИРОВАНИЯ
# ============================================================================

def run_descent_simulation():
    """Запуск численного моделирования спуска"""
    
    # Создание модели
    model = DescentModel()
    
    # Начальные условия
    v_y0 = v0 * np.sin(theta0)
    v_x0 = v0 * np.cos(theta0)
    y0 = [h0, 0.0, v_y0, v_x0]
    
    # Время моделирования
    t_max = 10000.0
    t_eval = np.linspace(0, t_max, 100000)
    
    # Событие для остановки при посадке
    def touchdown_event(t, y):
        return y[0]
    touchdown_event.terminal = True
    touchdown_event.direction = -1
    
    # Решение системы с использованием лямбда-функции
    sol = solve_ivp(
        lambda t, y: model.descent_equations(t, y),
        [0, t_max],
        y0,
        method='RK45',
        t_eval=t_eval,
        events=[touchdown_event],
        rtol=1e-9,
        atol=1e-12,
        max_step=1.0
    )
    
    return sol, model

# ============================================================================
# ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
# ============================================================================

def plot_results(sol, model, time_height_exp=None, height_exp=None, 
                 time_speed_exp=None, speed_exp=None):
    """Построение графиков скорости и высоты от времени"""
    t = sol.t
    h = sol.y[0]
    v_y = sol.y[2]
    v_x = sol.y[3]
    
    # Вычисление полной скорости и высоты в км
    v_total = np.sqrt(v_y**2 + v_x**2)
    altitude = h / 1000.0  # Конвертация в км
    
    # Определение времени раскрытия парашюта
    parachute_deploy_time = None
    parachute_deploy_altitude = None
    
    # Ищем момент раскрытия парашюта
    for i in range(1, len(t)):
        v_curr = np.sqrt(v_y[i]**2 + v_x[i]**2)
        if v_curr <= v_parachute_deploy and not hasattr(model, 'parachute_deployed_time'):
            model.parachute_deployed_time = t[i]
            parachute_deploy_time = t[i]
            parachute_deploy_altitude = altitude[i]
            break
    
    # Создание графиков
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # График 1: Высота от времени
    ax1 = axes[0]
    ax1.plot(t, altitude, 'b-', linewidth=2, label='Модель')
    
    # Добавление экспериментальных данных по высоте
    if time_height_exp is not None and height_exp is not None:
        ax1.plot(time_height_exp, height_exp, 'r--', linewidth=2, 
                label='Эксперимент', alpha=0.8)
    
    if parachute_deploy_time is not None:
        ax1.axvline(x=parachute_deploy_time, color='g', linestyle='--', alpha=0.7, 
                   label=f'Раскрытие парашюта: {parachute_deploy_time:.1f} с')
        # Отметка высоты раскрытия
        ax1.plot(parachute_deploy_time, parachute_deploy_altitude, 'go', markersize=8)
    
    ax1.set_xlabel('Время, с', fontsize=12)
    ax1.set_ylabel('Высота, км', fontsize=12)
    ax1.set_title('Спуск на планету Eve: Высота от времени', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    ax1.set_ylim(bottom=0)
    
    # График 2: Скорость от времени
    ax2 = axes[1]
    ax2.plot(t, v_total, 'b-', linewidth=2, label='Модель')
    
    # Добавление экспериментальных данных по скорости
    if time_speed_exp is not None and speed_exp is not None:
        ax2.plot(time_speed_exp, speed_exp, 'r--', linewidth=2, 
                label='Эксперимент', alpha=0.8)
    
    if parachute_deploy_time is not None:
        ax2.axvline(x=parachute_deploy_time, color='g', linestyle='--', alpha=0.7)
    
    ax2.axhline(y=v_parachute_deploy, color='orange', linestyle=':', alpha=0.7,
               label=f'Скорость раскрытия: {v_parachute_deploy} м/с')
    
    ax2.set_xlabel('Время, с', fontsize=12)
    ax2.set_ylabel('Скорость, м/с', fontsize=12)
    ax2.set_title('Спуск на планету Eve: Скорость от времени', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# ОСНОВНАЯ ЧАСТЬ ПРОГРАММЫ
# ============================================================================

if __name__ == "__main__":
    # Загрузка экспериментальных данных
    time_height_exp, height_exp, time_speed_exp, speed_exp = load_experimental_data()
    
    # Запуск моделирования
    sol, model = run_descent_simulation()
    
    # Построение графиков
    plot_results(sol, model, time_height_exp, height_exp, time_speed_exp, speed_exp)