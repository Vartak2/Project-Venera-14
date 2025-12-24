import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd

def load_experimental_data():
    """Загрузка экспериментальных данных из CSV-файлов"""
    try:
        # Загрузка данных по высоте
        height_df = pd.read_csv('height_data_eve.csv')
        height_df = height_df.iloc[63:].reset_index(drop=True)
        # Сдвигаем время так, чтобы первая запись начиналась с 0
        time_height = height_df.iloc[:, 0].values - height_df.iloc[0, 0]
        height_exp = height_df.iloc[:, 1].values  # второй столбец - высота
        
        # Загрузка данных по скорости
        speed_df = pd.read_csv('speed_data_eve.csv')
        speed_df = speed_df.iloc[63:].reset_index(drop=True)
        # Сдвигаем время так, чтобы первая запись начиналась с 0
        time_speed = speed_df.iloc[:, 0].values - speed_df.iloc[0, 0]
        speed_exp = speed_df.iloc[:, 1].values
        
        return time_height, height_exp, time_speed, speed_exp
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        return None, None, None, None

# ============================================================================
# ПАРАМЕТРЫ ПЛАНЕТЫ EVE (KSP)
# ============================================================================

R_eve = 700000  # Радиус планеты Eve [м]
g0_eve = 16.7   # Ускорение свободного падения на поверхности [м/с²]
rho0_eve = 4.7 # Плотность атмосферы на поверхности [кг/м³] (приближённо)
H_atm_eve = 8000  # Шкала высоты атмосферы [м]

# ============================================================================
# ПАРАМЕТРЫ СПУСКАЕМОГО АППАРАТА
# ============================================================================

m0 = 1700.0  # Начальная масса [кг]
D_parachute = 10.0  # Диаметр парашюта [м]
A_parachute = np.pi * (D_parachute/2)**2  # Площадь парашюта [м²]
Cd_parachute = 1.5  # Коэффициент сопротивления парашюта
Cd_body = 0.33  # Коэффициент сопротивления корпуса
A_body = 10.0  # Площадь поперечного сечения корпуса [м²]

# Высота раскрытия парашюта
parachute_deploy_altitude = 2500.0  # [м]

# ============================================================================
# ПАРАМЕТРЫ ВХОДА В АТМОСФЕРУ
# ============================================================================

entry_angle = -24.0 * np.pi / 180.0  # Угол входа в атмосферу (отрицательный = вниз)
v0_total = 3330.0  # Начальная полная скорость [м/с]
v0_vertical = v0_total * np.sin(entry_angle)  # Вертикальная составляющая
v0_horizontal = v0_total * np.cos(entry_angle)  # Горизонтальная составляющая

# ============================================================================
# ФУНКЦИИ ДЛЯ РАСЧЁТА СИЛ
# ============================================================================

def gravity(h):
    """Ускорение свободного падения на высоте h над поверхностью Eve"""
    return g0_eve * (R_eve / (R_eve + h))**2

def atmospheric_density(h):
    """Точная плотность атмосферы Eve из KSP"""
    # Данные атмосферы планеты Eve (высота в м, плотность в кг/м³)
    density_data = [
        (0, 4.97),      # Поверхность
        (10000, 0.31),
        (20000, 0.069),
        (30000, 0.018),
        (40000, 0.005),
        (50000, 0.0013),
        (60000, 0.0003),
        (70000, 0.00007),
        (80000, 0.00002),
        (90000, 0.000005)
    ]
    
    if h <= 0:
        return density_data[0][1]
    if h >= density_data[-1][0]:
        return density_data[-1][1]
    
    # Линейная интерполяция
    for i in range(len(density_data) - 1):
        h1, rho1 = density_data[i]
        h2, rho2 = density_data[i + 1]
        if h1 <= h <= h2:
            return rho1 + (rho2 - rho1) * (h - h1) / (h2 - h1)
    
    return 0.0

def get_drag_params(h):
    """
    Определяет параметры сопротивления в зависимости от высоты.
    Возвращает: (Cd, A) - эффективные коэффициент сопротивления и площадь
    """
    if h <= parachute_deploy_altitude:
        # Парашют раскрыт
        return Cd_parachute, A_parachute
    else:
        # Только аэродинамическое сопротивление корпуса
        return Cd_body, A_body

# ============================================================================
# СИСТЕМА ДИФФЕРЕНЦИАЛЬНЫХ УРАВНЕНИЙ
# ============================================================================

def descent_equations(t, y):
    """
    Система дифференциальных уравнений движения спускаемого аппарата.
    y = [h, v_y, v_x]
    h - высота над поверхностью [м]
    v_y - вертикальная скорость (положительная вверх, отрицательная вниз) [м/с]
    v_x - горизонтальная скорость [м/с]
    """
    h, v_y, v_x = y
    
    # Полная скорость
    v_total = np.sqrt(v_y**2 + v_x**2)
    
    # Получаем параметры сопротивления
    Cd, A = get_drag_params(h)
    
    # Сила тяжести (направлена вниз)
    g = gravity(h)
    
    # Атмосферное сопротивление (направлено против вектора скорости)
    if v_total > 0:
        rho = atmospheric_density(h)
        drag_force_total = 0.5 * rho * Cd * A * v_total**2
        drag_y = drag_force_total * (v_y / v_total)
        drag_x = drag_force_total * (v_x / v_total)
    else:
        drag_y = 0.0
        drag_x = 0.0
    
    # Уравнения движения
    dhdt = v_y  # Изменение высоты
    
    # Ускорения: сопротивление направлено против движения, тяжесть - вниз
    dv_ydt = -drag_y / m0 - g
    dv_xdt = -drag_x / m0
    
    return [dhdt, dv_ydt, dv_xdt]

# ============================================================================
# ВЫПОЛНЕНИЕ МОДЕЛИРОВАНИЯ
# ============================================================================

def run_descent_simulation():
    """Запуск численного моделирования спуска"""
    
    # Начальные условия: высота 100 км, вертикальная и горизонтальная скорости
    y0 = [90000.0, v0_vertical, v0_horizontal]
    
    # Время моделирования (до посадки или разумного предела)
    t_max = 2000.0
    
    # Временные точки для вывода решения
    t_eval = np.linspace(0, t_max, 10000)
    
    # Событие: посадка (высота = 1000 м)
    def landing_event(t, y):
        return y[0] - 1000
    landing_event.terminal = True
    landing_event.direction = -1
    
    # Решение системы дифференциальных уравнений
    sol = solve_ivp(
        descent_equations,
        [0, t_max],
        y0,
        events=[landing_event],
        method='RK45',
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-12,
        max_step=1.0
    )
    
    # Рассчитываем полную скорость для вывода
    v_total = np.sqrt(sol.y[1]**2 + sol.y[2]**2)
    
    return sol, v_total

# ============================================================================
# ПОСТРОЕНИЕ ГРАФИКОВ
# ============================================================================

def plot_results(sol, v_total, time_height_exp=None, height_exp=None, 
                 time_speed_exp=None, speed_exp=None):
    """Построение графиков скорости и высоты от времени"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # График полной скорости
    ax1.plot(sol.t, v_total, 'b-', linewidth=2, label='Модель')
    ax1.set_xlabel('Время, с')
    ax1.set_ylabel('Скорость, м/с')
    ax1.set_title('Скорость спускаемого аппарата')
    ax1.grid(True, alpha=0.3)

    # Добавление экспериментальных данных по скорости
    if time_speed_exp is not None and speed_exp is not None:
        ax1.plot(time_speed_exp, speed_exp, 'r--', linewidth=2, 
                label='Эксперимент', alpha=0.8)
    
    # Отметка раскрытия парашюта
    parachute_time_idx = np.argmax(sol.y[0, :] <= parachute_deploy_altitude)
    if parachute_time_idx > 0:
        parachute_time = sol.t[parachute_time_idx]
        # parachute_velocity = v_total[parachute_time_idx]
        # ax1.plot(parachute_time, parachute_velocity, 'ro', markersize=8, 
        #         label='Раскрытие парашюта')
        ax1.axvline(x=parachute_time, color='gray', linestyle='--', alpha=0.5, label="Раскрытие парашюта")
    
    # График высоты
    ax2.plot(sol.t, sol.y[0, :], 'g-', linewidth=2, label='Модель')
    ax2.set_xlabel('Время, с')
    ax2.set_ylabel('Высота, м')
    ax2.set_title('Высота над поверхностью')
    ax2.grid(True, alpha=0.3)

    # Добавление экспериментальных данных по высоте
    if time_height_exp is not None and height_exp is not None:
        ax2.plot(time_height_exp, height_exp, 'r--', linewidth=2, 
                label='Эксперимент', alpha=0.8)
    
    # Отметка раскрытия парашюта
    if parachute_time_idx > 0:
        # parachute_altitude = sol.y[0, parachute_time_idx]
        # ax2.plot(parachute_time, parachute_altitude, 'ro', markersize=8, 
        #         label='Раскрытие парашюта')
        ax2.axvline(x=parachute_time, color='gray', linestyle='--', alpha=0.5, label="Раскрытие парашюта")
    
    # Добавляем легенды
    ax1.legend()
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# ОСНОВНАЯ ПРОГРАММА
# ============================================================================

if __name__ == "__main__":
    time_height_exp, height_exp, time_speed_exp, speed_exp = load_experimental_data()

    # Запуск моделирования
    sol, v_total = run_descent_simulation()
    
    # Построение графиков
    plot_results(sol, v_total, time_height_exp, height_exp, time_speed_exp, speed_exp)