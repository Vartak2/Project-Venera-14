import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd

def load_experimental_data():
    """Загрузка экспериментальных данных из CSV-файлов"""
    try:
        # Загрузка данных по высоте
        height_df = pd.read_csv('height_data_to90.csv')
        height_df = height_df.iloc[8:].reset_index(drop=True)
        # Сдвигаем время так, чтобы первая запись начиналась с 0
        time_height = height_df.iloc[:, 0].values - height_df.iloc[0, 0]
        height_exp = height_df.iloc[:, 1].values  # второй столбец - высота
        
        # Загрузка данных по скорости
        speed_df = pd.read_csv('speed_data_to90.csv')
        speed_df = speed_df.iloc[8:].reset_index(drop=True)
        # Сдвигаем время так, чтобы первая запись начиналась с 0
        time_speed = speed_df.iloc[:, 0].values - speed_df.iloc[0, 0]
        speed_exp = speed_df.iloc[:, 1].values
        
        return time_height, height_exp, time_speed, speed_exp
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        return None, None, None, None


# Физические константы
g0 = 9.81          # Ускорение свободного падения на поверхности, м/с²
R_kerbin = 600000   # Радиус Земли, м

# Параметры атмосферы (экспоненциальная модель)
rho0 = 1.225           # Плотность воздуха на уровне моря, кг/м³
H_atm = 7000        # Шкала высоты атмосферы, м

# Ступень 1 (стартовая)
stage1 = {
    'm0': 92000,          # Начальная масса (топливо + конструкция), кг
    'm_dry': 10000,        # Сухая масса (конструкция), кг
    'm_fuel': 82000,       # Масса топлива, кг
    'burn_time': 47.8,     # Время работы, с
    'Isp': 315.0,          # Удельный импульс в вакууме, с
    'A': 10.0,             # Площадь поперечного сечения, м²
    'Cd': 0.5,             # Коэффициент лобового сопротивления
}

# Ступень 2
stage2 = {
    'm0': 42200,           # Начальная масса, кг
    'm_dry': 6000,      # Сухая масса, кг
    'm_fuel': 36200,     # Масса топлива, кг
    'burn_time': 50.4,     # Время работы, с
    'Isp': 310.0,          # Удельный импульс, с
    'A': 6.0,              # Площадь, м²
    'Cd': 0.4,             # Коэффициент сопротивления
}

# Ступень 3
stage3 = {
    'm0': 27800.0,         # Начальная масса, кг
    'm_dry': 9300.0,       # Сухая масса, кг
    'm_fuel': 18500.0,     # Масса топлива, кг
    'burn_time': 110.0,    # Время работы, с
    'Isp': 350.0,          # Удельный импульс, с
    'A': 4.0,              # Площадь, м²
    'Cd': 0.4,             # Коэффициент сопротивления
}

# Полезная нагрузка
payload_mass = 40000.0      # Масса полезной нагрузки, кг
# payload_mass = 0.0

# Программа управления углом тангажа (гравитационный разворот)
def pitch_program(t, total_burn_time):
    """Задаёт угол тангажа в зависимости от времени"""
    if t < 15.0:  # Вертикальный подъём первые 15 секунд
        return np.pi / 2.0
    # elif t > total_burn_time - 30.0:  # Последние 30 секунд - горизонтальный полёт
    #     return 0.0
    else:  # Гравитационный разворот между этими точками
        # Линейное уменьшение угла от 90 до 0 градусов
        frac = (t - 15.0) / (total_burn_time)
        return (90.0 - 90.0 * frac) * np.pi / 180.0

def gravity(h):
    """Ускорение свободного падения на высоте h"""
    return g0 * (R_kerbin / (R_kerbin + h))**2

def atmospheric_density(h):
    """Плотность атмосферы на высоте h (экспоненциальная модель)"""
    return rho0 * np.exp(-h / H_atm)

def get_stage_params(t, stages, stage_start_times):
    """
    Определяет параметры текущей активной ступени
    Возвращает: (m_dot, u, A, Cd, stage_active)
    m_dot = 0, если ступень закончила работу
    """
    t_elapsed = t
    current_stage = 0
    
    # Определяем, какая ступень сейчас активна
    for i, start_time in enumerate(stage_start_times):
        if t_elapsed >= start_time:
            current_stage = i
            t_in_stage = t_elapsed - start_time
        else:
            break
    
    if current_stage >= len(stages):
        # Все ступени отработали
        return 0.0, 0.0, stages[-1]['A'], stages[-1]['Cd'], False
    
    stage = stages[current_stage]
    
    # Проверяем, не закончилось ли время работы текущей ступени
    if t_in_stage > stage['burn_time']:
        # Двигатель выключен
        return 0.0, stage['Isp'] * g0, stage['A'], stage['Cd'], False
    
    # Двигатель работает
    m_dot = -stage['m_fuel'] / stage['burn_time']  # Отрицательный, т.к. масса уменьшается
    u = stage['Isp'] * g0  # Эффективная скорость истечения
    
    return m_dot, u, stage['A'], stage['Cd'], True


def rocket_equations(t, y, stages, stage_start_times, total_burn_time):
    """
    Система дифференциальных уравнений движения ракеты
    y = [h, x, v_y, v_x, m]
    """
    h, x, v_y, v_x, m = y
    
    # Скорость
    v = np.sqrt(v_x**2 + v_y**2)
    
    # Получаем параметры текущей ступени
    m_dot, u, A, Cd, engine_on = get_stage_params(t, stages, stage_start_times)
    
    # Угол тангажа
    theta = pitch_program(t, total_burn_time)
    
    # Сила тяги
    if engine_on and m_dot < 0:
        thrust = -u * m_dot  # m_dot отрицательный, поэтому thrust положительный
    else:
        thrust = 0.0
    
    # Сила тяжести
    g = gravity(h)
    
    # Атмосферное сопротивление
    if v > 0:
        rho = atmospheric_density(h)
        drag_force = 0.5 * rho * Cd * A * v**2
        drag_y = drag_force * (v_y / v)
        drag_x = drag_force * (v_x / v)
    else:
        drag_y = 0.0
        drag_x = 0.0
    
    # Уравнения движения
    dhdt = v_y  # Высота
    dxdt = v_x  # Горизонтальная дальность
    
    # Вертикальная скорость
    dv_ydt = (thrust * np.sin(theta) - drag_y) / m - g
    
    # Горизонтальная скорость
    dv_xdt = (thrust * np.cos(theta) - drag_x) / m
    
    # Масса
    dmdt = m_dot
    
    return [dhdt, dxdt, dv_ydt, dv_xdt, dmdt]

def setup_simulation():
    """Подготовка параметров для симуляции"""
    stages = [stage1, stage2, stage3]
    
    # Начальная масса
    m0_total = sum(s['m0'] for s in stages) + payload_mass
    
    # Начальные условия: [h, x, v_y, v_x, m]
    y0 = [0.0, 0.0, 0.0, 0.0, m0_total]
    
    # Время начала работы каждой ступени
    stage_start_times = [0.0]  # Первая ступень начинает в t=0
    for i in range(1, len(stages)):
        start_time = sum(stages[j]['burn_time'] for j in range(i))
        stage_start_times.append(start_time)
    
    # Общее время работы двигателей
    total_burn_time = sum(s['burn_time'] for s in stages)
    
    # Общее время моделирования (добавляем 10% после выключения двигателей)
    # t_max = total_burn_time * 1.1
    t_max = 110
    
    return stages, y0, stage_start_times, total_burn_time, t_max


def run_simulation():
    """Запуск численного моделирования"""
    # Настройка параметров
    stages, y0, stage_start_times, total_burn_time, t_max = setup_simulation()
    
    # Временные точки для вывода решения
    t_eval = np.linspace(0, t_max, 10000)
    
    # Параметры для передачи в функцию уравнений
    args = (stages, stage_start_times, total_burn_time)
    
    # Решение системы дифференциальных уравнений
    sol = solve_ivp(
        rocket_equations,
        [0, t_max],
        y0,
        args=args,
        method='RK45',
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-12,
        max_step=1.0
    )
    
    return sol, stage_start_times, stages

def plot_results(sol, stage_start_times, stages, 
                 time_height_exp=None, height_exp=None, 
                 time_speed_exp=None, speed_exp=None):
    """Построение графиков скорости и высоты от времени"""
    t = sol.t
    h = sol.y[0] # Высота в км
    v_y = sol.y[2]  # Вертикальная скорость
    v_x = sol.y[3]  # Горизонтальная скорость
    v_total = np.sqrt(v_y**2 + v_x**2)  # Полная скорость в км/с
    
    # Время отделения ступеней
    separation_times = []
    for i, start_time in enumerate(stage_start_times):
        if i < len(stages):
            separation_times.append(start_time + stages[i]['burn_time'])
    
    # Создание графиков
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # График 1: Высота от времени
    ax1 = axes[0]
    ax1.plot(t, h / 1000, 'b-', linewidth=2, label='Модель (высота)')
    
    # Добавление экспериментальных данных по высоте
    if time_height_exp is not None and height_exp is not None:
        ax1.plot(time_height_exp, height_exp / 1000, 'r--', linewidth=2, 
                label='Эксперимент (высота)', alpha=0.8)
    
    ax1.set_xlabel('Время, с', fontsize=12)
    ax1.set_ylabel('Высота, км', fontsize=12)
    ax1.set_title('Высота ракеты от времени', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='y')
    
    # Добавление линий отделения ступеней
    for i, sep_time in enumerate(separation_times):
        if sep_time < t[-1]:
            ax1.axvline(x=sep_time, color='gray', linestyle='--', alpha=0.5, 
                       label=f'Отделение {i+1} ступени' if i == 0 else "")
            ax1.text(sep_time-3, ax1.get_ylim()[0]*0.3, f'Ст.{i+1}', 
                    ha='center', va='bottom', fontsize=10, color='gray')
    
    ax1.legend(loc='upper left')
    
    # График 2: Скорость от времени
    ax2 = axes[1]
    ax2.plot(t, v_total / 1000, 'g-', linewidth=2, label='Модель (скорость)')
    
    # Добавление экспериментальных данных по скорости
    if time_speed_exp is not None and speed_exp is not None:
        ax2.plot(time_speed_exp, speed_exp / 1000, 'red', linestyle='--', 
                linewidth=2, label='Эксперимент (скорость)', alpha=0.8)
    
    ax2.set_xlabel('Время, с', fontsize=12)
    ax2.set_ylabel('Скорость, км/с', fontsize=12)
    ax2.set_title('Скорость ракеты от времени', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='y')
    
    # Добавление линий отделения ступеней
    for i, sep_time in enumerate(separation_times):
        if sep_time < t[-1]:
            ax2.axvline(x=sep_time, color='gray', linestyle='--', alpha=0.5)
            ax2.text(sep_time-3, ax2.get_ylim()[0]*0.2, f'Ст.{i+1}', 
                    ha='center', va='bottom', fontsize=10, color='gray')
    
    ax2.legend(loc='upper left')
    
    # Добавление информации о максимальных значениях
    max_h = np.max(h)
    max_v = np.max(v_total)
    
    info_text = f'Модель:\n' \
                f'Макс. высота: {round(max_h / 1000, 1)} км\n' \
                f'Макс. скорость: {round(max_v / 1000, 1)} км/с'
    
    # Добавление информации об экспериментальных данных, если они есть
    if time_height_exp is not None and height_exp is not None:
        max_h_exp = np.max(height_exp)
        info_text += f'\n\nЭксперимент (высота):\n' \
                     f'Макс. высота: {round(max_h_exp / 1000, 1)} км'
    
    if time_speed_exp is not None and speed_exp is not None:
        max_v_exp = np.max(speed_exp)
        info_text += f'\nЭксперимент (скорость):\n' \
                     f'Макс. скорость: {round(max_v_exp / 1000, 1)} км/с'
    
    plt.figtext(0.8, 0.09, info_text, fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.3, antialiased=True))
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Загрузка экспериментальных данных
    time_height_exp, height_exp, time_speed_exp, speed_exp = load_experimental_data()
    
    # Запуск моделирования
    sol, stage_start_times, stages = run_simulation()
    
    # Построение графиков с экспериментальными данными
    plot_results(sol, stage_start_times, stages, 
                 time_height_exp, height_exp, 

                 time_speed_exp, speed_exp)
