import krpc
import matplotlib.pyplot as plt
import csv  # Для сохранения данных в CSV


# Подключение к серверу kRPC
conn = krpc.connect('speed_checker')
vessel = conn.space_center.active_vessel

# Создание массивов для данных о времени и скорости
time_array = []
speed_array = []
# Получение скорости корабля во время полета
try:
    while True:
        time_val = conn.space_center.ut
        speed = vessel.flight(vessel.orbit.body.reference_frame).speed  # Получение текущей скорости
        time_array.append(time_val)  # Запись текущего времени в массив
        speed_array.append(speed)  # Запись текущей скорости в массив
        print("Время: {}, Скорость корабля: {} м/с".format(time_val, speed))  # Вывод считываемых данных в консоль
except KeyboardInterrupt:
    print("\nРучное прерывание. Завершение сбора данных...")

# Сохранение данных в CSV файл (дополнительный вариант)
with open('speedr_data.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Время (с)', 'Скорость (м/с)'])
    for t_val, s_val in zip(time_array, speed_array):
        writer.writerow([t_val, s_val])


# Построение графика скорости от времени
plt.plot(time_array, speed_array)
plt.title('Зависимость скорости от времени')
plt.xlabel('Время, s')
plt.ylabel('Скорость, m/s')
plt.grid(True)  # Добавлена сетка для лучшей читаемости
plt.savefig('speed_graph до выхода ксп.png', dpi=300, bbox_inches='tight')  # Сохранение графика в файл
print(f"График сохранен в файл: speed_graph.png")
plt.show()