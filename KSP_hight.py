import krpc
import matplotlib.pyplot as plt
import time as t
import csv


# Подключение к серверу kRPC
conn = krpc.connect('hight_checker')
vessel = conn.space_center.active_vessel

# Создание массивов для данных о времени и высоте
time_values = []
altitude_values = []
current_time = 0

# Получение высоты корабля на протяжении полета
try:
    while True:
        altitude = vessel.flight().surface_altitude
        time_values.append(current_time)  # Запись текущего времени в массив
        altitude_values.append(altitude)  # Запись текущей высоты в массив

        # Вывод считываемых данных в консоль
        print("Время: {}, Высота: {} м".format(current_time, altitude))
        t.sleep(1)  # КД 1 секунда для умеренных данных
        current_time += 1  # Исправлено имя переменной
except KeyboardInterrupt:
    pass

# Сохранение данных в файл CSV
with open('height_data_to90.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Время (s)', 'Высота (m)'])
    for time_val, alt_val in zip(time_values, altitude_values):
        writer.writerow([time_val, alt_val])


# Построение графика высоты от времени с помощью библиотеки pyplot
plt.figure(figsize=(10, 6))
plt.plot(time_values, altitude_values)
plt.title('Зависимость высоты от времени')
plt.xlabel('Время, s')
plt.ylabel('Высота, m')
plt.grid(True)
plt.savefig('flight_plot до 90.png')  # Сохранение графика в файл
plt.show()
