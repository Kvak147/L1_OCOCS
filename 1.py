import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.interpolate import make_interp_spline

calibration_files = ['100Hz(0,5).csv', '100Hz(1,0).csv', '100Hz(1,5).csv', '100Hz(2,0).csv', '100Hz(2,5).csv',
                     '100Hz(3,0).csv']
harmonic_files = ['100Hz(2exp).csv', '2500Hz(2exp).csv', '5000Hz(2exp).csv', '10000Hz(2exp).csv']

V_in_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

N = 9
Fs = 5000
Ms_per_s = 1000

print("--- ЧАСТЬ 1: КАЛИБРОВКА ---")
data_dict = {}
for i, fname in enumerate(calibration_files):
    df = pd.read_csv(fname, header=None)
    V_ADC_mean = df[0].mean()
    V_i = V_in_values[i]
    data_dict[V_i] = V_ADC_mean

V_i_sorted = sorted(data_dict.keys())
E_V_ADC_sorted = [data_dict[v] for v in V_i_sorted]

df_calibration = pd.DataFrame({
    'V_i': V_i_sorted,
    'E_V_ADC': E_V_ADC_sorted
})
print("Средние значения кодового выхода АЦП:")
print(df_calibration)

print("\n--- ЧАСТЬ 2: ЛИНЕЙНАЯ РЕГРЕССИЯ ---")
slope, intercept, r_value, p_value, std_err = linregress(df_calibration['V_i'], df_calibration['E_V_ADC'])

V_ref = (2 ** N) / slope
Delta = (intercept * V_ref) / (2 ** N)

print(f"Результаты линейной регрессии:")
print(f"Коэффициент V_ref: {V_ref:.4f} В")
print(f"Сдвиг Delta: {Delta:.4f} В")

plt.figure(figsize=(10, 6))
plt.scatter(df_calibration['V_i'], df_calibration['E_V_ADC'], label='Экспериментальные точки', color='b', marker='o')
V_i_regression = np.linspace(min(df_calibration['V_i']), max(df_calibration['V_i']), 100)
E_V_ADC_regression = slope * V_i_regression + intercept
plt.plot(V_i_regression, E_V_ADC_regression, label='Линейная регрессия', color='r', linestyle='--')
plt.title('График передаточной характеристики АЦП')
plt.xlabel('Входное напряжение $V_i$, В')
plt.ylabel('Среднее значение кодового выхода $E(V_{ADC})$')
plt.legend()
plt.grid(True)
plt.show()

print("\n--- ЧАСТЬ 3: АНАЛИЗ ПОСТОЯННЫХ СИГНАЛОВ ---")
for i, fname in enumerate(calibration_files):
    df = pd.read_csv(fname, header=None)
    V_i = V_in_values[i]
    df.rename(columns={0: 'V_ADC'}, inplace=True)
    df['V_out'] = (df['V_ADC'] * V_ref) / (2 ** N) - Delta
    df['t'] = (np.arange(len(df)) / Fs) * Ms_per_s

    mean_V = df['V_out'].mean()
    var_V = df['V_out'].var()

    print(f"\nФайл: {fname}")
    print(f"Входное напряжение: {V_i:.2f} В")
    print(f"Оценка мат. ожидания V(t): {mean_V:.4f} В")
    print(f"Оценка дисперсии V(t): {var_V:.6e} В^2")

    plt.figure(figsize=(10, 6))
    plt.plot(df['t'], df['V_out'], label='Преобразованное напряжение', color='g')
    plt.axhline(y=V_i, color='r', linestyle='--', label=f'Заданное напряжение {V_i:.2f} В')
    plt.title(f'График V(t) для постоянного сигнала {V_i:.2f} В')
    plt.xlabel('Время, мс')
    plt.ylabel('Напряжение V, В')
    plt.legend()
    plt.grid(True)
    plt.show()

print("\n--- ЧАСТЬ 4: АНАЛИЗ ГАРМОНИЧЕСКИХ СИГНАЛОВ ---")
harmonic_params = [
    {'A': 0.75, 'f': 100000, 'phi': 0.0, 'offset': 1.25},
    {'A': 0.75, 'f': 50.075, 'phi': np.pi * 11 / 20, 'offset': 1.15},
    {'A': 0.75, 'f': 0.8, 'phi': np.pi / 4, 'offset': 1.20},
    {'A': 0.75, 'f': 3.45, 'phi': np.pi * 26 / 50, 'offset': 1.20}
]

for i, fname in enumerate(harmonic_files):
    df_harm = pd.read_csv(fname, header=None)
    df_harm.rename(columns={0: 'V_ADC'}, inplace=True)
    df_harm['V_out'] = (df_harm['V_ADC'] * V_ref) / (2 ** N) - Delta

    df_harm['t'] = (np.arange(len(df_harm)) / Fs) * Ms_per_s # s -> ms

    params = harmonic_params[i]
    A, f, phi, offset = params['A'], params['f'], params['phi'], params['offset']

    plt.figure(figsize=(10, 6))
    plt.plot(df_harm['t'], df_harm['V_out'], label='Преобразованный сигнал', color='b')
    t_theory = np.linspace(0, max(df_harm['t']) / Ms_per_s, 500)
    V_theory = A * np.sin(2 * np.pi * f * t_theory + phi) + offset
    t_theory_ms = t_theory * Ms_per_s

    plt.plot(t_theory_ms, V_theory, label=f'Модель $A={A}, f={f}, ϕ={phi}, offset={offset}$', color='r', linestyle='--')

    plt.title(f'Анализ гармонического сигнала из файла {fname}')
    plt.xlabel('Время, мс')
    plt.ylabel('Напряжение V, В')
    plt.legend()
    plt.grid(True)
    plt.show()

print(
    "Выводы: Сравните форму экспериментального и модельного сигналов. Обсудите, насколько хорошо модель описывает реальный сигнал и почему могут быть расхождения (шум, нелинейность АЦП и т.д.).")

