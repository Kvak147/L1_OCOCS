import csv
import serial
import time
com_port_number = 5
file_name = '10000Hz(2exp).csv'
signal_time = 5
csvfile = open(file_name, 'w', newline='')
try:
    ser = serial.Serial("COM" + str(com_port_number),500000)
    start_time = time.time()
    while(time.time()-start_time < signal_time):
        res = ser.readline()
        dec = res.decode('utf-8')
        csvfile.write(dec)
    print("Запись прошла успешно")
    ser.close()
    csvfile.close()
except IOError:
  print ("Ошибка: Указан неверный номер порта или серийной соединение не закрыто.")