import serial
com_port_number = 5 # Номер COM-порта
try:
    ser = serial.Serial("COM" + str(com_port_number),500000)
    for _ in range(3):
        res = ser.readline()
        dec = res.decode('utf-8')
        print(dec)
    ser.close()
except IOError:
  print ("Ошибка: Указан неверный номер порта или серийной соединение не закрыто.")
