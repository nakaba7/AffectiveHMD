import serial

"""
16個の反射型光センサの値をシリアル通信でマイコンから受け取り, デコードする.
"""

class SerialConnection:
	def __init__(self, portNum, rate):
		self.ser = serial.Serial(portNum, rate, timeout = 1)
		self.values = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
		self.ser.reset_input_buffer()

	def UpdateSensorData(self):
		#<summary>
		# When micro computer receives the "b", it sends back the sensor data.
		# This system receive the 24 bytes which include the sensor data and start stop character
		#</summary>
		self.ser.write(str.encode('b'))
		byteBuffer = self.ser.read(24)
		SensorList = list(byteBuffer)
		if(len(SensorList) == 24 and SensorList[0] == ord("A") and SensorList[23] == ord("Y")):
			self.values[0] = (((SensorList[1]) & 0xff) << 2) + (((SensorList[9]) & 0xc0) >> 6)	    
			self.values[1] = (((SensorList[2]) & 0xff) << 2) + (((SensorList[9]) & 0x30) >> 4)
			self.values[2] = (((SensorList[3]) & 0xff) << 2) + (((SensorList[9]) & 0x0c) >> 2)
			self.values[3] = (((SensorList[4]) & 0xff) << 2) + ((SensorList[9]) & 0x03)
			self.values[4] = (((SensorList[5]) & 0xff) << 2) + (((SensorList[10]) & 0xc0) >> 6)	
			self.values[5] = (((SensorList[6]) & 0xff) << 2) + (((SensorList[10]) & 0x30) >> 4)
			self.values[6] = (((SensorList[7]) & 0xff) << 2) + (((SensorList[10]) & 0x0c) >> 2)
			self.values[7] = (((SensorList[8]) & 0xff) << 2) + ((SensorList[10]) & 0x03)
			self.values[8] = (((SensorList[13]) & 0xff) << 2) + (((SensorList[21]) & 0xc0) >> 6)
			self.values[9] = (((SensorList[14]) & 0xff) << 2) + (((SensorList[21]) & 0x30) >> 4)
			self.values[10] = (((SensorList[15]) & 0xff) << 2) + (((SensorList[21]) & 0x0c) >> 2)
			self.values[11] = (((SensorList[16]) & 0xff) << 2) + ((SensorList[21]) & 0x03)
			self.values[12] = (((SensorList[17]) & 0xff) << 2) + (((SensorList[22]) & 0xc0) >> 6)
			self.values[13] = (((SensorList[18]) & 0xff) << 2) + (((SensorList[22]) & 0x30) >> 4)
			self.values[14] = (((SensorList[19]) & 0xff) << 2) + (((SensorList[22])& 0x0c) >> 2)
			self.values[15] = (((SensorList[20]) & 0xff) << 2) + ((SensorList[22]) & 0x03)
		else:
			print("sensor data error")
			#print len(SensorList)

		self.ser.reset_input_buffer()
		self.ser.reset_output_buffer()

	def getSensorData(self):
		return self.values

class SerialConnectionExceptDisplay:
	def __init__(self, portNum, rate):
		self.ser = serial.Serial(portNum, rate, timeout = 1)
		self.values = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	def UpdateSensorData(self):
		sensorDataStr = self.ser.readline()
		sensorDataList = sensorDataStr.split(",")
		#print sensorDataList
		if(len(sensorDataList) != 17):
			#sensor data can't receive at first time
			print('sensor data error')
			return
		for i in range(0,len(self.values)):
			self.values[i] = int(sensorDataList[i], 10)

	def getSensorData(self):
		return self.values