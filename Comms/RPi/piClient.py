import socket
import serial
import queue as q
from threading import Timer,Thread,Event
import time
import csv
import random

#from Crypto.Util.Padding import pad
from Crypto import Random
from Crypto.Cipher import AES
from Crypto.Util.py3compat import *
import base64

# Crypto.Util.Padding pad function
def pad(data_to_pad, block_size, style='pkcs7'):
	padding_len = block_size - len(data_to_pad) % block_size
	if style == 'pkcs7':
		padding = bchr(padding_len) * padding_len
	elif style == 'x923':
		padding = bchr(0) * (padding_len - 1) + bchr(padding_len)
	elif style == 'iso7816':
		padding = bchr(128) + bchr(0) * (padding_len - 1)
	else:
		raise ValueError("Unknown padding style")
	return data_to_pad + padding

# global variable decleration
#dataQueue = q.Queue()
voltage = 0
current = 0
power = 0
cumPower = 0

HELLO = ('H').encode()
ACK = ('A').encode()
NACK = ('N').encode()
READY = ('R').encode()

class SerClass:

	def setup(self):
		self.ser = serial.Serial("/dev/serial0", 115200)
		self.ser.reset_input_buffer()
		self.ser.reset_output_buffer()

	def handshake(self):
		print("InitiateHandshake")
		self.ser.write(HELLO)
		time.sleep(1)
		if self.ser.in_waiting > 0:
			reply = self.ser.read().decode()
			if(reply == 'A'):
				self.ser.write(ACK)
				print('Handshake Complete')
				return True
			else:
				print('pending')
		return False


	def run(self):
		global dataQueue

		self.setup()
		while self.handshake() is False:
			continue
		dataThread = DataReceiveClass(self.ser)
		dataThread.start()

class SocketClass():
	currMove = None
	message = None

	def createMsg(self):
		global voltage
		global current
		global power
		global cumPower

		self.actions = ['handmotor', 'bunny', 'tapshoulder', 'rocket', 'cowboy', 'hunchback', 'jamesbond', 'chicken', 'movingsalute', 'whip', 'logout']

		if self.currMove == -1:
			self.message = None
		else:
			self.message = ("#" + self.actions[self.currMove] + "|" + str(format(voltage, '.2f')) + "|" + str(format(current, '.2f')) + "|" + str(format(power, '.2f')) + "|" + str(format(cumPower, '.2f')) + "|").encode('utf8').strip()

	def machine(self):
		# ML code that will return an index
		self.currMove=random.randint(0,9)

	def run(self):
		SECRET_KEY = bytes("dancedancedance!", 'utf8')
		# setup connection
		print('Connecting to server')
		self.ipaddress = '127.0.0.1'
		self.port = 1234
		self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.s.connect((self.ipaddress, self.port))
		print("Connected to server " +self.ipaddress+ ", port: " +str(self.port))

		while True:
			self.machine()
			self.createMsg()
			print(self.message)

			iv = Random.new().read(AES.block_size)
			cipher = AES.new(SECRET_KEY, AES.MODE_CBC, iv)
			padMessage = pad(self.message, AES.block_size)
			encryptMsg = cipher.encrypt(padMessage)
			encodedMsg = base64.b64encode(iv + encryptMsg)

			time.sleep(3)
			self.s.send(encodedMsg)

		self.s.close()

class DataReceiveClass(Thread):
	#global dataQueue
	global voltage
	global current
	global power
	global cumPower

	def __init__(self, ser):
		Thread.__init__(self)
		self.ser = ser

	def run(self):
		self.readData()

	def readData(self):
		packet = self.ser.readline().decode()
		packet = packet.strip()
		print(packet)

		checksum = packet.rsplit(",", 1)[1]
		packet = packet.rsplit(",", 1)[0]

		if(len(packet) != int(checksum)):
			self.ser.write(NACK)
		else:
			self.ser.write(ACK)
			with open('/home/pi/Desktop/data.csv', 'a') as csvfile:
				filewriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
				dataList = []
				for x in range (0, 18):
					if x==0 or x==7:
						continue
					elif x==14:
						voltage = float(packet.split(',', 18)[x])
					elif x==15:
						current = float(packet.split(',', 18)[x])
					elif x==16:
						power = float(packet.split(',', 18)[x])
					elif x==17:
						cumPower = float(packet.split(',', 18)[x])
					else:
						val = float(packet.split(',', 18)[x])
						dataList.append(val)
				print(dataList)
				print(voltage)
				print(current)
				print(power)
				print(cumPower)
				filewriter.writerow(dataList)
		Timer(0.03, self.readData).start()


if __name__ == '__main__':
	SerComm = SerClass()
	#SerComm.run()
	serThread = Thread(target=SerComm.run)

	SocketComm = SocketClass()
	#SocketComm.run()
	socketThread = Thread(target=SocketComm.run)

	serThread.start()
	socketThread.start()
