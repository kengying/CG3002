import socket
import serial
import queue as q
from threading import Timer,Thread,Event
import time
import csv
import random
import sys
import numpy as np

#from RFML2 import RFMLmain
#from cnn import cnn_main

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
voltage = 0
current = 0
power = 0
cumPower = 0
numpyArray = np.array([])

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
	RFMLmove = None
	slidingMove = None
	lastMsgTime = None

	def __init__(self, IPaddress, PORT):
		self.ipaddress = IPaddress
		self.port = PORT

	def createMsg(self):
		global voltage
		global current
		global power
		global cumPower

		self.actions = ['handmotor', 'bunny', 'tapshoulder', 'rocket', 'cowboy', 'hunchback', 'jamesbond', 'chicken', 'movingsalute', 'whip', 'logout']

		if self.currMove == None:
			self.message = None
		else:
			self.message = ("#" + self.actions[self.currMove] + "|" + str(format(voltage, '.2f')) + "|" + str(format(current, '.2f')) + "|" + str(format(power, '.2f')) + "|" + str(format(cumPower, '.2f')) + "|").encode('utf8').strip()

	def machine(self):
		global numpyArray

		# ML code that will return an index
		self.currMove=None
		self.continuePredict = True
		print("Running ML code")
		self.predictMove = [None, None, None, None, None]

		while(self.continuePredict):

			self.index = 0
			if self.index > 5:
				self.index = 0

			# pass array to ML only when there is length is 128
			# reset to empty
			if numpyArray.size > 128:
				self.predictMove[self.index] = RFMLmain(numpyArray)
				#self.predictMove[self.index] = cnn_main(numpyArray)
				numpyArray = np.array([])

			# index 5 is standing still
			if self.predictMove[self.index] == 5:
				self.predictMove[self.index] = None

			if self.predictMove[0] == self.predictMove[1] == self.predictMove[2] == self.predictMove[3] == self.predictMove[4]:
				self.currMove = self.predictMove[0]
				self.continuePredict = false

	def run(self):
		SECRET_KEY = bytes("dancedancedance!", 'utf8')
		# setup connection
		print('Connecting to server')
		self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.s.connect((self.ipaddress, self.port))
		print("Connected to server " +self.ipaddress+ ", port: " +str(self.port))
		self.lastMsgTime = time.time()

		while True:
			self.machine()
			self.createMsg()

			# send msg at an interval of 3s
			if self.message and (self.lastMsgTime is None or time.time() - self.lastMsgTime > 3):
				print(self.message)
				iv = Random.new().read(AES.block_size)
				cipher = AES.new(SECRET_KEY, AES.MODE_CBC, iv)
				padMessage = pad(self.message, AES.block_size)
				encryptMsg = cipher.encrypt(padMessage)
				encodedMsg = base64.b64encode(iv + encryptMsg)
				self.lastMsgTime = time.time()

				time.sleep(5)
				self.s.send(encodedMsg)

			#if self.currMove == 10: (ending move)
				#break
		self.s.close()

class DataReceiveClass(Thread):
	#global dataQueue

	def __init__(self, ser):
		Thread.__init__(self)
		self.ser = ser

	def run(self):
		self.readData()

	def readData(self):
		global voltage
		global current
		global power
		global cumPower
		global numpyArray

		packet = self.ser.readline().decode()
		packet = packet.strip()
		print(packet)

		checkSum = packet.rsplit(",", 1)[1]
		packet = packet.rsplit(",", 1)[0]

		checkList = bytearray(packet.encode())
		testSum = 0

		for x in range(len(packet)):
			testSum ^= checkList[x]

		if(testSum == int(checkSum)):
			self.ser.write(NACK)
		else:
			self.ser.write(ACK)

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
				numpyArray = np.append(numpyArray, dataList)
				print("curr: ", numpyArray)
		Timer(0.001, self.readData).start()


if __name__ == '__main__':

	if len(sys.argv) != 3:
		print('Invalid number of arguments')
		print('python3 piClient1.py [IP address] [Port]')
		sys.exit()

	SerComm = SerClass()
	#SerComm.run()
	serThread = Thread(target=SerComm.run)

	SocketComm = SocketClass(sys.argv[1], sys.argv[2])
	#SocketComm.run()
	socketThread = Thread(target=SocketComm.run)

	serThread.start()
	socketThread.start()
