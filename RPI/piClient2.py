import socket
import serial
import queue as q
from threading import Timer,Thread,Event
import time
import csv
import random
import sys
import numpy as np

from PredictMove import predictMain
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

class PiClass():
	HELLO = ('H').encode()
	ACK = ('A').encode()
	NACK = ('N').encode()
	READY = ('R').encode()
	YES = ('Y').encode()

	voltage = 0
	current = 0
	power = 0
	cumPower = 0
	numpyArray = np.array([])
	
	currMove = None
	message = None
	lastMsgTime = None
	
	def __init__(self, IPaddress, PORT):
		self.ipaddress = IPaddress
		self.port = int(PORT)

	def setup(self):
		self.ser = serial.Serial("/dev/serial0", 115200)
		self.ser.reset_input_buffer()
		self.ser.reset_output_buffer()

	def handshake(self):
		print("InitiateHandshake")
		self.ser.write(self.HELLO)
		time.sleep(1)
		if self.ser.in_waiting > 0:
			reply = self.ser.read().decode()
			if(reply == 'A'):
				self.ser.write(self.ACK)
				print('Handshake Complete')
				return True
			else:
				print('pending')
		return False

	def readData(self):
		self.ser.reset_input_buffer()
		self.ser.reset_output_buffer()
		continueReceiveData = True
		self.ser.write(self.YES)
		
		while continueReceiveData:
			if self.ser.in_waiting > 0:
				packet = self.ser.readline().decode()
				packet = packet.strip()
				#print(packet)

				checkSum = packet.rsplit(",", 1)[1]
				packet = packet.rsplit(",", 1)[0]

				checkList = bytearray(packet.encode())
				testSum = 0

				for x in range(len(packet)):
					testSum ^= checkList[x]

				if(testSum == int(checkSum)):
					self.ser.write(self.NACK)
				else:
					dataList = []
					for x in range (0, 18):
						if x==0 or x==7:
							continue
						elif x==14:
							self.voltage = float(packet.split(',', 18)[x])
						elif x==15:
							self.current = float(packet.split(',', 18)[x])
						elif x==16:
							self.power = float(packet.split(',', 18)[x])
						elif x==17:
							self.cumPower = float(packet.split(',', 18)[x])
						else:
							val = float(packet.split(',', 18)[x])
							dataList.append(val)
					#print(numpyArray)
					if self.numpyArray.size == 0:
						self.numpyArray = np.append(self.numpyArray, dataList)
					else:
						self.numpyArray = np.vstack([self.numpyArray, dataList])
			
			else:
				self.ser.write(self.YES)
			
			if len(self.numpyArray) > 127:
				self.ser.write(self.NACK)
				continueReceiveData = False

	def createMsg(self):
		actions = ['handmotor', 'bunny', 'tapshoulder', 'rocket', 'cowboy', 'hunchback', 'jamesbond', 'chicken', 'movingsalute', 'whip', 'logout']

		if self.currMove == None:
			self.message = None
		else:
			self.message = ("#" + actions[self.currMove] + "|" + str(format(self.voltage, '.2f')) + "|" + str(format(self.current, '.2f')) + "|" + str(format(self.power, '.2f')) + "|" + str(format(self.cumPower, '.2f')) + "|").encode('utf8').strip()
			self.currMove = None

	def run(self):
		self.setup()
		while self.handshake() is False:
			continue
		
		SECRET_KEY = bytes("dancedancedance!", 'utf8')
		# setup connection
		print('Connecting to server')
		self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.s.connect((self.ipaddress, self.port))
		print("Connected to server " +self.ipaddress+ ", port: " +str(self.port))
		self.lastMsgTime = time.time()

		while True:
			
			# ML code that will return an index
			self.currMove=None
			continuePredict = True
			count = 0
			predictIndex = [0,0,0,0,0]
			
			while(continuePredict):

				self.readData()
								
				#print(numpyArray)
				print("run ML")
				temp = predictMain(numpyArray)
				#self.temp = cnn_main(numpyArray)
				#self.temp = 1
				#print(self.temp)

				# index 5 is standing still
				if temp == 5:
					continue
				else:
					predictIndex[temp] += 1

				count += 1
				#print(self.count)
				self.numpyArray = np.array([])

				# check prediction accuracy every 3 times
				if count >= 3:
					for x in range (0, 5):
						if predictIndex[x] > 1:
							self.currMove = x
					#print(self.currMove)
					predictIndex = [0,0,0,0,0]
					count = 0
					continuePredict = False
			
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

				self.s.send(encodedMsg)
				#time.sleep(5)

			#if self.currMove == 10: (ending move)
				#break
		#self.s.close()

if __name__ == '__main__':

	if len(sys.argv) != 3:
		print('Invalid number of arguments')
		print('python3 piClient1.py [IP address] [Port]')
		sys.exit()

	piComm = PiClass(sys.argv[1], sys.argv[2])
	piComm.run()
