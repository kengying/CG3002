# Changing the actions in self.actions should automatically change the script to function with the new number of moves.

# Developed and improved by past CG3002 TAs and students: Tean Zheng Yang, Jireh Tan, Boyd Anderson,  Paul Tan, Bernard Tan Ke Xuan, Ashley Ong



import os

import random

import socket

import sys

import threading

import time

from tkinter import Label, Tk



import numpy as np

import pandas as pd



from server_auth import server_auth





class Server(threading.Thread):

    def __init__(self, ip_addr, port_num):

        threading.Thread.__init__(self)

        self.shutdown = threading.Event()



        # init server

        self.auth = server_auth()

        # Create a TCP/IP socket

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Bind the socket to the port

        server_address = (ip_addr, port_num)

        print('starting up on %s port %s' % server_address)

        self.sock.bind(server_address)

        # Listen for incoming connections

        self.sock.listen(1)

        self.actions = ['handmotor', 'handmotor', 'handmotor', 'handmotor',

                        'bunny', 'bunny', 'bunny', 'bunny',

                        'tapshoulders', 'tapshoulders', 'tapshoulders', 'tapshoulders',

                        'rocket', 'rocket', 'rocket', 'rocket',

                        'cowboy', 'cowboy', 'cowboy', 'cowboy']

        #           'hunchback', 'hunchback', 'hunchback', 'hunchback',

        #           'jamesbond', 'jamesbond', 'jamesbond', 'jamesbond',

        #           'chicken', 'chicken', 'chicken', 'chicken',

        #           'movingsalute', 'movingsalute', 'movingsalute', 'movingsalute',

        #           'whip', 'whip', 'whip', 'whip']



        self.n_moves = len(self.actions)

        self.indices = np.arange(self.n_moves)

        self.filename = "logServer.csv"

        self.columns = ['timestamp', 'action', 'goal', 'time_delta', 'correct', 'voltage', 'current', 'power',

                        'cumpower']

        self.df = pd.DataFrame(columns=self.columns)

        self.df = self.df.set_index('timestamp')

        self.action = None

        self.action_set_time = None

        self.x = 0

        self.timeout = 60

        self.no_response = False

        self.connection = None

        self.timer = None

        self.logout = False



    def run(self):

        random.shuffle(self.indices)



        self.timer = threading.Timer(self.timeout, self.get_action)

        self.timer.start()

        print("No actions for 60 seconds to give time to connect")



        # Wait for a connection

        print('waiting for a connection')

        self.connection, client_address = self.sock.accept()



        # secret_key = input("Enter the secret key: ")

        print("Enter the secret key: ")

        secret_key = sys.stdin.readline().strip()



        print('connection from', client_address)



        if len(secret_key) == 16 or len(secret_key) == 24 or len(secret_key) == 32:

            pass

        else:

            print("AES key must be either 16, 24, or 32 bytes long")

            self.stop()



        while not self.shutdown.is_set():

            data = self.connection.recv(1024)

            if data:

                try:

                    msg = data.decode("utf8")

                    decodedmsg = self.auth.decryptText(msg, secret_key)

#                    print(decodedmsg['action'])
#                    print(decodedmsg['voltage'])
#                    print(decodedmsg['current'])
#                    print(decodedmsg['power'])
#                    print(decodedmsg['cumpower'])

                    if decodedmsg['action'] == "logout":

                        self.logout = True

                        print("bye bye")

                        self.stop()

                    elif len(decodedmsg['action']) == 0:

                        pass

                    elif self.action is None:  # Ignore if no action has been set yet

                        pass

                    else:  # If action is available log it, and then...

                        self.no_response = False

                        self.log_move_made(decodedmsg['action'], decodedmsg['voltage'], decodedmsg['current'],

                                           decodedmsg['power'], decodedmsg['cumpower'])

                        print("{} :: {} :: {} :: {} :: {}".format(decodedmsg['action'], decodedmsg['voltage'],

                                                                  decodedmsg['current'], decodedmsg['power'],

                                                                  decodedmsg['cumpower']))

                        self.get_action()  # Get new action

                except Exception as e:

                    print(e)

            else:

                print('no more data from', client_address)

                self.stop()



    def stop(self):

        self.connection.close()

        self.shutdown.set()

        self.timer.cancel()



    def get_action(self):

        self.timer.cancel()

        if self.no_response:  # If no response was sent

            self.log_move_made("None", 0, 0, 0, 0)

            print("ACTION TIMEOUT")



        if self.x < self.n_moves:

            index = self.indices[self.x]

        else:

            index = self.n_moves - 1

        self.action = self.actions[index]

        self.x += 1

        self.action_set_time = time.time()

        print("NEW ACTION :: {}".format(self.action))

        self.timer = threading.Timer(self.timeout, self.get_action)

        self.no_response = True

        self.timer.start()



    def log_move_made(self, action_made, voltage, current, power, cumpower):

        file = "log" + str(groupID) + ".csv"

        if not os.path.isfile(file):

            with open(file, 'w') as f:

                self.df.to_csv(f)

        with open(file, 'a') as f:

            data = dict()

            data['timestamp'] = time.time()

            data['action'] = action_made

            data['goal'] = self.action

            data['time_delta'] = data['timestamp'] - self.action_set_time

            data['voltage'] = voltage

            data['current'] = current

            data['power'] = power

            data['cumpower'] = cumpower

            data['correct'] = (self.action == action_made)

            self.df = pd.DataFrame(data, index=[0])[self.columns].set_index('timestamp')

            self.df.to_csv(f, header=False)





if __name__ == '__main__':

    if len(sys.argv) != 4:

        print('Invalid number of arguments')

        print('python server.py [IP address] [Port] [groupID]')

        sys.exit()



    ip_addr = sys.argv[1]

    port_num = int(sys.argv[2])

    groupID = sys.argv[3]



    # IP address = 'x.x.x.x'

    # Port = 8888



    my_server = Server(ip_addr, port_num)

    my_server.start()



    # Create action display window

    display_window = Tk()

    display_label = Label(display_window, text=str(my_server.action))

    display_label.config(font=('times', 130, 'bold'))

    display_label.pack(expand=True)

    display_window.update()



    while my_server.x <= my_server.n_moves + 1 and not my_server.shutdown.is_set():  # Display new task

        if my_server.x == my_server.n_moves + 1:

            display_label.config(text=str(my_server.x) + ":" + 'logout')

            if my_server.logout is True:

                break

        else:

            display_label.config(text=str(my_server.x) + ":" + str(my_server.action))

        display_window.update()

        time.sleep(0.2)


