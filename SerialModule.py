"""
Serial Comm Module

-Install
sudo apt-get install python3-serial

-Allow read and write access to port
sudo chmod 666 /dev/ttyACsudo chmod 666 /dev/ttyACM0M0

"""

import serial


def initConnection(portNo, baudRate):
    """
    :return: Initialzed Serial Object
    """
    try:
        ser = serial.Serial(portNo, baudRate)
        print("Device Connected ")
        return ser
    except:
        print("Not Connected ")
        pass


def sendData(se, data, digits):

    myString = "$"
    for d in data:
        myString += str(int(d)).zfill(digits)

    try:
        se.write(myString.encode())
        print(myString)
    except:
        print("Data Transmission Failed ")


def getData():

    data = ser.readline()
    data = data.decode("utf-8")
    data = data.split('#')
    dataList = []
    [dataList.append(d) for d in data]
    return dataList[:-1]

def main():

    data = getData()  # Get List of Arduino Value
    print(data[0])    # Print First value which is potentiometer
    sendData(ser, [data[0], 0], 4) # Send the Pot Value back

if __name__ == "__main__":
    # Connect to Arduino Device
    ser = initConnection('/dev/ttyACM0', 9600)
    while True:
        main()
