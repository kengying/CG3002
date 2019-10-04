#include <Wire.h>

const int MPU1 = 0x68, MPU2 = 0x69; // MPU6050 I2C addresses
const int SEL_2G_200D = 0x0, SEL_4G_500D = 0x8, SEL_8G_1000D = 0x10, SEL_16G_2000D = 0x18;
const float CONV_2G = 16384.0, CONV_4G = 8192.0, CONV_8G = 4096.0, CONV_16G = 2048.0;
const float CONV_200D = 131.0, CONV_500D = 65.5, CONV_1000D = 32.8, CONV_2000D = 16.4;

struct SensorData {
  float accX;  // Accelerometer x-axis
  float accY;  // Accelerometer y-axis
  float accZ;  // Accelerometer z-axis
  float gX; // Gyroscope x-axis
  float gY; // Gyroscope y-axis
  float gZ; // Gyroscope z-axis
} MPU1Data, MPU2Data;

void setup() {  
  Wire.begin();                 // Initiates I2C communication
  Wire.beginTransmission(MPU1); // Slave address
  Wire.write(0x6B);             // Access the power management register
  Wire.write(0x00);             // Wakes up the MPU
  Wire.endTransmission(true);
  
  Wire.beginTransmission(MPU1); // Slave address
  Wire.write(0x1B);             // Access Gyroscope Scale Register
  Wire.write(SEL_8G_1000D);     // Set Scale to +-1000deg/s
  Wire.endTransmission(true);   // Communication done

  Wire.beginTransmission(MPU1); // Slave address
  Wire.write(0x1C);             // Access Accelerometer Scale Register
  Wire.write(SEL_4G_500D);      // Set Scale to +-4g
  Wire.endTransmission(true);   // Communication done
  
  Wire.begin();                 // Initiates I2C communication
  Wire.beginTransmission(MPU2); // Slave address
  Wire.write(0x6B);             // Access the power management register
  Wire.write(0x00);             // Wakes up the MPU
  Wire.endTransmission(true);   // Communication done

  Wire.beginTransmission(MPU2); // Slave address
  Wire.write(0x1B);             // Access Gyroscope Scale Register
  Wire.write(SEL_8G_1000D);     // Set Scale to +-1000deg/s
  Wire.endTransmission(true);   // Communication done

  Wire.beginTransmission(MPU2); // Slave address
  Wire.write(0x1C);             // Access Accelerometer Scale Register
  Wire.write(SEL_4G_500D);      // Set Scale to +-4g
  Wire.endTransmission(true);   // Communication done
  Serial.begin(9600);           // Intialize serial port baud rate to 9600 
}

SensorData getDataFromMPU(const int MPU, SensorData data) {
  Wire.beginTransmission(MPU);      // Begin communication
  Wire.write(0x3B);                 // Register 0x3B (ACCEL_XOUT_H) 
  Wire.endTransmission(false);      // End communication
  Wire.requestFrom(MPU, 14, true);  // Request 14 registers

  // Combines the upper 8 bits and the lower 8 bits to form 16 bits
  data.accX  = Wire.read() << 8 | Wire.read(); // 0x3B (ACCEL_XOUT_H) & 0x3C (ACCEL_XOUT_L)    
  data.accY  = Wire.read() << 8 | Wire.read(); // 0x3D (ACCEL_YOUT_H) & 0x3E (ACCEL_YOUT_L)
  data.accZ  = Wire.read() << 8 | Wire.read(); // 0x3F (ACCEL_ZOUT_H) & 0x40 (ACCEL_ZOUT_L)
  Wire.read() << 8 | Wire.read(); // 0x41 (TEMP_OUT_H) & 0x42 (TEMP_OUT_L) To clear Temp from buffer.
  data.gX = Wire.read() << 8 | Wire.read(); // 0x43 (GYRO_XOUT_H) & 0x44 (GYRO_XOUT_L)
  data.gY = Wire.read() << 8 | Wire.read(); // 0x45 (GYRO_YOUT_H) & 0x46 (GYRO_YOUT_L)
  data.gZ = Wire.read() << 8 | Wire.read(); // 0x47 (GYRO_ZOUT_H) & 0x48 (GYRO_ZOUT_L)

  data = translateValues(data);

  return data;
}
//void PrintMPUValues() {
//  Serial.print("%d,%d"); Serial.print(SensorData.AccX);
//  Serial.print("Accelerometer Values: [x = "); Serial.print(SensorData.AccX);
//  Serial.print(", y = "); Serial.print( SensorData.AccY);
//  Serial.print(", z = "); Serial.print(SensorData.AccZ); Serial.println("]"); 
//  Serial.print("Gyrorometer Values:   [x = "); Serial.print(SensorData.GyroX);
//  Serial.print(", y = "); Serial.print(SensorData.GyroY);
//  Serial.print(", z = "); Serial.print(SensorData.GyroZ); Serial.println("]");
//  Serial.println();
//  delay(1000);
//}

void printData() {

  Serial.print(MPU1Data.accX); Serial.print(","); Serial.print(MPU1Data.accY); Serial.print(","); 
  Serial.print(MPU1Data.accZ); Serial.print(","); Serial.print(MPU1Data.gX); Serial.print(",");
  Serial.print(MPU1Data.gY); Serial.print(",");  Serial.print(MPU1Data.gZ); Serial.print(",");

      Serial.print(MPU2Data.accX); Serial.print(","); Serial.print(MPU2Data.accY); Serial.print(","); 
  Serial.print(MPU2Data.accZ); Serial.print(","); Serial.print(MPU2Data.gX); Serial.print(",");
  Serial.print(MPU2Data.gY); Serial.print(",");  Serial.print(MPU2Data.gZ);

  Serial.println();
}


SensorData translateValues(SensorData data) {
  data.accX = data.accX/CONV_4G;
  data.accY = data.accY/CONV_4G;
  data.accZ = data.accZ/CONV_4G;
  data.gX = data.gX/CONV_1000D;
  data.gY = data.gY/CONV_1000D;
  data.gZ = data.gZ/CONV_1000D;

  return data;
//  data.Temp = (data.Temp/340.00) + 36.53;
}

void loop() {
  MPU1Data = getDataFromMPU(MPU1, MPU1Data);
  MPU2Data = getDataFromMPU(MPU2, MPU2Data);
  Serial.print(MPU1Data.accX); Serial.print(" "); Serial.println(MPU1Data.accY);
  printData();
  delay(50);
}
