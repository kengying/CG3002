#include <Arduino_FreeRTOS.h>
#include <semphr.h> 
#define STACK_SIZE    200

#include <Wire.h>

// ACC
const int MPU1 = 0x68, MPU2 = 0x69; // MPU6050 I2C addresses
const int SEL_2G_200D = 0x0, SEL_4G_500D = 0x8, SEL_8G_1000D = 0x10, SEL_16G_2000D = 0x18;
const float CONV_2G = 16384.0, CONV_4G = 8192.0, CONV_8G = 4096.0, CONV_16G = 2048.0;
const float CONV_200D = 131.0, CONV_500D = 65.5, CONV_1000D = 32.8, CONV_2000D = 16.4;

// Power

const float RS_FAT = 0.10;          // Shunt, RS resistor value (in ohms)
const float RL_INTERNAL = 10.0;     // RL value (in kilo??? ohms)

const float VCC = 5.0;              // Reference voltage for analog read
const int I_PIN = A0;               // Input pin for measuring Vout
const int V_DIVIDER_PIN = A1;       // Input pin for measuring 50% of V

float analogCurrent;
float analogVoltageDivider;

unsigned long startTime = 0;
unsigned long previousTime = 0;
unsigned long currentTime = 0;

float current = 0.00;
float voltage = 0.00;

float totalEnergy = 0.00;
float power = 0.00;



// Create Sempaphore
SemaphoreHandle_t xSemaphore;

struct SensorData {
  float accX;  // Accelerometer x-axis
  float accY;  // Accelerometer y-axis
  float accZ;  // Accelerometer z-axis
  float gX; // Gyroscope x-axis
  float gY; // Gyroscope y-axis
  float gZ; // Gyroscope z-axis
} MPU1Data, MPU2Data;

typedef struct DataPacket{
  int sensor1ID;
  float sensor1[6];
  int sensor2ID;
  float sensor2[6];
  float batt[4];
} DataPacket;

/* Packet Declaration */
DataPacket data;

int reply;
int previousReply = 0;

boolean handshake = false;
boolean recieved = false;
boolean dontSend = true;



char dataBuffer[1000];

//Checksum
int checksum = 0;
int checksum2;
char checksum_c[4];

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

  // Power
  startTime = currentTime = previousTime = millis();

  pinMode(I_PIN, INPUT);
  pinMode(V_DIVIDER_PIN, INPUT);
  
  Serial.begin(115200);
  Serial1.begin(115200);
//  Serial.println("SET UP");
  
 startHandshake();
  
  xSemaphore = xSemaphoreCreateBinary();
  if ( ( xSemaphore ) != NULL ) {
    xSemaphoreGive( (xSemaphore) );
  }
  
//  Serial.println("CREATE TASK");
  xTaskCreate(initRun, "initRun", 200, NULL, 2, NULL);
}


void loop() {
}

void startHandshake() {
  while (!recieved) {
    Serial.println("start handshake");
    reply = Serial1.read();
    if (reply == 'H' ){
      Serial1.write('A');
      Serial.println("Ack Handshake");
      recieved = true;
      reply = 0;
    } 
  }
  while (recieved  && !handshake) {
    reply = Serial1.read();
    if (reply == 'A') {
 //     Serial.println("Handshake complete");
      //delay(500);
      handshake = true;
    }
  }
}

/**
 * Task to start the run.
 */
void initRun(void *p){
  TickType_t xLastWakeTime = xTaskGetTickCount();
  const TickType_t xFrequency = 100;
  
  while (1) {
    TickType_t xCurrWakeTime = xTaskGetTickCount();

    
    reply = Serial1.read();
    if(reply == 'Y'){
      previousReply = reply;
      sensorValues();
      battValues();  
      packageData();
    } else if(reply == 'N'){
      previousReply = reply;
       /** do nothing **/
    } else if(previousReply == 'Y'){
        sensorValues();
        battValues();  
        packageData();
    } else if (previousReply == 'N'){
      /**do nothing**/
    }
    
//    Serial.println();
    // 30 ms interval, ~30 samples/s
    vTaskDelayUntil(&xCurrWakeTime, 20/portTICK_PERIOD_MS);
  }
}

SensorData getDataFromMPU(const int MPU, SensorData datain) {
 //   Serial.println("Get Data From MPU Run");

  Wire.beginTransmission(MPU);      // Begin communication
  Wire.write(0x3B);                 // Register 0x3B (ACCEL_XOUT_H) 
  Wire.endTransmission(false);      // End communication
  Wire.requestFrom(MPU, 14, true);  // Request 14 registers

  // Combines the upper 8 bits and the lower 8 bits to form 16 bits
  datain.accX  = Wire.read() << 8 | Wire.read(); // 0x3B (ACCEL_XOUT_H) & 0x3C (ACCEL_XOUT_L)    
  datain.accY  = Wire.read() << 8 | Wire.read(); // 0x3D (ACCEL_YOUT_H) & 0x3E (ACCEL_YOUT_L)
  datain.accZ  = Wire.read() << 8 | Wire.read(); // 0x3F (ACCEL_ZOUT_H) & 0x40 (ACCEL_ZOUT_L)
  Wire.read() << 8 | Wire.read(); // 0x41 (TEMP_OUT_H) & 0x42 (TEMP_OUT_L) To clear Temp from buffer.
  datain.gX = Wire.read() << 8 | Wire.read(); // 0x43 (GYRO_XOUT_H) & 0x44 (GYRO_XOUT_L)
  datain.gY = Wire.read() << 8 | Wire.read(); // 0x45 (GYRO_YOUT_H) & 0x46 (GYRO_YOUT_L)
  datain.gZ = Wire.read() << 8 | Wire.read(); // 0x47 (GYRO_ZOUT_H) & 0x48 (GYRO_ZOUT_L)

  datain = translateValues(datain);
  
  return datain;
}

SensorData translateValues(SensorData dataTranslate) {
 // Serial.println("Translate Run");
  dataTranslate.accX = dataTranslate.accX/CONV_4G;
  dataTranslate.accY = dataTranslate.accY/CONV_4G;
  dataTranslate.accZ = dataTranslate.accZ/CONV_4G;
  dataTranslate.gX = dataTranslate.gX/CONV_1000D;
  dataTranslate.gY = dataTranslate.gY/CONV_1000D;
  dataTranslate.gZ = dataTranslate.gZ/CONV_1000D;

  return dataTranslate;
//  data.Temp = (data.Temp/340.00) + 36.53;
}

/**
 *  Read Value
 */
void sensorValues() {
   // Serial.println("Read Sensor 1 Run");
      MPU1Data = getDataFromMPU(MPU1, MPU1Data);
      /** 
       * Hand Values 
       * Accelerometer 
       * -> x: 0, y: 1, z: 2
       * Gyroscope 
       * -> x: 3, y: 4, z: 5
       */
      data.sensor1ID = 1;
      data.sensor1[0] = MPU1Data.accX; 
      data.sensor1[1] = MPU1Data.accY;
      data.sensor1[2] = MPU1Data.accZ;
      data.sensor1[3] = MPU1Data.gX;
      data.sensor1[4] = MPU1Data.gY;
      data.sensor1[5] = MPU1Data.gZ;
      
    //    Serial.println("Read Sensor 2 Run");

      MPU2Data = getDataFromMPU(MPU2, MPU2Data);
      /** 
       * Elbow Values
       * Accelerometer 
       * -> x: 0, y: 1, z: 2
       * Gyroscope 
       * -> x: 3, y: 4, z: 5
       */
      data.sensor2ID = 2;
      data.sensor2[0] = MPU2Data.accX; 
      data.sensor2[1] = MPU2Data.accY;
      data.sensor2[2] = MPU2Data.accZ;
      data.sensor2[3] = MPU2Data.gX;
      data.sensor2[4] = MPU2Data.gY;
      data.sensor2[5] = MPU2Data.gZ;
//
//      Serial.print(MPU1Data.accX); Serial.print(","); Serial.print(MPU1Data.accY); Serial.print(","); 
//  Serial.print(MPU1Data.accZ); Serial.print(","); Serial.print(MPU1Data.gX); Serial.print(",");
//  Serial.print(MPU1Data.gY); Serial.print(",");  Serial.print(MPU1Data.gZ); Serial.print(",");
//
//      Serial.print(MPU2Data.accX); Serial.print(","); Serial.print(MPU2Data.accY); Serial.print(","); 
//  Serial.print(MPU2Data.accZ); Serial.print(","); Serial.print(MPU2Data.gX); Serial.print(",");
//  Serial.print(MPU2Data.gY); Serial.print(",");  Serial.print(MPU2Data.gZ);
//
//  Serial.println();
      
}

// Voltage-Current-Power-Energy
void getVIPE() {
// Read & Compute Current
  analogCurrent = analogRead(A0);
  current = (( analogCurrent * VCC / 1023)) / ( RS_FAT * RL_INTERNAL );

  // Read & Compute Voltage
  analogVoltageDivider = analogRead(A1);
  voltage = 2.0 * (analogVoltageDivider * VCC ) / 1023;

  //Calculate energy and power
  unsigned long timeElapsedSinceLastCycle = (currentTime - previousTime);  // in ms.
  unsigned long timeElapsedSinceStart = (currentTime - startTime);

  totalEnergy = totalEnergy + (voltage * current * (timeElapsedSinceLastCycle / 3600000.0)); //in Wh
  power = totalEnergy / (timeElapsedSinceStart/3600000.0) ; //in W
  
  previousTime = currentTime;
  }

void printVIPE() {
  Serial.print("--- ");
  Serial.print(currentTime);
  Serial.print("ms ---");
  Serial.println();
  Serial.print(current, 3);
  Serial.print(" A | ");
  Serial.print(voltage, 3);
  Serial.println(" V");
  Serial.print(totalEnergy, 5);
  Serial.print(" Wh || ");
  Serial.print(power, 5);
  Serial.println(" W");
}

void battValues() {
  /** 
       * Batt Values
       * batt[0]: voltage
       * batt[1]: current
       * batt[2]: power
       * batt[3]: cumpower
       * TBD: calculation for cumpower
  */

  currentTime = millis();
  getVIPE();
  //printVIPE();
 // Serial.println("Reading batt");
  data.batt[0] = voltage;
  data.batt[1] = current;
  data.batt[2] = power;
  data.batt[3] = totalEnergy;
}

void packageData() {
//  Serial.println("Packing and Send Data");
  //Clear the dataBuffer
  memset(dataBuffer, 0, sizeof(dataBuffer));
  char floatChar[0];
  
  dtostrf(data.sensor1ID, 0, 0, floatChar);
  strcat(dataBuffer, floatChar);
  strcat(dataBuffer, ",");
  Serial1.write(floatChar);
  Serial1.write(",");
  for(int i = 0; i < 6; i++){
    dtostrf(data.sensor1[i], 3, 2, floatChar);
    strcat(dataBuffer, floatChar);
    strcat(dataBuffer, ",");
    Serial1.write(floatChar);
    Serial1.write(",");
  }
  
  dtostrf(data.sensor2ID, 0, 0, floatChar);
  strcat(dataBuffer, floatChar);
  strcat(dataBuffer, ",");
  Serial1.write(floatChar);
  Serial1.write(",");
  for(int i = 0; i < 6; i++){
    dtostrf(data.sensor2[i], 3, 2, floatChar);
    strcat(dataBuffer, floatChar);
    strcat(dataBuffer, ",");
    Serial1.write(floatChar);
    Serial1.write(",");
  }
  for(int i = 0; i < 4; i++){
    dtostrf(data.batt[i], 3, 2, floatChar);
    strcat(dataBuffer, floatChar);
    //if(i != 3){
    strcat(dataBuffer, ",");
    //}
    Serial1.write(floatChar);
    Serial1.write(",");
  }
  //Append the checksum
  int counter = strlen(dataBuffer);
  checksum = 0;
  for (int j = 0; j < counter; j++) {
    checksum ^= dataBuffer[j];
//    Serial.println(dataBuffer[j]);
//    
//    Serial.print(bitRead(dataBuffer[j], 0));
//    Serial.print(bitRead(dataBuffer[j], 1));
//    Serial.print(bitRead(dataBuffer[j], 2));
//    Serial.println(bitRead(dataBuffer[j], 3));  
//    Serial.print(bitRead(checksum, 0));
//    Serial.print(bitRead(checksum, 1));
//    Serial.print(bitRead(checksum, 2));
//    Serial.println(bitRead(checksum, 3));  
  }
  checksum2 = (int)checksum;
  memset(checksum_c, 0, sizeof(checksum_c));
  itoa(checksum2, checksum_c, 10);
  Serial1.write(checksum_c);
  Serial1.write("\n");
//  Serial.print(dataBuffer);
//  Serial.println(checksum_c);
}
 
void checkAck() {
  if (reply == 'Y'){
    dontSend = false;
  } else if (reply == 'N'){
    dontSend = true;
    while(dontSend){
      reply = Serial1.read();
      if (reply == 'Y' ){ 
        dontSend = false;
      }
    }
  }
} 
