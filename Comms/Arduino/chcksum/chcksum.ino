#include <Arduino_FreeRTOS.h>
#include <semphr.h> 
#define STACK_SIZE    200

// Create Sempaphore
SemaphoreHandle_t xSemaphore;

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

boolean handshake = false;
boolean recieved = false;
boolean resendingPckage = false;

char dataBuffer[1000];

//Checksum
int checksum = 0;
int checksum2;
char checksum_c[4];

void setup() {
  Serial.begin(115200);
  Serial1.begin(115200);
  Serial.println("SET UP");
  
  startHandshake();
  
  xSemaphore = xSemaphoreCreateBinary();
  if ( ( xSemaphore ) != NULL ) {
    xSemaphoreGive( (xSemaphore) );
  }
  
  Serial.println("CREATE TASK");
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
      Serial.println("Handshake complete");
      delay(500);
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

    sensorValues();
    battValues();  
    packageData();
    checkAck();
    
    Serial.println();
    // 30 ms interval, ~30 samples/s
    vTaskDelayUntil(&xCurrWakeTime, 30/portTICK_PERIOD_MS);
  }
}
/**
 *  Read Value
 */
void sensorValues() {
      /** 
       * Hand Values 
       * Accelerometer 
       * -> x: 0, y: 1, z: 2
       * Gyroscope 
       * -> x: 3, y: 4, z: 5
       */
      data.sensor1ID = 1;
      data.sensor1[0] = 1; 
      data.sensor1[1] = 2;
      data.sensor1[2] = 3;
      data.sensor1[3] = 4;
      data.sensor1[4] = 5.1;
      data.sensor1[5] = 6.1;
      /** 
       * Elbow Values
       * Accelerometer 
       * -> x: 0, y: 1, z: 2
       * Gyroscope 
       * -> x: 3, y: 4, z: 5
       */
      data.sensor2ID = 2;
      data.sensor2[0] = -1;
      data.sensor2[1] = -2;
      data.sensor2[2] = -3;
      data.sensor2[3] = -4;
      data.sensor2[4] = -5;
      data.sensor2[5] = -6;
      
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
  Serial.println("Reading batt");
  data.batt[0] = 1.1;
  data.batt[1] = 2.1;
  data.batt[2] = 3.11;
  data.batt[3] = 4.1;
}

void packageData() {
  Serial.println("Packing and Send Data");
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
    if(i != 3){
    strcat(dataBuffer, ",");
    }
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
  Serial.print(dataBuffer);
  Serial.println(checksum_c);
  if(resendingPckage == true)
    checkAck();
}

void checkAck() {
  Serial.println("Check ACK");
  reply = Serial1.read();
  while (reply == 'N'){
      Serial.println("Resending Package");
      packageData();
      reply = Serial1.read();
      resendingPckage = true;
  } 
  if (reply == 'A'){
    resendingPckage = false;
  }
}
