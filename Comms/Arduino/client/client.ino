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

boolean handshake = false;
boolean recieved = false;


char dataBuffer[1000];
void setup() {
  Serial.begin(115200);
  Serial2.begin(115200);
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
  int reply;
  while (!recieved) {
    Serial.println("start handshake");
    reply = Serial2.read();
    if (reply == 'H' ){
      Serial2.write('A');
      Serial.println("Ack Handshake");
      recieved = true;
      reply = 0;
    } 
  }
  while (recieved  && !handshake) {
    reply = Serial2.read();
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
  Serial2.write(floatChar);
  Serial2.write(",");
  for(int i = 0; i < 6; i++){
    dtostrf(data.sensor1[i], 3, 2, floatChar);
    strcat(dataBuffer, floatChar);
    strcat(dataBuffer, ",");
    Serial2.write(floatChar);
    Serial2.write(",");
  }
  
  dtostrf(data.sensor2ID, 0, 0, floatChar);
  strcat(dataBuffer, floatChar);
  strcat(dataBuffer, ",");
  Serial2.write(floatChar);
  Serial2.write(",");
  for(int i = 0; i < 6; i++){
    dtostrf(data.sensor2[i], 3, 2, floatChar);
    strcat(dataBuffer, floatChar);
    strcat(dataBuffer, ",");
    Serial2.write(floatChar);
    Serial2.write(",");
  }
  for(int i = 0; i < 4; i++){
    dtostrf(data.batt[i], 3, 2, floatChar);
    strcat(dataBuffer, floatChar);
    if(i != 3){
    strcat(dataBuffer, ",");
    }
    Serial2.write(floatChar);
    Serial2.write(",");
  }
  int counter = strlen(dataBuffer);
  char checksum[16];
  itoa(counter, checksum, 10);
  Serial2.write(checksum);
  Serial2.write("\n");
  Serial.print(dataBuffer);
  Serial.println(checksum);
  
}

void checkAck() {
  Serial.println("Check ACK");
  int reply = Serial2.read();
  while (reply == 'N'){
      Serial.println("Resending Package");
      packageData();
      reply = Serial2.read();
  } 
}
