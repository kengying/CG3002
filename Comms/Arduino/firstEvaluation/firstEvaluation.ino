#include <Arduino_FreeRTOS.h>
#include <semphr.h> 
#define STACK_SIZE    200

// Create Sempaphore
SemaphoreHandle_t xSemaphore;

typedef struct DataPacket{
  float sensor1ID[6];
  float sensor2ID[6];
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
  
  //startHandshake();
  
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
      /* Hand Values */
      data.sensor1ID[0] = 1;
      data.sensor1ID[1] = 2;
      data.sensor1ID[2] = 3;
      data.sensor1ID[3] = 4;
      data.sensor1ID[4] = 5;
      data.sensor1ID[5] = 6;

      /* Elbow Values */
      data.sensor2ID[0] = -1;
      data.sensor2ID[1] = -2;
      data.sensor2ID[2] = -3;
      data.sensor2ID[3] = -4;
      data.sensor2ID[4] = -5;
      data.sensor2ID[5] = -6;
      
}

void battValues() {
  Serial.println("Reading batt");
  data.batt[0] = 1;
  data.batt[1] = 2;
  data.batt[2] = 3;
  data.batt[3] = 4;
}

void packageData() {
  Serial.println("Packing and Send Data");
  //Clear the dataBuffer
  memset(dataBuffer, 0, sizeof(dataBuffer));
  char floatChar[0];
  for(int i = 0; i < 6; i++){
    dtostrf(data.sensor1ID[i], 3, 2, floatChar);
    strcat(dataBuffer, floatChar);
    strcat(dataBuffer, ",");
    Serial2.write(floatChar);
    Serial2.write(",");
  }
  for(int i = 0; i < 6; i++){
    dtostrf(data.sensor2ID[i], 3, 2, floatChar);
    strcat(dataBuffer, floatChar);
    strcat(dataBuffer, ",");
    Serial2.write(floatChar);
    Serial2.write(",");
  }
  for(int i = 0; i < 4; i++){
    dtostrf(data.sensor2ID[i], 3, 2, floatChar);
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
  } 
}
