const int LOOPDELAY = 500;          // Delay for programme loop in ms

const float RS_FAT = 0.10;          // Shunt, RS resistor value (in ohms)
const float RL_INTERNAL = 10.0;     // RL value (in kilo??? ohms)

const float VCC = 5.0;              // Reference voltage for analog read
const int I_PIN = A0;               // Input pin for measuring Vout
const int V_DIVIDER_PIN = A1;       // Input pin for measuring 50% of V

float analogCurrent;
float analogVoltageDivider;

unsigned long startTime;
unsigned long previousTime;
unsigned long currentTime;

float current = 0.00;
float voltage = 0.00;

float totalEnergy = 0.00;
float power = 0.00;

void setup() {
  // Initialize serial monitor
  Serial.begin(9600);
  startTime = currentTime = previousTime = millis();

  pinMode(I_PIN, INPUT);
  pinMode(V_DIVIDER_PIN, INPUT);
  
  Serial.print("=== START TIME : ");
  Serial.print(startTime);
  Serial.print("ms ===");
  Serial.println(" ");
}

void loop() {
  currentTime = millis();
  getVIPE();
  printVIPE();
  delay(LOOPDELAY);
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
  unsigned long timeElapsedSinceLastCycle = (currentTime - previousTime) / 3600000.0;  // in hour.
  unsigned long timeElapsedSinceStart = (currentTime - startTime) / 3600000.0;

  totalEnergy = totalEnergy + (voltage * current * timeElapsedSinceLastCycle); //in Wh
  power = totalEnergy / timeElapsedSinceStart ; //in W
  
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
