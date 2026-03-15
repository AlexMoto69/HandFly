#define PPM_PIN 3 
#define NUM_CHANNELS 6
#define FRAME_LENGTH 20000 
#define PULSE_WIDTH 400    

// Default safe values
int channels[6] = {1511, 1511, 1005, 1511, 1000, 1000};
unsigned long lastMessageTime = 0;

void setup() {
  pinMode(PPM_PIN, OUTPUT);
  digitalWrite(PPM_PIN, HIGH); 
  Serial.begin(115200);
  Serial.setTimeout(5);
}

void loop() {
  // 1. Listen for Python/Serial commands: "R:1500 P:1500 T:1100 Y:1500"
  if (Serial.available() > 0) {
    String msg = Serial.readStringUntil('\n');
    int tr, tp, tt, ty;
    if (sscanf(msg.c_str(), "R:%d P:%d T:%d Y:%d", &tr, &tp, &tt, &ty) == 4) {
      channels[0] = constrain(tr, 1000, 2000);
      channels[1] = constrain(tp, 1000, 2000);
      channels[2] = constrain(tt, 1000, 2000);
      channels[3] = constrain(ty, 1000, 2000);
      lastMessageTime = millis();
    }
  }

  // 2. Failsafe: Reset to neutral if no message received for 500ms
  if (millis() - lastMessageTime > 500) {
    channels[0] = 1511; // Roll center
    channels[1] = 1511; // Pitch center
    channels[2] = 1005; // Throttle min
    channels[3] = 1511; // Yaw center
  }

  // 3. The "Verified" PPM Timing Loop
  unsigned long elapsed_time = 0;
  for (int i = 0; i < NUM_CHANNELS; i++) {
    digitalWrite(PPM_PIN, LOW);   
    delayMicroseconds(PULSE_WIDTH);
    digitalWrite(PPM_PIN, HIGH);  
    delayMicroseconds(channels[i] - PULSE_WIDTH);
    elapsed_time += channels[i];
  }

  unsigned long sync_gap = FRAME_LENGTH - elapsed_time;
  digitalWrite(PPM_PIN, LOW);
  delayMicroseconds(PULSE_WIDTH);
  digitalWrite(PPM_PIN, HIGH);
  delayMicroseconds(sync_gap - PULSE_WIDTH);
}
