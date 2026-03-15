#define PPM_PIN 3 
#define NUM_CHANNELS 6
#define FRAME_LENGTH 20000  // From your screenshot
#define PULSE_WIDTH 400     // From your screenshot

// Use the idle values the user suggested (900-1000)
int channels[6] = {1500, 1500, 1000, 1500, 1500, 1500};

void setup() {
  pinMode(PPM_PIN, OUTPUT);
  digitalWrite(PPM_PIN, HIGH); // Start High for Negative Polarity
}

void loop() {
  unsigned long elapsed_time = 0;

  for (int i = 0; i < NUM_CHANNELS; i++) {
    // 1. The Pulse (Negative Polarity means the "spike" goes LOW)
    digitalWrite(PPM_PIN, LOW);   
    delayMicroseconds(PULSE_WIDTH);

    // 2. The Data (The signal stays HIGH for the duration of the channel)
    digitalWrite(PPM_PIN, HIGH);  
    delayMicroseconds(channels[i] - PULSE_WIDTH);
    
    elapsed_time += channels[i];
  }

  // 3. The Sync Gap to finish the 20ms frame
  unsigned long sync_gap = FRAME_LENGTH - elapsed_time;
  digitalWrite(PPM_PIN, LOW);
  delayMicroseconds(PULSE_WIDTH);
  digitalWrite(PPM_PIN, HIGH);
  delayMicroseconds(sync_gap - PULSE_WIDTH);
}