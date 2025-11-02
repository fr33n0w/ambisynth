# 🎵 Ambient Drone Synthesizer (432Hz)

A Python-based multi-voice synthesizer for creating continuous, soothing ambient drones. Specially tuned to 432Hz for a harmonious, relaxing sound. Works on desktop and **Termux** (Android).

## Features

- **Three Voice Types:**
  - **Drone**: Rich harmonic drone with multiple overtones
  - **Pad**: Slowly evolving, detuned pad sounds
  - **White Noise**: Filtered noise for texture

- **432Hz Tuning**: All frequencies based on A=432Hz for natural harmony
- **Organic Modulation**: Multiple LFOs create slowly evolving, breathing sounds
- **Stereo Processing**: Spatial effects and delay for immersive soundscapes
- **Real-time Control**: Interactive controls to shape your ambient space

## Installation

### Quick Install (All Platforms)

```bash
pip install -r requirements.txt
python ambient_drone_synth.py
```

### Termux (Android) Installation

1. **Install Termux** from F-Droid (recommended) or Google Play

2. **Open Termux and run:**
```bash
# Update packages
pkg update

# Install Python and audio libraries
pkg install python portaudio libsndfile build-essential pkg-config -y

# Install Python packages
pip install numpy sounddevice

# Download and run the synthesizer
python ambient_drone_synth.py
```

### Alternative Setup Script

```bash
chmod +x setup.sh
./setup.sh
```

## Usage

Run the synthesizer:
```bash
python ambient_drone_synth.py
```

### Controls

- `q` - Quit the program
- `r` - Randomize harmonies (change chord relationships)
- `+` - Increase volume
- `-` - Decrease volume
- `d` - Toggle drone voice on/off
- `p` - Toggle pad voice on/off
- `n` - Toggle noise voice on/off

## Technical Details

### Audio Architecture

- **Sample Rate**: 44100 Hz
- **Buffer Size**: 256 samples (low latency)
- **Output**: Stereo (2 channels)

### Synthesis Methods

1. **Drone Voice**:
   - 8 harmonics with decreasing amplitude
   - Frequency modulation (0.03 Hz)
   - Amplitude modulation (0.05 Hz)
   - Soft clipping for warmth

2. **Pad Voice**:
   - 3 detuned oscillators
   - Mixed sine and triangle waves
   - Very slow modulation (0.01-0.015 Hz)
   - Simple low-pass filtering

3. **White Noise**:
   - Gaussian white noise generator
   - Modulated low-pass filter
   - Breathing amplitude envelope

### 432Hz Tuning System

The synthesizer uses just intonation ratios based on A=432Hz:
- Root (1:1)
- Fifth (3:2)
- Fourth (4:3)
- Major Third (5:4)
- Minor Third (6:5)

## Troubleshooting

### Termux Issues

1. **No Sound Output**:
   - Make sure headphones/speakers are connected
   - Check Termux has audio permissions
   - Try: `termux-setup-storage` to ensure permissions

2. **Import Error (sounddevice)**:
   - Install portaudio: `pkg install portaudio`
   - Reinstall sounddevice: `pip install --force-reinstall sounddevice`

3. **Performance Issues**:
   - Reduce buffer size in code (BUFFER_SIZE)
   - Close other apps to free up resources

### Desktop Issues

1. **No Audio Device Found**:
   - Check system audio settings
   - Install portaudio: 
     - Linux: `sudo apt-get install portaudio19-dev`
     - Mac: `brew install portaudio`

2. **Latency/Glitches**:
   - Increase BUFFER_SIZE to 512 or 1024
   - Close other audio applications

## Customization

### Change Base Frequency
Edit line in `ambient_drone_synth.py`:
```python
BASE_FREQ_432 = 432.0  # Change to 440.0 for standard tuning
```

### Adjust Voice Levels
Modify amplitude values in voice constructors:
```python
self.drone.params.amplitude = 0.25  # Drone volume
self.pad.params.amplitude = 0.2     # Pad volume  
self.noise.params.amplitude = 0.05  # Noise volume
```

### Modulation Rates
Adjust LFO frequencies for faster/slower evolution:
```python
self.freq_lfo = LFO(frequency=0.03, amplitude=0.002)  # Slower = lower frequency
```

## Theory & Philosophy

This synthesizer is tuned to **432Hz**, which many believe creates more harmonious and relaxing tones compared to the standard 440Hz tuning. The slow modulations and harmonic relationships are designed to create a meditative, calming atmosphere perfect for:

- Meditation and mindfulness
- Sleep and relaxation
- Focus and concentration
- Sound healing
- Ambient music production

The continuous drone nature means there are no sudden changes or disruptions, allowing the mind to settle into a peaceful state.

## License

MIT License - Feel free to modify and use as you wish!

## Credits

Created with 💜 for the ambient music and meditation community.
Special optimization for Termux/mobile devices.

---

*"In the drone, we find the eternal present moment"*
