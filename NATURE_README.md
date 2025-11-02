# 🌿 Nature Noise Synthesizer

An evolving, organic noise synthesizer that creates realistic nature soundscapes using white, pink, and brown noise with advanced filtering and modulation. Creates continuous, soothing ambient environments that automatically evolve over time.

## 🎧 Sound Scenes

### Pre-configured Environments:
1. **Ocean** 🌊 - Rolling waves with foam and spray
2. **Forest** 🌲 - Birds chirping with gentle ambience  
3. **Rain** 🌧️ - Rainfall with varying intensity
4. **Fire** 🔥 - Crackling campfire with pops
5. **Wind** 💨 - Gusting wind with whistling
6. **Cave** 🕳️ - Deep, resonant ambience
7. **Night** 🌙 - Quiet nighttime atmosphere
8. **Storm** ⛈️ - Dramatic mix of wind, rain, and waves

## Features

### Sound Generation
- **Multiple Noise Types:**
  - White noise (rain, spray)
  - Pink noise (wind, ambience)
  - Brown/Red noise (ocean, rumble)
  - Blue noise (brightness, air)

- **Advanced Processing:**
  - Resonant multi-mode filters (lowpass, highpass, bandpass, notch)
  - ADSR envelopes with looping
  - Multiple LFOs per voice
  - Stereo field positioning
  - Reverb for spaciousness

### Organic Evolution
- **Auto-Evolution Mode:** Scenes gradually change every 20-40 seconds
- **Parameter Drift:** Filter frequencies, resonance, and amplitudes slowly shift
- **Natural Modulation:** Complex LFOs create non-repetitive, breathing sounds

### Specialized Generators

#### Ocean Waves
- Brown noise base for deep rumble
- Rhythmic wave cycles
- High-frequency foam at wave peaks
- Variable intensity

#### Wind
- Pink noise through resonant filters
- Random gust generation
- Whistling overtones
- Dynamic intensity changes

#### Rain
- Dense white noise
- Variable density/intensity
- High-frequency droplet sounds
- Subtle spatial movement

#### Fire
- Brown noise rumble base
- Crackling white noise bursts
- Random "pop" generation
- Flickering amplitude

#### Birds
- Filtered noise bursts
- Randomized chirp timing
- Frequency sweeps
- Natural spacing

## Installation

```bash
# Same requirements as the ambient drone synth
pip install numpy sounddevice

# Run the synthesizer
python nature_noise_synth.py
```

## Usage

### Keyboard Controls

- **Number Keys (1-8):** Switch between nature scenes
- **r:** Randomize scene (create unique mix)
- **e:** Manually evolve current scene
- **a:** Toggle auto-evolution on/off
- **+/-:** Increase/decrease volume
- **q:** Quit

## Technical Details

### Architecture
- **Sample Rate:** 44.1kHz
- **Buffer Size:** 256 samples
- **Channels:** Stereo
- **Processing:** Real-time with ~6ms latency

### Noise Generation Algorithms

#### Pink Noise (1/f noise)
Uses Voss-McCartney algorithm with 16 generators for accurate spectral characteristics

#### Brown Noise (1/f² noise)  
Integration of white noise with slight decay to prevent DC drift

#### Blue Noise (f noise)
High-pass filtered white noise for bright, airy textures

### Filter Design
State Variable Filter (SVF) implementation providing:
- Simultaneous lowpass, highpass, bandpass, and notch outputs
- Stable resonance up to self-oscillation
- Smooth modulation without clicks

### Evolution System
1. **Timer-based triggers:** Every 20-40 seconds
2. **Parameter interpolation:** Smooth transitions over ~3 seconds
3. **Bounded randomization:** Changes stay within musical ranges
4. **Scene memory:** Can return to preset scenes

## Customization

### Adjust Evolution Rate
```python
# In NatureSynthesizer.__init__()
self.evolution_interval = SAMPLE_RATE * random.uniform(10, 20)  # Faster evolution
```

### Create Custom Scenes
```python
# Add to scene_presets dictionary
NatureScene.CUSTOM: {
    'ocean': 0.2, 'wind': 0.3, 'rain': 0.2,
    'fire': 0.0, 'birds': 0.1, 'ambience': 0.2
}
```

### Modify Voice Character
```python
# Adjust filter cutoffs and resonance
self.ocean.filter1.cutoff = 300  # Deeper ocean
self.wind.whistle_filter.resonance = 4.0  # More pronounced whistling
```

## Use Cases

- **Sleep & Relaxation:** Ocean, rain, and cave scenes
- **Focus & Concentration:** Forest, rain, and gentle wind
- **Meditation:** Any scene with auto-evolution off
- **Sound Masking:** White/pink noise dominant scenes
- **ASMR & Ambience:** Fire crackling, gentle rain
- **Sound Design:** Use as a base for music production

## Performance

- **CPU Usage:** ~5-10% on modern processors
- **Memory:** ~50MB RAM
- **Latency:** <10ms typical
- **Mobile-friendly:** Optimized for Termux/Android

## Theory

The synthesizer uses psychoacoustic principles:
- **Pink noise** matches natural environmental spectra
- **Slow modulation** (0.01-0.3 Hz) creates calming, non-intrusive movement
- **Stereo imaging** provides spatial immersion without fatigue
- **Filtered noise** can effectively simulate natural textures

## License

MIT License - Free to use and modify

---

*"In noise, we find the patterns of nature"*
