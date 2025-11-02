#!/usr/bin/env python3
"""
Ambient Drone Synthesizer
A multi-voice synthesizer for creating continuous ambient drones with 432Hz tuning
Works on Termux and other platforms
"""

import numpy as np
import sounddevice as sd
import threading
import time
import random
from dataclasses import dataclass
from typing import Optional
import sys

# Audio configuration
SAMPLE_RATE = 44100
BUFFER_SIZE = 256
CHANNELS = 2  # Stereo output

# 432Hz tuning base frequencies (A=432Hz)
# Using just intonation ratios for a more harmonic sound
BASE_FREQ_432 = 432.0
FREQ_RATIOS = {
    'root': 1.0,           # A
    'fifth': 3/2,          # E
    'fourth': 4/3,         # D
    'major_third': 5/4,    # C#
    'minor_third': 6/5,    # C
    'octave_down': 0.5,    # A (lower)
    'octave_down2': 0.25,  # A (2 octaves lower)
}

@dataclass
class VoiceParams:
    """Parameters for each voice in the synthesizer"""
    amplitude: float = 0.3
    frequency: float = BASE_FREQ_432
    phase: float = 0.0
    pan: float = 0.5  # 0=left, 1=right
    active: bool = True
    
class LFO:
    """Low Frequency Oscillator for modulation"""
    def __init__(self, frequency=0.1, amplitude=1.0, phase=0.0):
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase
        self.sample_count = 0
        
    def generate(self, num_samples):
        """Generate LFO samples"""
        t = np.arange(num_samples) / SAMPLE_RATE + self.sample_count / SAMPLE_RATE
        self.sample_count += num_samples
        # Use multiple sine waves for more organic modulation
        primary = np.sin(2 * np.pi * self.frequency * t + self.phase)
        secondary = np.sin(2 * np.pi * self.frequency * 2.1 * t) * 0.3
        tertiary = np.sin(2 * np.pi * self.frequency * 0.47 * t) * 0.2
        return (primary + secondary + tertiary) * self.amplitude

class DroneVoice:
    """Drone voice with rich harmonics"""
    def __init__(self, base_freq=BASE_FREQ_432/2):
        self.params = VoiceParams(frequency=base_freq, amplitude=0.25)
        self.phase = 0.0
        self.harmonic_amps = [1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05]
        
        # Slow modulation for organic movement
        self.freq_lfo = LFO(frequency=0.03, amplitude=0.002)
        self.amp_lfo = LFO(frequency=0.05, amplitude=0.1, phase=np.pi/3)
        self.pan_lfo = LFO(frequency=0.02, amplitude=0.3, phase=np.pi/2)
        
    def generate(self, num_samples):
        """Generate drone samples with harmonics"""
        # Frequency modulation
        freq_mod = 1.0 + self.freq_lfo.generate(num_samples)
        frequencies = self.params.frequency * freq_mod
        
        # Generate harmonics
        signal = np.zeros(num_samples)
        for i, amp in enumerate(self.harmonic_amps):
            harmonic = i + 1
            t = np.arange(num_samples) / SAMPLE_RATE
            phase_increment = 2 * np.pi * frequencies * harmonic * t + self.phase * harmonic
            signal += np.sin(phase_increment) * amp
            
        # Update phase for continuity
        last_freq = frequencies[-1] if isinstance(frequencies, np.ndarray) else frequencies
        self.phase += 2 * np.pi * last_freq * num_samples / SAMPLE_RATE
        self.phase = self.phase % (2 * np.pi)
        
        # Normalize and apply amplitude envelope
        signal = signal / len(self.harmonic_amps)
        amp_mod = 1.0 + self.amp_lfo.generate(num_samples)
        signal *= self.params.amplitude * amp_mod
        
        # Apply soft clipping for warmth
        signal = np.tanh(signal * 0.7) / 0.7
        
        return signal

class PadVoice:
    """Pad voice with slow evolving textures"""
    def __init__(self, base_freq=BASE_FREQ_432/4):
        self.params = VoiceParams(frequency=base_freq, amplitude=0.2)
        self.phase1 = 0.0
        self.phase2 = 0.0
        self.phase3 = 0.0
        
        # Very slow modulation for pad evolution
        self.freq_lfo = LFO(frequency=0.01, amplitude=0.001)
        self.amp_lfo = LFO(frequency=0.015, amplitude=0.15)
        self.detune_lfo = LFO(frequency=0.008, amplitude=0.002)
        
    def generate(self, num_samples):
        """Generate pad samples with detuned oscillators"""
        freq_mod = 1.0 + self.freq_lfo.generate(num_samples)
        detune_mod = self.detune_lfo.generate(num_samples)
        
        # Three slightly detuned oscillators for thickness
        freq1 = self.params.frequency * freq_mod
        freq2 = self.params.frequency * freq_mod * (1.002 + detune_mod)
        freq3 = self.params.frequency * freq_mod * (0.998 - detune_mod)
        
        t = np.arange(num_samples) / SAMPLE_RATE
        
        # Generate three oscillators with different waveforms
        osc1 = np.sin(2 * np.pi * freq1 * t + self.phase1)
        osc2 = np.sin(2 * np.pi * freq2 * t + self.phase2) * 0.7
        # Add subtle triangle wave for texture
        osc3_phase = 2 * np.pi * freq3 * t + self.phase3
        osc3 = 2/np.pi * np.arcsin(np.sin(osc3_phase)) * 0.5
        
        # Update phases
        last_freq1 = freq1[-1] if isinstance(freq1, np.ndarray) else freq1
        last_freq2 = freq2[-1] if isinstance(freq2, np.ndarray) else freq2
        last_freq3 = freq3[-1] if isinstance(freq3, np.ndarray) else freq3
        
        self.phase1 += 2 * np.pi * last_freq1 * num_samples / SAMPLE_RATE
        self.phase2 += 2 * np.pi * last_freq2 * num_samples / SAMPLE_RATE
        self.phase3 += 2 * np.pi * last_freq3 * num_samples / SAMPLE_RATE
        self.phase1 = self.phase1 % (2 * np.pi)
        self.phase2 = self.phase2 % (2 * np.pi)
        self.phase3 = self.phase3 % (2 * np.pi)
        
        # Mix oscillators
        signal = (osc1 + osc2 + osc3) / 2.5
        
        # Apply amplitude modulation
        amp_mod = 1.0 + self.amp_lfo.generate(num_samples)
        signal *= self.params.amplitude * amp_mod
        
        # Low-pass filter simulation (simple moving average)
        window_size = 5
        signal = np.convolve(signal, np.ones(window_size)/window_size, mode='same')
        
        return signal

class NoiseVoice:
    """Filtered white noise voice for texture"""
    def __init__(self):
        self.params = VoiceParams(amplitude=0.05)
        self.filter_cutoff = 500  # Hz
        self.resonance = 0.5
        
        # LFOs for noise modulation
        self.amp_lfo = LFO(frequency=0.04, amplitude=0.3)
        self.filter_lfo = LFO(frequency=0.02, amplitude=0.4)
        
        # Simple IIR filter state
        self.prev_input = 0.0
        self.prev_output = 0.0
        
    def generate(self, num_samples):
        """Generate filtered noise"""
        # Generate white noise
        noise = np.random.normal(0, 0.1, num_samples)
        
        # Apply amplitude modulation for breathing effect
        amp_mod = 1.0 + self.amp_lfo.generate(num_samples)
        noise *= self.params.amplitude * amp_mod
        
        # Simple low-pass filter with modulated cutoff
        filter_mod = 1.0 + self.filter_lfo.generate(num_samples)
        
        # Apply simple RC low-pass filter
        filtered = np.zeros(num_samples)
        
        for i in range(num_samples):
            # Calculate cutoff for this sample
            cutoff = self.filter_cutoff * filter_mod[i]
            rc = 1.0 / (2 * np.pi * cutoff)
            dt = 1.0 / SAMPLE_RATE
            alpha = dt / (rc + dt)
            
            if i > 0:
                filtered[i] = filtered[i-1] + alpha * (noise[i] - filtered[i-1])
            else:
                filtered[i] = self.prev_output + alpha * (noise[i] - self.prev_output)
        
        self.prev_output = filtered[-1] if num_samples > 0 else self.prev_output
        
        # Add subtle resonance
        filtered = np.tanh(filtered * (1 + self.resonance * 2))
        
        return filtered

class AmbientDroneSynth:
    """Main synthesizer class"""
    def __init__(self):
        self.is_running = False
        self.stream = None
        
        # Initialize voices with harmonic relationships
        root_freq = BASE_FREQ_432 / 4  # Start 2 octaves below A432
        
        self.drone = DroneVoice(root_freq * FREQ_RATIOS['root'])
        self.pad = PadVoice(root_freq * FREQ_RATIOS['fifth'])
        self.noise = NoiseVoice()
        
        # Master output level
        self.master_volume = 0.7
        
        # Reverb-like delay buffer for spaciousness
        self.delay_buffer_size = int(SAMPLE_RATE * 0.1)  # 100ms
        self.delay_buffer_l = np.zeros(self.delay_buffer_size)
        self.delay_buffer_r = np.zeros(self.delay_buffer_size)
        self.delay_index = 0
        self.delay_feedback = 0.4
        self.delay_mix = 0.3
        
    def audio_callback(self, outdata, frames, time_info, status):
        """Callback function for audio stream"""
        if status:
            print(f"Audio callback status: {status}")
        
        # Generate samples from each voice
        drone_signal = self.drone.generate(frames)
        pad_signal = self.pad.generate(frames)
        noise_signal = self.noise.generate(frames)
        
        # Mix voices
        mixed = drone_signal + pad_signal + noise_signal
        
        # Apply simple stereo spreading
        left = mixed.copy()
        right = mixed.copy()
        
        # Pan drone slightly left, pad slightly right
        left *= 1.1
        right *= 0.9
        
        # Add delay/reverb effect
        for i in range(frames):
            # Read from delay buffer
            delay_pos = (self.delay_index - int(SAMPLE_RATE * 0.05) + i) % self.delay_buffer_size
            delayed_l = self.delay_buffer_l[delay_pos] * self.delay_mix
            delayed_r = self.delay_buffer_r[delay_pos] * self.delay_mix
            
            # Add delayed signal
            left[i] += delayed_r * 0.5  # Cross-feed for width
            right[i] += delayed_l * 0.5
            
            # Update delay buffer with feedback
            write_pos = (self.delay_index + i) % self.delay_buffer_size
            self.delay_buffer_l[write_pos] = left[i] + delayed_l * self.delay_feedback
            self.delay_buffer_r[write_pos] = right[i] + delayed_r * self.delay_feedback
        
        self.delay_index = (self.delay_index + frames) % self.delay_buffer_size
        
        # Apply master volume and soft limiting
        left = np.tanh(left * self.master_volume) * 0.9
        right = np.tanh(right * self.master_volume) * 0.9
        
        # Write to output
        outdata[:, 0] = left
        outdata[:, 1] = right
        
    def start(self):
        """Start the synthesizer"""
        if not self.is_running:
            try:
                self.stream = sd.OutputStream(
                    samplerate=SAMPLE_RATE,
                    blocksize=BUFFER_SIZE,
                    channels=CHANNELS,
                    callback=self.audio_callback,
                    dtype='float32'
                )
                self.stream.start()
                self.is_running = True
                print("🎵 Ambient Drone Synthesizer Started (432Hz tuning)")
                print("=" * 50)
                print("Controls:")
                print("  'q' - Quit")
                print("  'r' - Randomize harmonies")
                print("  '+' - Increase volume")
                print("  '-' - Decrease volume")
                print("  'd' - Toggle drone")
                print("  'p' - Toggle pad")
                print("  'n' - Toggle noise")
                print("=" * 50)
                print("Creating soothing ambient drones...")
                
            except Exception as e:
                print(f"Error starting audio stream: {e}")
                sys.exit(1)
    
    def stop(self):
        """Stop the synthesizer"""
        if self.is_running and self.stream:
            self.stream.stop()
            self.stream.close()
            self.is_running = False
            print("\n🛑 Synthesizer stopped")
    
    def randomize_harmonies(self):
        """Randomly change the harmonic relationships"""
        # Choose random harmonic intervals for variety
        intervals = list(FREQ_RATIOS.values())
        root_freq = BASE_FREQ_432 / 4
        
        self.drone.params.frequency = root_freq * random.choice(intervals)
        self.pad.params.frequency = root_freq * random.choice(intervals) * 0.5
        
        # Randomize some modulation rates for evolution
        self.drone.freq_lfo.frequency = random.uniform(0.02, 0.05)
        self.pad.amp_lfo.frequency = random.uniform(0.01, 0.03)
        
        print(f"🎲 New harmonies: Drone={self.drone.params.frequency:.1f}Hz, Pad={self.pad.params.frequency:.1f}Hz")
    
    def adjust_volume(self, delta):
        """Adjust master volume"""
        self.master_volume = max(0.1, min(1.0, self.master_volume + delta))
        print(f"🔊 Volume: {int(self.master_volume * 100)}%")
    
    def toggle_voice(self, voice_name):
        """Toggle a voice on/off"""
        if voice_name == 'drone':
            current = self.drone.params.amplitude
            self.drone.params.amplitude = 0.25 if current == 0 else 0
            state = "ON" if self.drone.params.amplitude > 0 else "OFF"
            print(f"🎹 Drone: {state}")
        elif voice_name == 'pad':
            current = self.pad.params.amplitude
            self.pad.params.amplitude = 0.2 if current == 0 else 0
            state = "ON" if self.pad.params.amplitude > 0 else "OFF"
            print(f"🎹 Pad: {state}")
        elif voice_name == 'noise':
            current = self.noise.params.amplitude
            self.noise.params.amplitude = 0.05 if current == 0 else 0
            state = "ON" if self.noise.params.amplitude > 0 else "OFF"
            print(f"🎹 Noise: {state}")

def main():
    """Main function"""
    synth = AmbientDroneSynth()
    
    try:
        # For Termux compatibility, check if we can use keyboard input
        try:
            import termios
            import tty
            
            def get_key():
                """Get single keypress (Unix/Termux)"""
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.setraw(sys.stdin.fileno())
                    key = sys.stdin.read(1)
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                return key
                
            use_keyboard = True
        except ImportError:
            # Fallback for systems without termios
            use_keyboard = False
            print("Note: Real-time keyboard control not available. Using simple input.")
    
        synth.start()
        
        if use_keyboard:
            # Real-time keyboard control
            while True:
                key = get_key()
                
                if key == 'q':
                    break
                elif key == 'r':
                    synth.randomize_harmonies()
                elif key == '+':
                    synth.adjust_volume(0.1)
                elif key == '-':
                    synth.adjust_volume(-0.1)
                elif key == 'd':
                    synth.toggle_voice('drone')
                elif key == 'p':
                    synth.toggle_voice('pad')
                elif key == 'n':
                    synth.toggle_voice('noise')
        else:
            # Simple input loop for compatibility
            while True:
                try:
                    cmd = input("\nEnter command (q/r/+/-/d/p/n): ").strip().lower()
                    if cmd == 'q':
                        break
                    elif cmd == 'r':
                        synth.randomize_harmonies()
                    elif cmd == '+':
                        synth.adjust_volume(0.1)
                    elif cmd == '-':
                        synth.adjust_volume(-0.1)
                    elif cmd == 'd':
                        synth.toggle_voice('drone')
                    elif cmd == 'p':
                        synth.toggle_voice('pad')
                    elif cmd == 'n':
                        synth.toggle_voice('noise')
                except KeyboardInterrupt:
                    break
                    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        synth.stop()
        print("Thank you for using the Ambient Drone Synthesizer! 🎵")

if __name__ == "__main__":
    main()
