#!/usr/bin/env python3
"""
Nature Noise Synthesizer - 432Hz Tuned
A multi-layered noise synthesizer for creating evolving natural soundscapes
Simulates: Ocean, Wind, Rain, Fire, Forest, Birds
All resonant frequencies tuned to 432Hz harmonic series
"""

import numpy as np
import sounddevice as sd
import threading
import time
import random
from dataclasses import dataclass
from typing import Optional, List, Tuple
import sys
from enum import Enum

# Audio configuration
SAMPLE_RATE = 44100
BUFFER_SIZE = 512  # Increased for better performance
CHANNELS = 2  # Stereo output

# 432Hz Tuning Base (A = 432Hz)
BASE_FREQ_432 = 432.0
HARMONICS_432 = {
    'sub_bass': BASE_FREQ_432 / 8,      # 54Hz
    'bass': BASE_FREQ_432 / 4,          # 108Hz  
    'low_mid': BASE_FREQ_432 / 2,       # 216Hz
    'mid': BASE_FREQ_432,                # 432Hz
    'high_mid': BASE_FREQ_432 * 2,      # 864Hz
    'presence': BASE_FREQ_432 * 4,      # 1728Hz
    'brilliance': BASE_FREQ_432 * 8,    # 3456Hz
    'air': BASE_FREQ_432 * 16,          # 6912Hz
}

class NatureScene(Enum):
    """Different nature scenes with their characteristic parameters"""
    OCEAN = "ocean"
    FOREST = "forest"
    RAIN = "rain"
    FIRE = "fire"
    WIND = "wind"
    CAVE = "cave"
    NIGHT = "night"
    STORM = "storm"

class NoiseGenerator:
    """Generate different types of colored noise with smooth continuity"""
    
    def __init__(self, sample_rate=SAMPLE_RATE):
        self.sample_rate = sample_rate
        # Pink noise state - simplified
        self.pink_state = np.array([0.0, 0.0, 0.0])
        # Brown noise state
        self.brown_last = 0.0
        
        # Pre-generate white noise with overlap for continuity
        self.buffer_size = 8192
        self.white_buffer = np.random.normal(0, 0.5, self.buffer_size)  # Reduced amplitude
        self.buffer_index = 0
        
        # Crossfade buffer for smooth transitions
        self.crossfade_len = 64
        
    def white_noise(self, num_samples):
        """Generate white noise with smooth buffer transitions"""
        output = np.zeros(num_samples)
        
        samples_remaining = num_samples
        output_index = 0
        
        while samples_remaining > 0:
            # How many samples can we take from current position
            available = min(samples_remaining, self.buffer_size - self.buffer_index)
            
            # Copy samples
            output[output_index:output_index + available] = (
                self.white_buffer[self.buffer_index:self.buffer_index + available]
            )
            
            self.buffer_index += available
            output_index += available
            samples_remaining -= available
            
            # Regenerate buffer if needed
            if self.buffer_index >= self.buffer_size:
                # Crossfade with new buffer for continuity
                old_end = self.white_buffer[-self.crossfade_len:]
                self.white_buffer = np.random.normal(0, 0.5, self.buffer_size)
                
                # Crossfade the beginning of new buffer with end of old
                fade_in = np.linspace(0, 1, self.crossfade_len)
                fade_out = 1 - fade_in
                self.white_buffer[:self.crossfade_len] = (
                    old_end * fade_out + self.white_buffer[:self.crossfade_len] * fade_in
                )
                self.buffer_index = 0
                
        return output
    
    def pink_noise(self, num_samples):
        """Generate pink noise with simple filtering"""
        white = self.white_noise(num_samples)
        pink = np.zeros(num_samples)
        
        # Simple 3-stage pinking filter (less CPU intensive)
        for i in range(num_samples):
            # Update states with leaky integrators
            self.pink_state[0] = 0.99886 * self.pink_state[0] + white[i] * 0.0555179
            self.pink_state[1] = 0.99332 * self.pink_state[1] + white[i] * 0.0750759
            self.pink_state[2] = 0.96900 * self.pink_state[2] + white[i] * 0.1538520
            
            # Mix stages
            pink[i] = (self.pink_state[0] + self.pink_state[1] + 
                      self.pink_state[2] + white[i] * 0.3104856) * 0.2
            
            # Prevent drift
            if abs(pink[i]) > 2.0:
                self.pink_state *= 0.95
                
        return pink
    
    def brown_noise(self, num_samples):
        """Generate brown noise with drift prevention"""
        white = self.white_noise(num_samples) * 0.05  # Reduced input
        brown = np.zeros(num_samples)
        
        # Leaky integrator with DC blocking
        leak = 0.997
        dc_block = 0.999
        
        for i in range(num_samples):
            # Integrate with leak
            self.brown_last = self.brown_last * leak + white[i]
            
            # DC blocking to prevent drift
            self.brown_last = self.brown_last - (self.brown_last * (1 - dc_block))
            
            # Limit range
            self.brown_last = np.clip(self.brown_last, -1.0, 1.0)
            
            brown[i] = self.brown_last
            
        return brown * 0.7  # Scale down
    
    def blue_noise(self, num_samples):
        """Generate blue noise with differentiation"""
        white = self.white_noise(num_samples)
        # Simple high-pass filter for blue noise
        blue = np.zeros(num_samples)
        prev = 0.0
        
        for i in range(num_samples):
            blue[i] = white[i] - prev * 0.85
            prev = white[i]
            
        return blue * 0.5  # Scale down

class ResonantFilter:
    """Optimized resonant filter with click prevention"""
    
    def __init__(self, cutoff=1000, resonance=1.0, mode='lowpass'):
        self.cutoff = cutoff
        self.resonance = max(0.0, min(1.0, resonance))  # Further reduced resonance
        self.mode = mode
        
        # Biquad filter state
        self.x1 = 0.0
        self.x2 = 0.0
        self.y1 = 0.0
        self.y2 = 0.0
        
        # Coefficients
        self.a1 = 0.0
        self.a2 = 0.0
        self.b0 = 1.0
        self.b1 = 0.0
        self.b2 = 0.0
        
        # Smoothing for coefficient changes
        self.target_cutoff = cutoff
        self.current_cutoff = cutoff
        
        # DC blocker
        self.dc_x1 = 0.0
        self.dc_y1 = 0.0
        
        self.update_coefficients(cutoff)
        
    def update_coefficients(self, cutoff):
        """Update filter coefficients with stability checks"""
        # Clamp and smooth cutoff
        cutoff = max(30, min(15000, cutoff))
        
        # Nyquist safety
        nyquist = SAMPLE_RATE * 0.45
        cutoff = min(cutoff, nyquist)
        
        omega = 2.0 * np.pi * cutoff / SAMPLE_RATE
        sin_omega = np.sin(omega)
        cos_omega = np.cos(omega)
        
        # Lower Q for stability
        q = max(0.5, min(1.0, self.resonance))
        alpha = sin_omega / (2.0 * q)
        
        # Coefficient calculation with normalization
        if self.mode == 'lowpass':
            self.b0 = (1 - cos_omega) / 2
            self.b1 = 1 - cos_omega
            self.b2 = (1 - cos_omega) / 2
            a0 = 1 + alpha
            self.a1 = -2 * cos_omega / a0
            self.a2 = (1 - alpha) / a0
        elif self.mode == 'highpass':
            self.b0 = (1 + cos_omega) / 2
            self.b1 = -(1 + cos_omega)
            self.b2 = (1 + cos_omega) / 2
            a0 = 1 + alpha
            self.a1 = -2 * cos_omega / a0
            self.a2 = (1 - alpha) / a0
        elif self.mode == 'bandpass':
            self.b0 = alpha
            self.b1 = 0
            self.b2 = -alpha
            a0 = 1 + alpha
            self.a1 = -2 * cos_omega / a0
            self.a2 = (1 - alpha) / a0
        else:  # notch
            self.b0 = 1
            self.b1 = -2 * cos_omega
            self.b2 = 1
            a0 = 1 + alpha
            self.a1 = -2 * cos_omega / a0
            self.a2 = (1 - alpha) / a0
            
        # Normalize coefficients
        norm = a0
        self.b0 /= norm
        self.b1 /= norm
        self.b2 /= norm
        
    def process(self, input_signal, cutoff_mod=None):
        """Process with click prevention"""
        output = np.zeros_like(input_signal)
        
        # Smooth cutoff changes
        cutoff_smoothing = 0.995
        
        for i in range(len(input_signal)):
            # Smooth cutoff modulation
            if cutoff_mod is not None and i < len(cutoff_mod):
                self.target_cutoff = self.cutoff * cutoff_mod[i]
            else:
                self.target_cutoff = self.cutoff
                
            # Exponential smoothing of cutoff
            self.current_cutoff = (self.current_cutoff * cutoff_smoothing + 
                                   self.target_cutoff * (1 - cutoff_smoothing))
            
            # Update coefficients only when needed
            if i % 64 == 0:  # Update every 64 samples
                self.update_coefficients(self.current_cutoff)
            
            # Apply filter (Direct Form II)
            x = input_signal[i]
            y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2 - self.a1 * self.y1 - self.a2 * self.y2
            
            # Check for instability
            if np.isnan(y) or np.isinf(y) or abs(y) > 10.0:
                # Reset filter state
                self.x1 = self.x2 = self.y1 = self.y2 = 0.0
                y = 0.0
            
            # Update state
            self.x2 = self.x1
            self.x1 = x
            self.y2 = self.y1
            self.y1 = y
            
            # DC blocker (removes clicks from DC offset)
            dc_out = y - self.dc_x1 + 0.995 * self.dc_y1
            self.dc_x1 = y
            self.dc_y1 = dc_out
            
            # Soft clipping
            output[i] = np.tanh(dc_out * 0.5) * 2.0
                
        return output

class Envelope:
    """ADSR-style envelope with looping capability"""
    
    def __init__(self, attack=0.1, decay=0.2, sustain=0.7, release=0.3, loop=True):
        self.attack = attack
        self.decay = decay
        self.sustain = sustain
        self.release = release
        self.loop = loop
        self.phase = 0.0
        self.time = 0.0
        self.released = False
        
    def generate(self, num_samples):
        """Generate envelope values"""
        envelope = np.zeros(num_samples)
        dt = 1.0 / SAMPLE_RATE
        
        for i in range(num_samples):
            if not self.released:
                if self.time < self.attack:
                    # Attack phase
                    envelope[i] = self.time / self.attack if self.attack > 0 else 1.0
                elif self.time < self.attack + self.decay:
                    # Decay phase
                    decay_progress = (self.time - self.attack) / self.decay if self.decay > 0 else 1.0
                    envelope[i] = 1.0 - (1.0 - self.sustain) * decay_progress
                else:
                    # Sustain phase
                    envelope[i] = self.sustain
                    if self.loop and self.time > self.attack + self.decay + 2.0:
                        # Loop back to attack
                        self.time = 0.0
            else:
                # Release phase
                envelope[i] = self.sustain * np.exp(-3.0 * self.time)
                
            self.time += dt
            
        return envelope

class LFO:
    """Low Frequency Oscillator with multiple waveforms"""
    
    def __init__(self, frequency=0.1, amplitude=1.0, waveform='sine'):
        self.frequency = frequency
        self.amplitude = amplitude
        self.waveform = waveform
        self.phase = random.random() * 2 * np.pi  # Random start phase
        self.sample_count = 0
        
    def generate(self, num_samples):
        """Generate LFO samples"""
        t = np.arange(num_samples) / SAMPLE_RATE + self.sample_count / SAMPLE_RATE
        self.sample_count += num_samples
        
        if self.waveform == 'sine':
            # Complex sine for organic movement
            primary = np.sin(2 * np.pi * self.frequency * t + self.phase)
            secondary = np.sin(2 * np.pi * self.frequency * 3.7 * t) * 0.2
            tertiary = np.sin(2 * np.pi * self.frequency * 0.31 * t) * 0.3
            output = primary + secondary + tertiary
        elif self.waveform == 'triangle':
            output = 2 * np.arcsin(np.sin(2 * np.pi * self.frequency * t + self.phase)) / np.pi
        elif self.waveform == 'square':
            output = np.sign(np.sin(2 * np.pi * self.frequency * t + self.phase))
        elif self.waveform == 'random':
            # Smooth random using interpolation
            key_points = int(num_samples * self.frequency / 10) + 2
            random_points = np.random.randn(key_points)
            x = np.linspace(0, 1, key_points)
            xnew = np.linspace(0, 1, num_samples)
            output = np.interp(xnew, x, random_points)
        else:
            output = np.zeros(num_samples)
            
        return output * self.amplitude

class NatureVoice:
    """Individual nature sound layer"""
    
    def __init__(self, name="", noise_type='white', base_cutoff=1000, resonance=1.0):
        self.name = name
        self.noise_gen = NoiseGenerator()
        self.noise_type = noise_type
        
        # Filters
        self.filter1 = ResonantFilter(base_cutoff, resonance, 'lowpass')
        self.filter2 = ResonantFilter(base_cutoff * 2, resonance * 0.5, 'bandpass')
        
        # Envelopes
        self.amp_envelope = Envelope(0.5, 1.0, 0.6, 0.5, loop=True)
        
        # LFOs for modulation
        self.filter_lfo = LFO(0.1, 0.3, 'sine')
        self.amp_lfo = LFO(0.05, 0.2, 'sine')
        self.pan_lfo = LFO(0.03, 0.4, 'sine')
        
        # Voice parameters
        self.amplitude = 0.3
        self.active = True
        
    def generate(self, num_samples):
        """Generate nature voice samples"""
        if not self.active:
            return np.zeros(num_samples)
            
        # Generate base noise
        if self.noise_type == 'white':
            noise = self.noise_gen.white_noise(num_samples)
        elif self.noise_type == 'pink':
            noise = self.noise_gen.pink_noise(num_samples)
        elif self.noise_type == 'brown':
            noise = self.noise_gen.brown_noise(num_samples)
        elif self.noise_type == 'blue':
            noise = self.noise_gen.blue_noise(num_samples)
        else:
            noise = self.noise_gen.white_noise(num_samples)
            
        # Apply filter modulation
        filter_mod = 1.0 + self.filter_lfo.generate(num_samples)
        
        # Process through filters
        filtered1 = self.filter1.process(noise, filter_mod)
        filtered2 = self.filter2.process(noise, filter_mod * 1.5)
        
        # Mix filter outputs
        signal = filtered1 * 0.7 + filtered2 * 0.3
        
        # Apply envelopes and modulation
        amp_env = self.amp_envelope.generate(num_samples)
        amp_mod = 1.0 + self.amp_lfo.generate(num_samples)
        
        signal *= amp_env * amp_mod * self.amplitude
        
        return signal

class OceanWaves(NatureVoice):
    """Specialized ocean wave generator - 432Hz tuned"""
    
    def __init__(self):
        # Ocean tuned to low 432Hz harmonics for deep, calming waves
        super().__init__("Ocean", 'brown', HARMONICS_432['bass'], 0.5)  # 108Hz base
        
        # Wave-specific parameters
        self.wave_cycle = LFO(0.1, 1.0, 'sine')  # Main wave rhythm
        self.foam_gen = NoiseGenerator()
        # Foam at 432Hz presence frequency
        self.foam_filter = ResonantFilter(HARMONICS_432['presence'], 0.3, 'highpass')  # 1728Hz
        
        # Slower modulation for ocean
        self.filter_lfo = LFO(0.07, 0.4, 'sine')
        self.amp_lfo = LFO(0.12, 0.3, 'sine')
        
    def generate(self, num_samples):
        """Generate ocean wave sounds"""
        # Base wave sound (brown noise through low-pass)
        wave_base = self.noise_gen.brown_noise(num_samples)
        
        # Wave rhythm modulation
        wave_mod = (self.wave_cycle.generate(num_samples) + 1) * 0.5
        wave_mod = np.power(wave_mod, 2)  # Make waves more dramatic
        
        # Filter modulation for wave movement
        filter_mod = 1.0 + self.filter_lfo.generate(num_samples) * wave_mod
        
        # Main wave sound
        waves = self.filter1.process(wave_base, filter_mod)
        
        # Add foam/spray (white noise at wave peaks)
        foam = self.foam_gen.white_noise(num_samples) * 0.1
        foam = self.foam_filter.process(foam)
        foam *= np.maximum(0, wave_mod - 0.5) * 2  # Only at wave peaks
        
        # Combine wave and foam
        signal = waves + foam * 0.3
        
        # Apply amplitude modulation
        amp_mod = 1.0 + self.amp_lfo.generate(num_samples)
        signal *= self.amplitude * amp_mod * wave_mod
        
        return signal

class WindSound(NatureVoice):
    """Wind sound generator with gusts - 432Hz tuned"""
    
    def __init__(self):
        # Wind tuned to mid 432Hz frequencies
        super().__init__("Wind", 'pink', HARMONICS_432['mid'], 1.5)  # 432Hz base
        
        # Wind-specific parameters
        self.gust_lfo = LFO(0.03, 0.5, 'random')
        # Whistle at high-mid 432Hz harmonic
        self.whistle_filter = ResonantFilter(HARMONICS_432['high_mid'], 2.0, 'bandpass')  # 864Hz
        
        # Different modulation rates
        self.filter_lfo = LFO(0.15, 0.5, 'random')
        self.amp_lfo = LFO(0.08, 0.4, 'random')
        
    def generate(self, num_samples):
        """Generate wind sounds with occasional gusts"""
        # Base wind (pink noise)
        wind_base = self.noise_gen.pink_noise(num_samples)
        
        # Gust modulation
        gust_mod = np.abs(self.gust_lfo.generate(num_samples))
        gust_mod = np.power(gust_mod, 0.5)  # Smoother gusts
        
        # Filter modulation
        filter_mod = 1.0 + self.filter_lfo.generate(num_samples) + gust_mod
        
        # Process wind sound
        wind_low = self.filter1.process(wind_base, filter_mod)
        wind_whistle = self.whistle_filter.process(wind_base, filter_mod * 2)
        
        # Mix components
        signal = wind_low + wind_whistle * 0.2 * gust_mod
        
        # Apply amplitude with gusts
        amp_mod = 1.0 + self.amp_lfo.generate(num_samples)
        signal *= self.amplitude * amp_mod * (0.3 + 0.7 * gust_mod)
        
        return signal

class RainSound(NatureVoice):
    """Rain sound generator - 432Hz tuned"""
    
    def __init__(self):
        # Rain tuned to presence frequencies for clarity
        super().__init__("Rain", 'white', HARMONICS_432['presence'], 0.5)  # 1728Hz
        
        # Rain-specific parameters
        self.density = 0.5  # Rain density/intensity
        # Drops at brilliance frequency
        self.drop_filter = ResonantFilter(HARMONICS_432['brilliance'], 0.3, 'highpass')  # 3456Hz
        
        # Subtle modulation for rain
        self.filter_lfo = LFO(0.02, 0.1, 'sine')
        self.amp_lfo = LFO(0.03, 0.1, 'sine')
        
    def generate(self, num_samples):
        """Generate rain sounds"""
        # Dense white noise for rain
        rain = self.noise_gen.white_noise(num_samples) * self.density
        
        # Filter to create rain texture
        filter_mod = 1.0 + self.filter_lfo.generate(num_samples)
        rain_body = self.filter1.process(rain, filter_mod)
        rain_drops = self.drop_filter.process(rain, filter_mod)
        
        # Mix rain components
        signal = rain_body * 0.7 + rain_drops * 0.3
        
        # Subtle amplitude modulation
        amp_mod = 1.0 + self.amp_lfo.generate(num_samples)
        signal *= self.amplitude * amp_mod
        
        return signal

class FireSound(NatureVoice):
    """Fire crackling sound generator - 432Hz tuned"""
    
    def __init__(self):
        # Fire tuned to low-mid frequencies for warmth
        super().__init__("Fire", 'pink', HARMONICS_432['low_mid'], 0.7)  # 216Hz
        
        # Fire-specific parameters
        self.crackle_gen = NoiseGenerator()
        # Crackles at mid frequency
        self.crackle_filter = ResonantFilter(HARMONICS_432['mid'], 1.5, 'bandpass')  # 432Hz
        self.pop_threshold = 0.98
        
        # Flickering modulation
        self.filter_lfo = LFO(0.3, 0.4, 'random')
        self.amp_lfo = LFO(0.2, 0.3, 'random')
        
    def generate(self, num_samples):
        """Generate fire crackling sounds"""
        # Base fire rumble (brown noise)
        fire_base = self.noise_gen.brown_noise(num_samples) * 0.5
        
        # Crackling (filtered white noise with pops)
        crackle = self.crackle_gen.white_noise(num_samples)
        
        # Add random pops
        pops = np.random.random(num_samples)
        pop_mask = pops > self.pop_threshold
        crackle[pop_mask] *= 3.0
        
        # Filter modulation for flickering
        filter_mod = 1.0 + self.filter_lfo.generate(num_samples)
        
        # Process fire components
        rumble = self.filter1.process(fire_base, filter_mod * 0.5)
        crackles = self.crackle_filter.process(crackle, filter_mod)
        
        # Mix fire components
        signal = rumble + crackles * 0.4
        
        # Flickering amplitude
        amp_mod = 1.0 + self.amp_lfo.generate(num_samples)
        signal *= self.amplitude * amp_mod
        
        return signal

class BirdSound:
    """Optimized bird chirping sound generator - 432Hz tuned"""
    
    def __init__(self):
        self.noise_gen = NoiseGenerator()
        # Bird calls tuned to 432Hz harmonic series
        base_432 = HARMONICS_432['presence']  # 1728Hz
        self.chirp_filter = ResonantFilter(base_432, 1.5, 'bandpass')
        self.amplitude = 0.1
        self.active = True
        
        # Chirp timing
        self.next_chirp_time = SAMPLE_RATE * random.uniform(2, 5)
        self.samples_processed = 0
        
        # Pre-generate chirp envelope
        self.chirp_duration = int(SAMPLE_RATE * 0.1)  # 100ms
        self.chirp_envelope = np.sin(np.pi * np.arange(self.chirp_duration) / self.chirp_duration)
        
    def generate(self, num_samples):
        """Generate bird sounds - optimized"""
        if not self.active:
            return np.zeros(num_samples)
            
        output = np.zeros(num_samples)
        
        # Check if it's time for a chirp
        if self.samples_processed >= self.next_chirp_time:
            # Generate a chirp
            chirp_samples = min(self.chirp_duration, num_samples)
            
            # Generate noise burst
            noise = self.noise_gen.white_noise(chirp_samples) * 0.3
            
            # Apply envelope
            noise *= self.chirp_envelope[:chirp_samples]
            
            # Filter for bird-like sound
            filtered = self.chirp_filter.process(noise)
            
            # Add to output
            output[:chirp_samples] = filtered * self.amplitude
            
            # Schedule next chirp
            self.samples_processed = chirp_samples
            self.next_chirp_time = SAMPLE_RATE * random.uniform(1.5, 4)
        else:
            self.samples_processed += num_samples
            
        return output

class NatureSynthesizer:
    """Main nature synthesizer combining all elements"""
    
    def __init__(self):
        self.is_running = False
        self.stream = None
        
        # Fade-in on startup to prevent pops
        self.startup_fade = 0.0
        self.startup_fade_rate = 1.0 / (SAMPLE_RATE * 0.5)  # 0.5 second fade-in
        
        # Initialize all nature voices
        self.ocean = OceanWaves()
        self.wind = WindSound()
        self.rain = RainSound()
        self.fire = FireSound()
        self.birds = BirdSound()
        
        # Background ambience - tuned to 432Hz low-mid
        self.ambience = NatureVoice("Ambience", 'pink', HARMONICS_432['low_mid'], 0.5)
        self.ambience.amplitude = 0.1
        
        # Current scene
        self.current_scene = NatureScene.OCEAN
        self.transition_time = 0
        self.transition_duration = SAMPLE_RATE * 3  # 3 second transitions
        
        # Scene presets with reduced amplitudes
        self.scene_presets = {
            NatureScene.OCEAN: {
                'ocean': 0.3, 'wind': 0.1, 'rain': 0.0, 
                'fire': 0.0, 'birds': 0.0, 'ambience': 0.05
            },
            NatureScene.FOREST: {
                'ocean': 0.0, 'wind': 0.05, 'rain': 0.0,
                'fire': 0.0, 'birds': 0.2, 'ambience': 0.15
            },
            NatureScene.RAIN: {
                'ocean': 0.0, 'wind': 0.05, 'rain': 0.35,
                'fire': 0.0, 'birds': 0.0, 'ambience': 0.1
            },
            NatureScene.FIRE: {
                'ocean': 0.0, 'wind': 0.02, 'rain': 0.0,
                'fire': 0.3, 'birds': 0.0, 'ambience': 0.1
            },
            NatureScene.WIND: {
                'ocean': 0.0, 'wind': 0.35, 'rain': 0.0,
                'fire': 0.0, 'birds': 0.0, 'ambience': 0.1
            },
            NatureScene.CAVE: {
                'ocean': 0.0, 'wind': 0.02, 'rain': 0.0,
                'fire': 0.0, 'birds': 0.0, 'ambience': 0.25
            },
            NatureScene.NIGHT: {
                'ocean': 0.0, 'wind': 0.05, 'rain': 0.0,
                'fire': 0.05, 'birds': 0.0, 'ambience': 0.2
            },
            NatureScene.STORM: {
                'ocean': 0.15, 'wind': 0.25, 'rain': 0.2,
                'fire': 0.0, 'birds': 0.0, 'ambience': 0.05
            }
        }
        
        # Current and target amplitudes for smooth transitions
        self.current_amps = self.scene_presets[self.current_scene].copy()
        self.target_amps = self.current_amps.copy()
        
        # Master volume
        self.master_volume = 0.7
        
        # Reverb parameters
        self.reverb_buffer_size = int(SAMPLE_RATE * 0.15)
        self.reverb_buffer_l = np.zeros(self.reverb_buffer_size)
        self.reverb_buffer_r = np.zeros(self.reverb_buffer_size)
        self.reverb_index = 0
        self.reverb_mix = 0.3
        
        # Auto-evolution parameters
        self.auto_evolve = True
        self.evolution_timer = 0
        self.evolution_interval = SAMPLE_RATE * random.uniform(20, 40)  # 20-40 seconds
        
    def audio_callback(self, outdata, frames, time_info, status):
        """Optimized audio callback with click prevention"""
        if status:
            pass  # Don't spam console
            
        try:
            # Apply startup fade-in
            if self.startup_fade < 1.0:
                self.startup_fade = min(1.0, self.startup_fade + frames * self.startup_fade_rate)
            
            # Update evolution timer less frequently
            if self.auto_evolve and frames > 0:
                self.evolution_timer += frames
                if self.evolution_timer >= self.evolution_interval:
                    self.evolution_timer = 0
                    self.evolution_interval = SAMPLE_RATE * random.uniform(30, 60)
                    self.evolve_scene()
            
            # Smooth amplitude transitions
            transition_speed = 0.002  # Slower transitions
            for key in self.current_amps:
                target = self.target_amps[key]
                current = self.current_amps[key]
                if abs(target - current) > 0.001:
                    # Exponential smoothing
                    self.current_amps[key] = current * 0.998 + target * 0.002
                    
            # Generate sounds from active voices only
            mixed = np.zeros(frames, dtype=np.float32)
            
            # Generate each voice with safety checks
            try:
                if self.current_amps['ocean'] > 0.01:
                    ocean_out = self.ocean.generate(frames)
                    if len(ocean_out) == frames:
                        mixed += ocean_out * self.current_amps['ocean']
                        
                if self.current_amps['wind'] > 0.01:
                    wind_out = self.wind.generate(frames)
                    if len(wind_out) == frames:
                        mixed += wind_out * self.current_amps['wind']
                        
                if self.current_amps['rain'] > 0.01:
                    rain_out = self.rain.generate(frames)
                    if len(rain_out) == frames:
                        mixed += rain_out * self.current_amps['rain']
                        
                if self.current_amps['fire'] > 0.01:
                    fire_out = self.fire.generate(frames)
                    if len(fire_out) == frames:
                        mixed += fire_out * self.current_amps['fire']
                        
                if self.current_amps['birds'] > 0.01:
                    birds_out = self.birds.generate(frames)
                    if len(birds_out) == frames:
                        mixed += birds_out * self.current_amps['birds']
                        
                if self.current_amps['ambience'] > 0.01:
                    amb_out = self.ambience.generate(frames)
                    if len(amb_out) == frames:
                        mixed += amb_out * self.current_amps['ambience']
            except Exception:
                # If any voice fails, continue with others
                pass
            
            # Apply master volume with startup fade
            mixed *= self.master_volume * self.startup_fade * 0.5  # Overall reduction
            
            # Soft limiting with gentler curve
            mixed = np.tanh(mixed * 0.5) * 1.5
            
            # Apply a slight high-pass filter to remove DC
            if not hasattr(self, 'hp_state'):
                self.hp_state = 0.0
            
            for i in range(len(mixed)):
                hp_out = mixed[i] - self.hp_state
                self.hp_state = mixed[i] * 0.01  # Very gentle HP
                mixed[i] = hp_out
            
            # Ensure no NaN or Inf values
            mixed = np.nan_to_num(mixed, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Clip to safe range
            mixed = np.clip(mixed, -0.95, 0.95)
            
            # Output to both channels (mono to stereo)
            outdata[:, 0] = mixed
            outdata[:, 1] = mixed
            
        except Exception:
            # In case of any error, output silence
            outdata.fill(0)
        
    def set_scene(self, scene: NatureScene):
        """Change to a specific scene"""
        self.current_scene = scene
        self.target_amps = self.scene_presets[scene].copy()
        print(f"🌍 Scene changed to: {scene.value}")
        
    def evolve_scene(self):
        """Gradually evolve the current scene"""
        # Randomly adjust current scene parameters
        for key in self.target_amps:
            # Small random changes
            change = random.uniform(-0.1, 0.1)
            self.target_amps[key] = max(0, min(1, self.target_amps[key] + change))
            
        # Occasionally change filter parameters to 432Hz harmonics
        if random.random() > 0.7:
            # Use 432Hz harmonic frequencies
            harmonic_freqs = list(HARMONICS_432.values())
            self.ocean.filter1.cutoff = random.choice([HARMONICS_432['sub_bass'], HARMONICS_432['bass']])
            self.wind.filter1.cutoff = random.choice([HARMONICS_432['low_mid'], HARMONICS_432['mid']])
            self.rain.filter1.cutoff = random.choice([HARMONICS_432['presence'], HARMONICS_432['brilliance']])
            
        # Change modulation rates
        if random.random() > 0.8:
            self.ocean.filter_lfo.frequency = random.uniform(0.05, 0.1)
            self.wind.gust_lfo.frequency = random.uniform(0.02, 0.05)
            
    def randomize_scene(self):
        """Create a completely random scene mix"""
        total = 0
        for key in self.target_amps:
            self.target_amps[key] = random.random()
            total += self.target_amps[key]
            
        # Normalize to reasonable level
        if total > 0:
            for key in self.target_amps:
                self.target_amps[key] = self.target_amps[key] / total
                
        print("🎲 Randomized scene created")
        
    def start(self):
        """Start the synthesizer"""
        if not self.is_running:
            try:
                self.stream = sd.OutputStream(
                    samplerate=SAMPLE_RATE,
                    blocksize=1024,  # Increased from BUFFER_SIZE for better stability
                    channels=CHANNELS,
                    callback=self.audio_callback,
                    dtype='float32',
                    latency='high'  # Use high latency for stability
                )
                self.stream.start()
                self.is_running = True
                print("🌿 Nature Noise Synthesizer Started (432Hz Tuned)")
                print("=" * 50)
                print("Controls:")
                print("  'q' - Quit")
                print("  '1-8' - Select scenes:")
                print("    1: Ocean  2: Forest  3: Rain    4: Fire")
                print("    5: Wind   6: Cave    7: Night   8: Storm")
                print("  'r' - Random scene mix")
                print("  'e' - Evolve current scene")
                print("  'a' - Toggle auto-evolution")
                print("  '+/-' - Volume up/down")
                print("=" * 50)
                print(f"Starting with: {self.current_scene.value}")
                print("Auto-evolution: ON")
                print("All resonances tuned to 432Hz harmonic series")
                
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

def main():
    """Main function"""
    synth = NatureSynthesizer()
    
    try:
        # Check for keyboard input capability
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
            use_keyboard = False
            print("Note: Real-time keyboard control not available. Using simple input.")
            
        synth.start()
        
        if use_keyboard:
            while True:
                key = get_key()
                
                if key == 'q':
                    break
                elif key == '1':
                    synth.set_scene(NatureScene.OCEAN)
                elif key == '2':
                    synth.set_scene(NatureScene.FOREST)
                elif key == '3':
                    synth.set_scene(NatureScene.RAIN)
                elif key == '4':
                    synth.set_scene(NatureScene.FIRE)
                elif key == '5':
                    synth.set_scene(NatureScene.WIND)
                elif key == '6':
                    synth.set_scene(NatureScene.CAVE)
                elif key == '7':
                    synth.set_scene(NatureScene.NIGHT)
                elif key == '8':
                    synth.set_scene(NatureScene.STORM)
                elif key == 'r':
                    synth.randomize_scene()
                elif key == 'e':
                    synth.evolve_scene()
                    print("🌊 Scene evolved")
                elif key == 'a':
                    synth.auto_evolve = not synth.auto_evolve
                    state = "ON" if synth.auto_evolve else "OFF"
                    print(f"🔄 Auto-evolution: {state}")
                elif key == '+':
                    synth.master_volume = min(1.0, synth.master_volume + 0.1)
                    print(f"🔊 Volume: {int(synth.master_volume * 100)}%")
                elif key == '-':
                    synth.master_volume = max(0.1, synth.master_volume - 0.1)
                    print(f"🔊 Volume: {int(synth.master_volume * 100)}%")
        else:
            # Simple input loop
            while True:
                try:
                    cmd = input("\nCommand (q/1-8/r/e/a/+/-): ").strip().lower()
                    if cmd == 'q':
                        break
                    elif cmd in '12345678':
                        scenes = [
                            NatureScene.OCEAN, NatureScene.FOREST, NatureScene.RAIN,
                            NatureScene.FIRE, NatureScene.WIND, NatureScene.CAVE,
                            NatureScene.NIGHT, NatureScene.STORM
                        ]
                        synth.set_scene(scenes[int(cmd) - 1])
                    elif cmd == 'r':
                        synth.randomize_scene()
                    elif cmd == 'e':
                        synth.evolve_scene()
                        print("🌊 Scene evolved")
                    elif cmd == 'a':
                        synth.auto_evolve = not synth.auto_evolve
                        state = "ON" if synth.auto_evolve else "OFF"
                        print(f"🔄 Auto-evolution: {state}")
                    elif cmd == '+':
                        synth.master_volume = min(1.0, synth.master_volume + 0.1)
                        print(f"🔊 Volume: {int(synth.master_volume * 100)}%")
                    elif cmd == '-':
                        synth.master_volume = max(0.1, synth.master_volume - 0.1)
                        print(f"🔊 Volume: {int(synth.master_volume * 100)}%")
                except KeyboardInterrupt:
                    break
                    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        synth.stop()
        print("Thank you for experiencing nature sounds! 🌿")

if __name__ == "__main__":
    main()
