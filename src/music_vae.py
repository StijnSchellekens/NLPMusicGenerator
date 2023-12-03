import os
import note_seq
import pretty_midi
import io
import streamlit as st

from pydub import AudioSegment
from pydub.generators import Sine, Square, Pulse, Sawtooth, Triangle
from magenta.models.music_vae import TrainedModel, configs

project_root_dir = os.path.dirname(os.path.abspath(__file__))

def setModel(music_type):
    global config, checkpoint_path, temperature
    if "calm_soothing" in music_type:
        config = configs.CONFIG_MAP['cat-mel_2bar_big']
        checkpoint_path = os.path.join(project_root_dir, 'Configs', 'cat-mel_2bar_big.ckpt')
        temperature = 0.3
        waveformtype = "Sine"
        return TrainedModel(config, batch_size=1, checkpoint_dir_or_path=checkpoint_path), temperature, waveformtype
    elif "upbeat_energetic" in music_type:
        config = configs.CONFIG_MAP['hierdec-mel_16bar']
        checkpoint_path = os.path.join(project_root_dir, 'Configs', 'hierdec-mel_16bar.ckpt')
        temperature = 2.5
        waveformtype = "Sawtooth"
        return TrainedModel(config, batch_size=1, checkpoint_dir_or_path=checkpoint_path), temperature, waveformtype
    elif "ambient_relaxing" in music_type:
        config = configs.CONFIG_MAP['hierdec-mel_16bar']
        checkpoint_path = os.path.join(project_root_dir, 'Configs', 'hierdec-mel_16bar.ckpt')
        temperature = 0.01
        waveformtype = "Sine"
        return TrainedModel(config, batch_size=1, checkpoint_dir_or_path=checkpoint_path), temperature, waveformtype


# Use Magenta to generate a MIDI file
def generateMidi(music_type):
    music_vae, temperature, waveformtype = setModel(music_type)
    st.warning("Generating a song with the following parameters:")
    st.warning(f"Music Type: {music_type}")
    st.warning(f"Temperature: {temperature}")
    st.warning(f"Waveform Type: {waveformtype}")
    print(temperature)
    return music_vae.sample(n=1, length=80, temperature=temperature)[0], waveformtype

def apply_reverb(sound, delay_ms=50, decay=0.5):
    reverb = sound
    for _ in range(3):
        sound = sound - delay_ms
        sound = sound - decay
        reverb = reverb.overlay(sound)
    return reverb

# Use PyDub to generate a WAV file
def getSample(music_type):
    generated_seq, waveform = generateMidi(music_type)

    midi_io = io.BytesIO()
    note_seq.sequence_proto_to_pretty_midi(generated_seq).write(midi_io)
    midi_io.seek(0)

    midi_data = pretty_midi.PrettyMIDI(midi_io)

    combined_audio = AudioSegment.silent(duration=10)

    for note in midi_data.instruments[0].notes:
        frequency = pretty_midi.note_number_to_hz(note.pitch)
        duration = int((note.end - note.start) * 1000)
        if waveform == "Sine":
            note_audio = Sine(frequency).to_audio_segment(duration=duration)
            note_audio = apply_reverb(note_audio)
        elif waveform == "Sawtooth":
            note_audio = Sawtooth(frequency).to_audio_segment(duration=duration)
        else:
            note_audio = Sine(frequency).to_audio_segment(duration=duration)

        combined_audio += note_audio

    wav_io = io.BytesIO()
    combined_audio.export(wav_io, format="wav")
    wav_io.seek(0)

    return wav_io