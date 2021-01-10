# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 12:18:05 2020

@author: BMCQ
"""

#DIFFERENT FILTER DESIGNS


import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import signal
from scipy.io import wavfile

#-------------------------------------------------------------------------------
#                            QUESTION 1
#-------------------------------------------------------------------------------

#plots amplitude vs. time(samples), spectrogram and the FFT of unprocessed audio .wav file
def analyze_audio(x):
    '''INPUT: "x" in quotations=audio file wanting to plot waveform/FFT
    OUPUT: plot of the unprocessed audio file, spectrogram, FFT '''
  
    audio_name = x # Audio File
    fs, audio_data = wavfile.read(audio_name) #sample rate in Hz/audio data file
    #print(fs)

    #find spectrum
    n = len(audio_data) 
    Audio_freq = scipy.fft.fft(audio_data)
    #print(n)
    
    Audio_freq = Audio_freq[0:int(np.ceil((n+1)/2.0))] #Half of the spectrum
    MagFreq = np.abs(Audio_freq) # Magnitude
    MagFreq = MagFreq / float(n)
    
    #Spectrogram of audio file
    N = 1024 #Number of points in the fft
    f, t, S = scipy.signal.spectrogram(audio_data, fs,window = scipy.signal.blackman(N),nfft=N) #finds fs, time and windowing of audio
    
    plt.figure()
    plt.pcolormesh(t, f,10*np.log10(S)) # dB spectrogram
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrogram of {}'.format(audio_name))
    plt.show()
    plt.savefig('spectrogram{}.png'.format(audio_name))

    #waveform of audio file
    plt.figure()
    plt.title("{} in time".format(audio_name))
    plt.xlabel("samples(48kHz/sec)")
    plt.ylabel("amplitude")
    plt.plot(audio_data)
    plt.show()
    plt.savefig('wavefile{}.png'.format(audio_name))
    
    #FFT of audio file
    plt.figure()
    plt.title("FFT of {}".format(audio_name))
    plt.ylabel("amplitude")
    plt.plot(np.abs(MagFreq)) #actual frequency = 48kHz* len(n)/MagFreq
    plt.show()
    plt.savefig('FFT of {}.png'.format(audio_name))



#-------------------------------------------------------------------------------
#                            QUESTION 2
#-------------------------------------------------------------------------------
#LOWPASS FILTER
def butter_lowpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# Filter parameters
fs = 48000       # sample rate, Hz
cutoff = 3000  # desired cutoff frequency of the filter, Hz

# Get the filter coefficients so we can check its frequency response for several orders
for order in [2,10,20]:
    b, a = butter_lowpass(cutoff, fs, order=order)

# Plot the frequency response
    plt.figure(4)   
    w, h = scipy.signal.freqz(b, a, worN=20000) #coefficients of the transfer function
    plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b',label="order=%d"%order)
plt.axvline(cutoff, color='k')
plt.xlim(0, 0.5*fs)
plt.title("Butterworth Lowpass Filter Frequency Response")
plt.xlabel('Frequency(Hz)')
plt.ylabel('Gain (dB)')
plt.grid(True)
plt.show()
plt.savefig('LPfreqresponse.png')


#HIGHPASS FILTER
def butter_highpass(cutoff,fs,order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a# Filter parameters

#filter parameters
fs = 48000       # sample rate, Hz
cutoff = 7000  # cutoff frequency of the filter, Hz

# Get the filter coefficients so we can check its frequency response for several orders
for order in [2,10,20]:
    b, a = butter_highpass(cutoff, fs, order=order)

# Plot the frequency response
    plt.figure(5)   
    w, h = scipy.signal.freqz(b, a, worN=20000) #coefficients of the transfer function
    plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b',label="order=%d"%order,color='red')
plt.axvline(cutoff, color='k')
plt.xlim(0, 0.5*fs)
plt.title("Butterworth Highpass Filter Frequency Response")
plt.xlabel('Frequency(Hz)')
plt.ylabel('Gain (dB)')
plt.grid(True)
plt.show()
plt.savefig('HPfreqresponse.png')

    
#BANDPASS FILTER
def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    return b, a

# Filter parameters
fs = 48000       # sample rate, Hz
lowcut= 5000  # desired cutoff frequency of the filter, Hz
highcut=15000

# Get the filter coefficients so we can check its frequency response for several orders
for order in [2,10,20]:
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)


# Plot the frequency response
    plt.figure(6)   
    w, h = scipy.signal.freqz(b, a, worN=20000) #coefficients of the transfer function
    plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b',label="order=%d"%order,color='green')
plt.axvline(lowcut, color='k')
plt.axvline(highcut, color='k')
plt.xlim(0, 0.5*fs)
plt.title("Butterworth Bandpass Filter Frequency Response")
plt.xlabel('Frequency(Hz)')
plt.ylabel('Gain (dB)')
plt.grid(True)
plt.show()
plt.savefig('BPfreqresponse.png')

#Infinite Impulse Response(IIR) LP filter
def IIRbutter_LPfilter(audio,order,cutoff):
    '''INPUT: audio='file.wav' (need quotes) to filter, and the lowpass cutoff frequency
    OUPUT: save filtered audio file as .wav'''
    audio=audio
    fs,x=wavfile.read(audio)
    b, a = butter_lowpass(cutoff, fs, order)
    x= signal.lfilter(b,a, x)
    wavfile.write('{} audioIIRbutter_LPfilter.wav'.format(audio), fs, x.astype(np.int16))

#Infinite Impulse Response(IIR) HP filter
def IIRbutter_HPfilter(audio,order,cutoff):
    '''INPUT: audio='file.wav' (need quotes) to filter, and the highpass cutoff frequency
    OUPUT: save filtered audio file as .wav'''
    audio=audio
    fs,x=wavfile.read(audio)
    b, a = butter_highpass(cutoff, fs, order)
    x= signal.lfilter(b,a, x)
    wavfile.write('{} audioIIRbutter_HPfilter.wav'.format(audio), fs, x.astype(np.int16))

#Infinite Impulse Response(IIR) BP filter
def IIRbutter_BPfilter(audio,order,lowcut,highcut):
    '''INPUT: audio='file.wav' (need quotes) to filter, and the lowpass/highpass cutoff frequency
    OUPUT: save filtered audio file as .wav'''
    audio=audio
    fs,x=wavfile.read(audio)
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    x= signal.lfilter(b,a, x)
    wavfile.write('{} audioIIRbutter_BPfilter.wav'.format(audio), fs, x.astype(np.int16))


#Plot of Window function and its Frequency response
#based on https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.blackman.html#scipy.signal.windows.blackman

for M in [20,50,100]:
    window = signal.blackman(M)
    plt.figure(7)
    plt.plot(window, label=M)
    plt.legend()
    plt.title("Blackman window")
    plt.ylabel("Amplitude")
    plt.xlabel("Sample")
    plt.show()
    plt.savefig("Blackman window function.png")

    plt.figure(8)
    A = scipy.fft(window, 2048) / (len(window)/2.0)
    freq = np.linspace(-0.5, 0.5, len(A))
    response = np.abs(scipy.fft.fftshift(A / abs(A).max()))
    response = 20*np.log10(np.maximum(response, 1e-10))
    plt.plot(freq, response,label=M)
    plt.legend()
    plt.axis([-0.5, 0.5, -120, 0])
    plt.title("Frequency response of the Blackman window")
    plt.ylabel("Normalized magnitude [dB]")
    plt.xlabel("Normalized frequency [cycles per sample]")
    plt.show()
    plt.savefig("blackman freq response.png")
    
#Finite Impulse Resonse (FIR) LP filter
def FIR_LPfilter(numtaps,audio,cutoff):
    '''INPUT: audio='file.wav' (need quotes) to filter, and the highpass cutoff frequency
    OUPUT: save filtered audio file as .wav'''
    audio=audio
    sr, x = wavfile.read(audio)     # 16-bit mono 48kHz
    b = signal.firwin(numtaps, cutoff=cutoff, fs=0.5*sr,window='blackman') 
    x = signal.lfilter(b, [1.0], x)
    wavfile.write('{} audioFIR_LPfilter.wav'.format(audio), sr, x.astype(np.int16))

#Finite Impulse Resonse HP filter
def FIR_HPfilter(numtaps,audio,cutoff):
    '''INPUT: audio='file.wav' (need quotes around file name) to filter, 
            and the highpass cutoff frequency
    OUPUT: save filtered audio file as .wav'''
    audio=audio
    sr, x = wavfile.read(audio)     # 16-bit mono 48kHz 
    b = signal.firwin(numtaps, cutoff=cutoff, fs=0.5*sr,window='blackman', pass_zero=False)
    x = signal.lfilter(b, [1.0], x)
    wavfile.write('{} audioFIRHP_filter.wav'.format(audio), sr, x.astype(np.int16))
   
def FIR_BPfilter(numtaps,audio,lowcut,highcut):
    '''INPUT: audio='file.wav' (need quotes around file name) to filter, 
            and the lowpass/highpass cutoff frequency
    OUPUT: save filtered audio file as .wav'''
    audio=audio
    sr, x = wavfile.read(audio)     # 16-bit mono 48kHz 
    b = signal.firwin(numtaps, [lowcut,highcut], fs=sr, window='blackman',pass_zero=False)
    x = signal.lfilter(b, [1.0], x)
    wavfile.write('{} audioFIR_BPfilter.wav'.format(audio), sr, x.astype(np.int16))


#-------------------------------------------------------------------------------
#                            QUESTION 3
#-------------------------------------------------------------------------------
# get the waveform,spectrogram and FFT of audio clips
# analyze_audio("Kickdrum5000Hz.wav")
# analyze_audio(("violin90hz.wav"))
analyze_audio(("flute150and15kHz.wav"))

FIR_BPfilter(601,"flute150and15kHz.wav",350,13000)
IIRbutter_BPfilter("flute150and15kHz.wav",9, 350, 11000)

#analyze the filtered audio signals
analyze_audio("flute150and15kHz.wav audioIIRbutter_BPfilter.wav")
analyze_audio("flute150and15kHz.wav audioFIR_BPfilter.wav")



