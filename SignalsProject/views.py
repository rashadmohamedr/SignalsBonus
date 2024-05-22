"""
Routes and views for the flask application.
"""

from SignalsProject import app
from datetime import datetime
from flask import render_template,request
import numpy as np
from bokeh.layouts import column
from bokeh.plotting import figure, show, output_notebook
from bokeh.embed import components
import matplotlib.pyplot as plt
import base64
from io import BytesIO

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def home():
    # Define the input signal x(t) (assuming a sine wave for simplicity)
    Fs = 44000  # Sampling rate
    t = np.arange(0, 0.01, 1/Fs)  # Time vector for 0.01 second duration
    f0 = 1000  # Fundamental frequency (Hz)
    x_t = np.sin(2 * np.pi * f0 * t)  # 1000 Hz sine wave

    # Define the amplifier system equation as a lambda function
    amplifier_system = lambda x: 20 * x + 0.02 * x**2 + 0.01 * x**3

    # Compute the output signal Y(t)
    y_t = amplifier_system(x_t)

    # Calculate the FFT of the output signal
    fft_y = np.fft.fft(y_t)

    # Calculate the square of the amplitude of the fundamental frequency
    fundamental_amplitude_squared = np.abs(fft_y[round(f0/Fs * len(t))])**2

    # Calculate the sum of squares of harmonics (up to the 10th harmonic)
    harmonics_range = np.arange(2, 11)  # Note: Python uses 0-based indexing
    sum_of_squares_harmonics = np.sum(np.abs(fft_y[np.round((harmonics_range * f0)/Fs * len(t)).astype(int)])**2)
    #arrays used as indices must be of integer (or boolean) type
    # Calculate the distortion percentage
    distortion_percentage = 100 * (sum_of_squares_harmonics / fundamental_amplitude_squared)

    # Prepare data for Chart.js
    chart_data = {
        'time': t.tolist(),
        'input_signal': x_t.tolist(),
        'output_signal': y_t.tolist(),
        'frequencies': (Fs * np.arange(0, len(fft_y)/2) / len(fft_y))[:100].tolist(),
        'magnitudes': np.abs(fft_y[:100]).tolist()
    }
    if request.method == 'POST':
        f0 = float(request.form.get('frequency'))
        Fs = float(request.form.get('samplingFrequency'))

        # --- Your distortion calculation logic ---
        t = np.arange(0, 0.01, 1/Fs)  # Time vector for 0.01 second duration
        x_t = np.sin(2 * np.pi * f0 * t)  # 1000 Hz sine wave

        # Define the amplifier system equation as a lambda function
        amplifier_system = lambda x: 20 * x + 0.02 * x**2 + 0.01 * x**3

        # Compute the output signal Y(t)
        y_t = amplifier_system(x_t)

        # Calculate the FFT of the output signal
        fft_y = np.fft.fft(y_t)

        # Calculate the square of the amplitude of the fundamental frequency
        fundamental_amplitude_squared = np.abs(fft_y[round(f0/Fs * len(t))])**2

        # Calculate the sum of squares of harmonics (up to the 10th harmonic)
        harmonics_range = np.arange(2, 11)  # Note: Python uses 0-based indexing
        sum_of_squares_harmonics = np.sum(np.abs(fft_y[np.round((harmonics_range * f0)/Fs * len(t)).astype(int)])**2)
        #arrays used as indices must be of integer (or boolean) type
        # Calculate the distortion percentage
        distortion_percentage = 100 * (sum_of_squares_harmonics / fundamental_amplitude_squared)

        # Prepare data for Chart.js
        chart_data = {
            'time': t.tolist(),
            'input_signal': x_t.tolist(),
            'output_signal': y_t.tolist(),
            'frequencies': (Fs * np.arange(0, len(fft_y)/2) / len(fft_y))[:100].tolist(),
            'magnitudes': np.abs(fft_y[:100]).tolist()
        }
        # --- End of distortion calculation ---
    return render_template('index.html', 
        title='Home Page', chart_data=chart_data,
                           distortion_percentage=f'{distortion_percentage:.10f}%')

@app.route('/contact')
def contact():
    """Renders the contact page."""
    return render_template(
        'contact.html',
        title='Contact',
        year=datetime.now().year,
        message='Your contact page.'
    )

@app.route('/about')
def about():
    """Renders the about page."""
    return render_template(
        'about.html',
        title='About',
        year=datetime.now().year,
        message='Your application description page.'
    )
