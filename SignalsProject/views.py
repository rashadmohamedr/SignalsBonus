
from SignalsProject import app
from datetime import datetime
from flask import render_template, request
import numpy as np
import io
import librosa

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def home():
    distortion_percentage = None
    chart_data = {}  
    if request.method == 'POST':
        if 'audioFile' in request.files:
            audio_file = request.files['audioFile']
            if audio_file.filename != '':
                try:
                    data, fs = librosa.load(io.BytesIO(audio_file.read()), sr=None)

                    # Check dimensions ONLY if you need to extract a channel
                    if len(data.shape) > 1:
                        data = data[:, 0] # Take the first channel

                    # Perform distortion calculation (make sure these variables are defined)
                    f0 = 1000  # Assuming a fundamental frequency for calculation
                    fft_y = np.fft.fft(data)
                    fundamental_amplitude_squared = np.abs(fft_y[round(f0 / fs * len(data))]) ** 2
                    harmonics_range = np.arange(2, 11)
                    sum_of_squares_harmonics = np.sum(
                        np.abs(fft_y[np.round((harmonics_range * f0) / fs * len(data)).astype(int)]) ** 2)
                    distortion_percentage = 100 * (sum_of_squares_harmonics / fundamental_amplitude_squared)

                    # Apply distortion (make sure y_t is calculated correctly)
                    amplifier_system = lambda x: 20 * x + 0.02 * x**2 + 0.01 * x**3
                    y_t = amplifier_system(data)  # Apply to the loaded audio data

                    # Prepare chart data
                    chart_data = {
                        'time': (np.arange(0, len(data)) / fs).tolist(),
                        'input_signal': data.tolist(),
                        'output_signal': y_t.tolist(),  
                        'frequencies': (fs * np.arange(0, len(fft_y) / 2) / len(fft_y))[:100].tolist(),
                        'magnitudes': np.abs(fft_y[:100]).tolist()
                    }

                except Exception as e:
                    error_message = f"Error processing audio: {str(e)}"
                    return render_template('index.html', error=error_message)
        else:
            f0 = float(request.form.get('frequency'))
            Fs = float(request.form.get('samplingFrequency'))
            t = np.arange(0, 0.01, 1/Fs)
            x_t = np.sin(2 * np.pi * f0 * t)
            amplifier_system = lambda x: 20 * x + 0.02 * x**2 + 0.01 * x**3
            y_t = amplifier_system(x_t)
            fft_y = np.fft.fft(y_t)
            fundamental_amplitude_squared = np.abs(fft_y[round(f0/Fs * len(t))])**2
            harmonics_range = np.arange(2, 11)
            sum_of_squares_harmonics = np.sum(np.abs(fft_y[np.round((harmonics_range * f0)/Fs * len(t)).astype(int)])**2)
            distortion_percentage = 100 * (sum_of_squares_harmonics / fundamental_amplitude_squared)
            chart_data = {
                'time': t.tolist(),
                'input_signal': x_t.tolist(),
                'output_signal': y_t.tolist(),
                'frequencies': (Fs * np.arange(0, len(fft_y)/2) / len(fft_y))[:100].tolist(),
                'magnitudes': np.abs(fft_y[:100]).tolist()
            }
    return render_template('index.html',
        title='Home Page', chart_data=chart_data,
                           distortion_percentage=f'{distortion_percentage:.10f}%' if distortion_percentage is not None else None)


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
